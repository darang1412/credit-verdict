"""
Phase 8 - Data Analysis in SQL/Postgres: score the held-out validation set
with the final calibrated model, attach each applicant's top-3 SHAP factors,
and write one row per applicant to a real Postgres table - the fact table
Phase 9's Power BI dashboard connects to directly, instead of a CSV export.

Scores X_val (not the full ~307K-row dataset): same honesty principle as
calibration.py and shap_analysis.py - these are genuinely out-of-sample
predictions, not the model grading its own training rows. Reuses
calibration.py's exact finalized hyperparameters (n_estimators=765,
num_leaves=15, min_child_samples=100) as the one authoritative "final
model" config for this project, rather than shap_analysis.py's separately
reproduced (and, on inspection, drifted - n_estimators=441, num_leaves=31)
config - that mismatch is a pre-existing reproducibility gap worth flagging
in the report, not silently fixed here.
"""

from __future__ import annotations

import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from train import BASE_DIR, load_features, split_data

N_TOP_FACTORS = 3

# Same finalized hyperparameters as Phase 6's calibration.py - see that
# file's LGB_PARAMS docstring for why these four values are the ones to
# re-sync from if Phase 5's search or split ever changes.
LGB_PARAMS = dict(
    objective="binary",
    n_estimators=765,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=100,
    random_state=42,
    n_jobs=-1,
)


def cast_categoricals(X_fit: pd.DataFrame, X_calib: pd.DataFrame, X_val: pd.DataFrame):
    X_fit = X_fit.copy()
    X_calib = X_calib.copy()
    X_val = X_val.copy()
    cat_cols = X_fit.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        X_fit[col] = X_fit[col].astype("category")
        categories = X_fit[col].cat.categories
        X_calib[col] = X_calib[col].astype(pd.CategoricalDtype(categories=categories))
        X_val[col] = X_val[col].astype(pd.CategoricalDtype(categories=categories))
    return X_fit, X_calib, X_val


def build_model_and_calibrator():
    """Reproduces Phase 6's exact model + calibrator (same splits, same
    random_state=42) so the scores written to Postgres describe the model
    that was actually evaluated and calibrated, not a coincidentally similar
    one."""
    df = load_features()
    X_train, X_val, y_train, y_val = split_data(df)
    ids_val = df.loc[X_val.index, "SK_ID_CURR"].to_numpy()
    amt_credit_val = df.loc[X_val.index, "AMT_CREDIT"].to_numpy()

    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    X_fit, X_calib, X_val = cast_categoricals(X_fit, X_calib, X_val)

    scale_pos_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
    model = lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, **LGB_PARAMS)
    model.fit(X_fit, y_fit)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(model.predict_proba(X_calib)[:, 1], y_calib)

    return model, calibrator, X_val, ids_val, amt_credit_val


def risk_segment(calibrated_pd: np.ndarray) -> np.ndarray:
    """Tertiles of the model's own calibrated probability - a model-driven
    segmentation, unlike Phase 0.5's raw-AMT_CREDIT RiskBand. That contrast
    (naive feature threshold vs. model-driven segmentation) is itself worth
    stating in an interview."""
    low_cut, high_cut = np.quantile(calibrated_pd, [1 / 3, 2 / 3])
    bands = np.where(
        calibrated_pd <= low_cut, "Low", np.where(calibrated_pd <= high_cut, "Medium", "High")
    )
    return bands


def main():
    print("Rebuilding Phase 6's final model + calibrator on X_fit/X_calib...")
    model, calibrator, X_val, ids_val, amt_credit_val = build_model_and_calibrator()

    predicted_pd = model.predict_proba(X_val)[:, 1]
    calibrated_pd = calibrator.predict(predicted_pd)
    segments = risk_segment(calibrated_pd)

    print(f"Computing SHAP TreeExplainer values for all {len(X_val):,} validation applicants...")
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_val)
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]
    shap_values = explanation.values

    # Top-3 factors = largest POSITIVE SHAP values (pushed toward default),
    # same convention as shap_analysis.py - not the most negative ones.
    top_idx = np.argsort(shap_values, axis=1)[:, ::-1][:, :N_TOP_FACTORS]
    feature_names = X_val.columns.to_numpy()
    top_factor_names = feature_names[top_idx]  # shape (n_rows, 3)

    out = pd.DataFrame(
        {
            "sk_id_curr": ids_val,
            "predicted_pd": predicted_pd,
            "calibrated_pd": calibrated_pd,
            "risk_segment": segments,
            "top_shap_factor_1": top_factor_names[:, 0],
            "top_shap_factor_2": top_factor_names[:, 1],
            "top_shap_factor_3": top_factor_names[:, 2],
            "amt_credit": amt_credit_val,
        }
    )

    # Host-side port is 5433, not the container-internal 5432 (see
    # docker-compose.yml's comment) - something else on the Windows host
    # already owns 5432, which silently shadowed Docker's port mapping.
    db_url = (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'credit_risk')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@"
        f"127.0.0.1:{os.environ.get('POSTGRES_HOST_PORT', '5433')}/"
        f"{os.environ.get('POSTGRES_DB', 'credit_risk')}"
    )
    print(f"Writing {len(out):,} scored applicants to Postgres table 'scored_applications'...")
    engine = create_engine(db_url)
    out.to_sql("scored_applications", engine, if_exists="replace", index=False)
    print("Done. Segment counts:")
    print(out["risk_segment"].value_counts())


if __name__ == "__main__":
    main()
