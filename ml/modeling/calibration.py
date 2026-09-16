"""
Phase 6 - Model Calibration.

AUC (Phase 5) only measures whether the model RANKS applicants correctly -
riskier ones score higher. It says nothing about whether "72% predicted
probability" corresponds to an actual 72% observed default rate, which is
what any real use of the number (a business cutoff, an expected-loss
estimate) actually needs.

This is expected to matter a lot here specifically, not just in theory:
Phase 5's LightGBM used scale_pos_weight (~11.4) to fix the class-imbalance
ranking problem, which works by making the training loss treat a missed
default as ~11x more costly. That deliberately distorts the model away from
predicting the true base rate - so the raw probabilities coming out of that
model are expected to be systematically miscalibrated as a direct, known
side effect of the Phase 5 choice, not a separate flaw. This script measures
that distortion and corrects it with isotonic regression.
"""

from __future__ import annotations

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from train import BASE_DIR, MLFLOW_DB_PATH, MLFLOW_EXPERIMENT, load_features, split_data

FIGURES_DIR = BASE_DIR / "reports" / "figures"

# Same finalized hyperparameters as Phase 5's LightGBM (see train.py).
# n_estimators is fixed at 441 - the manually-determined true best iteration
# from Phase 5 - rather than re-run through early stopping here, since that
# mechanism was found to be unreliable under scale_pos_weight in this
# LightGBM release (see train.py's comments). Phase 6's job is to calibrate
# that already-chosen model, not re-tune it.
LGB_PARAMS = dict(
    objective="binary",
    n_estimators=441,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
)


def cast_categoricals(
    X_fit: pd.DataFrame, X_calib: pd.DataFrame, X_val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Same treatment as train.py's LightGBM path: cast object columns to
    pandas `category` dtype so LightGBM handles them natively, with X_calib
    and X_val forced to share X_fit's exact category set so an unseen
    category becomes a real missing value (NaN) rather than an unrecognized
    category the model never learned a split for."""
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


def main():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_features()
    # X_val here is IDENTICAL to Phase 5's validation set (same function,
    # same random_state) - its AUC stays directly comparable to Phase 5's
    # reported number, and it has never been touched by either the
    # classifier's training or the calibrator's fitting below.
    X_train, X_val, y_train, y_val = split_data(df)

    # Carve a calibration-only slice out of the TRAINING portion. Fitting the
    # calibrator on the same data used to judge it would make "it's
    # calibrated now" a claim about memorization, not genuine calibration -
    # this is the same train/test leakage principle as everywhere else, just
    # applied to the calibration step specifically.
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    X_fit, X_calib, X_val = cast_categoricals(X_fit, X_calib, X_val)

    scale_pos_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
    model = lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, **LGB_PARAMS)
    model.fit(X_fit, y_fit)

    raw_val_proba = model.predict_proba(X_val)[:, 1]
    raw_calib_proba = model.predict_proba(X_calib)[:, 1]

    # Isotonic regression fits a monotonic step function mapping the raw
    # predicted probability to a calibrated one, learned only from the
    # calibration slice's true outcomes. "Monotonic" is the key constraint:
    # it can reshape the raw scores however the data demands (unlike Platt
    # scaling's fixed sigmoid-curve assumption), but it can never reverse the
    # model's ranking of applicants - which is why the AUC below is expected
    # to come out unchanged (up to floating-point noise) before vs. after.
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_calib_proba, y_calib)
    calibrated_val_proba = calibrator.predict(raw_val_proba)

    raw_auc = roc_auc_score(y_val, raw_val_proba)
    calibrated_auc = roc_auc_score(y_val, calibrated_val_proba)
    raw_brier = brier_score_loss(y_val, raw_val_proba)
    calibrated_brier = brier_score_loss(y_val, calibrated_val_proba)

    print(f"Fit rows: {len(X_fit):,}, calibration rows: {len(X_calib):,}, validation rows: {len(X_val):,}")
    print(f"AUC         - raw: {raw_auc:.4f}, calibrated: {calibrated_auc:.4f} (should match - isotonic preserves ranking)")
    print(f"Brier score - raw: {raw_brier:.4f}, calibrated: {calibrated_brier:.4f} (lower is better)")

    # Quantile (equal-frequency) bins, not equal-width: LightGBM's predicted
    # probabilities cluster in a fairly narrow band rather than spreading
    # evenly across [0, 1], so equal-width bins would leave several bins
    # nearly empty - an unstable, near-meaningless "observed rate" for them.
    # Equal-frequency bins guarantee every point on the diagram reflects a
    # similar number of real applicants.
    fig, ax = plt.subplots(figsize=(7, 7))
    CalibrationDisplay.from_predictions(
        y_val, raw_val_proba, n_bins=10, strategy="quantile",
        name=f"Raw LightGBM (Brier={raw_brier:.3f})", ax=ax,
    )
    CalibrationDisplay.from_predictions(
        y_val, calibrated_val_proba, n_bins=10, strategy="quantile",
        name=f"Isotonic-calibrated (Brier={calibrated_brier:.3f})", ax=ax,
    )
    ax.set_title(
        "Reliability diagram: raw vs. isotonic-calibrated LightGBM\n"
        "(validation set - held out from both model training and calibration fitting)"
    )
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "calibration_reliability_diagram.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved reliability diagram to {fig_path}")

    with mlflow.start_run(run_name="lightgbm_calibration"):
        mlflow.log_param("model", "LightGBM")
        mlflow.log_param("calibration_method", "isotonic")
        mlflow.log_param("n_fit_rows", len(X_fit))
        mlflow.log_param("n_calibration_rows", len(X_calib))
        mlflow.log_metric("auc_raw", raw_auc)
        mlflow.log_metric("auc_calibrated", calibrated_auc)
        mlflow.log_metric("brier_raw", raw_brier)
        mlflow.log_metric("brier_calibrated", calibrated_brier)
        mlflow.log_artifact(str(fig_path))
        mlflow.lightgbm.log_model(model, name="model")


if __name__ == "__main__":
    main()
