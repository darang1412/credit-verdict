"""
Phase 7 - Explainability with SHAP.

Two views of the same underlying question - "why does the model think this
applicant is risky?" - answered with Shapley values rather than an ad hoc
importance heuristic:

  - GLOBAL: which features matter most across the whole validation set
    (a beeswarm/summary plot). This is checked against the hypotheses
    written down in Phase 3's EDA (notebooks/01_eda_application.ipynb),
    before any model existed - agreement is a sanity check, disagreement is
    worth explaining, not silently accepting.
  - LOCAL: for individual high-risk applicants, which specific features
    pushed THIS prediction toward default, and by how much. The top-3 such
    factors per high-risk applicant are saved to disk for Phase 8's Adverse
    Action generator to retrieve regulatory/policy text against.

TreeExplainer is used (not the generic, model-agnostic SHAP explainer)
because it is exact and fast specifically for tree ensembles like LightGBM -
it walks the actual tree structure to compute contributions, rather than
approximating them by sampling feature subsets.

IMPORTANT SEAM BETWEEN PHASE 6 AND PHASE 7: SHAP explains the raw LightGBM
model's output (log-odds / margin space), not the isotonic-calibrated
probability from Phase 6. Isotonic calibration is a separate, non-parametric
step function wrapped AROUND the tree model's output - it has no tree
structure for TreeExplainer to attach to. So these explanations answer "why
did the tree model score this applicant as risky," and calibration is a
separate step afterward that only rescales the resulting probability,
without changing which features drove the decision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

# train.py lives in ml/modeling/, a sibling directory of this script
# (ml/explain/) - Python only auto-adds a script's OWN directory to the
# import path, so ml/modeling/ has to be added explicitly to reuse train.py's
# load_features and split_data rather than duplicating them a second time.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modeling"))
from train import BASE_DIR, load_features, split_data  # noqa: E402

FIGURES_DIR = BASE_DIR / "reports" / "figures"
OUTPUT_DIR = BASE_DIR / "reports"

# Same finalized hyperparameters as Phase 5/6's LightGBM (see train.py and
# calibration.py). Reconstructed here by retraining deterministically
# (same random_state, same split logic) rather than importing shared code -
# continuing the pattern calibration.py already established, rather than
# reopening already-committed Phase 5/6 files to extract a shared module.
LGB_PARAMS = dict(
    objective="binary",
    n_estimators=441,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
)

# Placeholder rule for choosing which validation applicants count as
# "high-risk" for the LOCAL explanations and the Phase 8 handoff below -
# NOT the real business approval/decline cutoff, which is a deliberate
# tradeoff decision made later in Phase 10. This just needs *some* concrete
# group of clearly-risky applicants to explain and hand forward.
HIGH_RISK_PERCENTILE = 90
N_TOP_FACTORS = 3


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
    """Reproduces Phase 6's exact model + calibrator (same data splits, same
    random_state), so this script's explanations describe the model that was
    actually evaluated and calibrated, not a coincidentally similar one.

    Also returns ids_val (the SK_ID_CURR for each validation row) separately
    from X_val, since split_data() drops SK_ID_CURR as a feature column - it
    is not, and should not be, part of what the model or SHAP sees."""
    df = load_features()
    X_train, X_val, y_train, y_val = split_data(df)
    ids_val = df.loc[X_val.index, "SK_ID_CURR"].to_numpy()

    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    X_fit, X_calib, X_val = cast_categoricals(X_fit, X_calib, X_val)

    scale_pos_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
    model = lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, **LGB_PARAMS)
    model.fit(X_fit, y_fit)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(model.predict_proba(X_calib)[:, 1], y_calib)

    return model, calibrator, X_val, y_val, ids_val


def main():
    model, calibrator, X_val, y_val, ids_val = build_model_and_calibrator()
    calibrated_proba = calibrator.predict(model.predict_proba(X_val)[:, 1])

    print("Computing SHAP values via TreeExplainer (exact, tree-structure-based)...")
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_val)

    # For a binary LightGBM model, some SHAP/library version combinations
    # return one set of contributions (log-odds toward the positive class)
    # and others return contributions per class - normalize to the former,
    # since that's what both the summary plot and the per-applicant factor
    # extraction below assume.
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]

    # --- Global: which features matter most across the whole validation set ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title("SHAP global feature importance (validation set)")
    plt.tight_layout()
    global_fig_path = FIGURES_DIR / "shap_global_summary.png"
    plt.savefig(global_fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved global SHAP summary to {global_fig_path}")

    mean_abs_shap = pd.Series(
        np.abs(explanation.values).mean(axis=0), index=X_val.columns
    ).sort_values(ascending=False)
    print("\nTop 10 features by mean |SHAP value|:")
    print(mean_abs_shap.head(10))

    # --- Local: high-risk applicants, and their top-3 driving factors ---
    # A positive SHAP value here means "this feature pushed the prediction
    # TOWARD default" (increased the log-odds of class 1) - so the top
    # factors for a high-risk applicant are their largest POSITIVE SHAP
    # values, not the most negative ones (the roadmap's "top-3 negative
    # factors" refers to factors working against the applicant, not to the
    # mathematical sign of the SHAP value itself - worth being explicit
    # about, since the two readings would produce opposite selections).
    threshold = np.percentile(calibrated_proba, HIGH_RISK_PERCENTILE)
    high_risk_mask = calibrated_proba >= threshold
    high_risk_idx = np.where(high_risk_mask)[0]
    print(
        f"\n{high_risk_mask.sum()} applicants at/above the {HIGH_RISK_PERCENTILE}th "
        f"percentile of calibrated probability (threshold={threshold:.4f}) - "
        f"illustrative 'high-risk' group for local explanations, not the real "
        f"business cutoff (that's decided in Phase 10)."
    )

    # One concrete local example, saved as a static waterfall plot - SHAP's
    # "force plot" is interactive/JS-based and doesn't save cleanly as a
    # static image; the waterfall plot is the standard static equivalent for
    # a single prediction and is what actually ends up embedded in a report.
    example_idx = int(high_risk_idx[0])
    plt.figure()
    shap.plots.waterfall(explanation[example_idx], max_display=12, show=False)
    # shap.plots.waterfall sizes its own figure based on feature-name length,
    # which clips value labels ("+0.2" cut off mid-digit) when several
    # long feature names and narrow bars land in the same plot - widening it
    # after the fact fixes the clipping without changing any underlying data.
    plt.gcf().set_size_inches(11, 8)
    plt.title(f"SHAP local explanation - example high-risk applicant (SK_ID_CURR={ids_val[example_idx]})")
    plt.tight_layout()
    local_fig_path = FIGURES_DIR / "shap_local_example.png"
    plt.savefig(local_fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved example local SHAP explanation to {local_fig_path}")

    # Top-3 factors per high-risk applicant, WITH the applicant's actual raw
    # feature value alongside the SHAP contribution - Phase 8's generator
    # needs to name specifics ("debt-to-income ratio of 45%"), not just a
    # feature name, per the domain notes' "specific, not generic" standard.
    records = []
    for i in high_risk_idx:
        row_shap = explanation.values[i]
        top_feature_idx = np.argsort(row_shap)[::-1][:N_TOP_FACTORS]
        top_factors = [
            {
                "feature": X_val.columns[j],
                "applicant_value": (
                    X_val.iloc[i, j].item()
                    if pd.notnull(X_val.iloc[i, j]) and not isinstance(X_val.iloc[i, j], str)
                    else str(X_val.iloc[i, j])
                ),
                "shap_contribution": float(row_shap[j]),
            }
            for j in top_feature_idx
        ]
        records.append(
            {
                "row_index": int(i),
                "sk_id_curr": int(ids_val[i]),
                "calibrated_probability_of_default": float(calibrated_proba[i]),
                "top_factors": top_factors,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "top_shap_factors.json"
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved top-{N_TOP_FACTORS} SHAP factors for {len(records)} high-risk applicants to {output_path}")


if __name__ == "__main__":
    main()
