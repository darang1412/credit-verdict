"""
Phase 5 - Model Training & Experimentation, tracked in MLflow.

Trains two models on the Phase 4 feature table and compares them honestly on
a held-out validation split:

  - Logistic regression: an interpretable baseline. Its coefficients can be
    read directly ("for a one-unit increase in this feature, the log-odds of
    default change by X"), which is a useful sanity check independent of
    SHAP (Phase 7).
  - LightGBM: the primary candidate, chosen over a neural network because
    gradient-boosted trees are the established strong default for
    well-engineered tabular data at this scale (see the roadmap's Phase 1
    note on why this project skips deep-learning theory entirely), and
    because it handles missing values and categorical columns natively -
    which this dataset has a lot of.

Both models use class weighting, not resampling, to handle the ~8%/92% class
imbalance confirmed in Phase 3's EDA: it reweights the training loss so a
mistake on the minority (default) class counts proportionally more, using
every real row instead of discarding majority-class examples the way
downsampling would.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Resolved relative to this file, not the current working directory - the
# same class of bug bit both ml/pipeline/features.py and the EDA notebook
# earlier in this project, so every path here is anchored to the repo root
# instead of assuming a particular cwd. This file lives at
# <repo_root>/ml/modeling/train.py, three levels below the repo root.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
# MLflow 3.x puts the plain filesystem store ("./mlruns") into maintenance
# mode and refuses to use it for new tracking - a SQLite backend is the
# supported local option instead, still fully local/file-based, just backed
# by a single .db file rather than a directory of run folders.
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_EXPERIMENT = "credit-risk-phase5"

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"


def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    y = df[TARGET_COL]
    # Stratify on TARGET so both the train and validation sets keep the same
    # ~8.07% default rate as the full dataset. Without this, an ordinary
    # random split could easily land at, say, 6% default in validation just
    # by chance, on a target this imbalanced - making metrics noisier and
    # harder to compare across runs than they need to be.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_val, y_train, y_val


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    # Precision/recall/F1 require picking a decision threshold, unlike AUC
    # (which measures ranking quality independent of any cutoff). 0.5 is used
    # here only as a fixed, comparable reference point between the two
    # models - the *actual* business cutoff is a deliberate decision made
    # later, in Phase 10's approval-rate/expected-loss tradeoff analysis, not
    # something to optimize prematurely here.
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def train_logistic_regression(X_train, y_train, X_val, y_val) -> dict:
    """Interpretable baseline. Needs an explicit preprocessing pipeline that
    LightGBM (below) does not: sklearn's LogisticRegression can't consume raw
    strings or NaN, so categoricals are one-hot encoded and missing values
    imputed here, specifically for this model."""
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()

    # Logistic regression is a gradient-based, distance-sensitive model: a raw
    # column like AMT_CREDIT (values in the hundreds of thousands) and a
    # ratio like CREDIT_INCOME_RATIO (values near 1) on wildly different
    # scales make the optimizer take tiny, slow steps on one and huge,
    # unstable ones on the other - StandardScaler (mean 0, unit variance)
    # puts every numeric feature on comparable footing, which is why the
    # unscaled version above failed to converge within 1000 iterations.
    # LightGBM needs none of this: tree splits only compare a feature to
    # itself at different thresholds, so scale is irrelevant to it.
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    # class_weight="balanced" reweights the loss function so mistakes on the
    # minority (default) class count proportionally more - roughly the same
    # correction scale_pos_weight applies for LightGBM below. Without it, a
    # linear model trained on 92%-non-default data can minimize its average
    # loss by mostly just predicting "no default" for everyone.
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "classify",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            ),
        ]
    )

    with mlflow.start_run(run_name="logistic_regression_baseline"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("n_features_raw", X_train.shape[1])

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_proba)
        mlflow.log_metrics(metrics)
        # MLflow 3.x defaults sklearn model logging to `skops` serialization,
        # which refuses to save this pipeline because it can't yet vouch for
        # every type inside it (numpy.dtype objects held by the
        # ColumnTransformer). This is a real security feature - skops is
        # deliberately more cautious than pickle about arbitrary code
        # execution on load - but for a local, single-user portfolio project
        # with no untrusted model files in play, cloudpickle is the pragmatic
        # choice over hand-auditing/whitelisting internal sklearn types.
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

    return metrics


def train_lightgbm(X_train, y_train, X_val, y_val) -> dict:
    """Primary candidate. Categorical columns are cast to pandas' `category`
    dtype - LightGBM detects these automatically and splits on them directly
    (grouping category values by how they relate to the target), instead of
    needing them exploded into many one-hot columns first. It also handles
    NaN internally, learning per split which direction a missing value should
    go - so none of the imputation used for logistic regression above is
    needed, or wanted: imputing would throw away the "this was missing"
    signal Phase 3's EDA showed is itself informative for several columns
    (EXT_SOURCE missingness, bureau history)."""
    X_train = X_train.copy()
    X_val = X_val.copy()
    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
    for col in categorical_cols:
        X_train[col] = X_train[col].astype("category")
        # Val must share train's exact category set - if a category appears
        # only in validation, casting it against train's categories turns it
        # into NaN (a real "unknown category", which LightGBM handles like
        # any other missing value) rather than silently creating a category
        # the trained model never learned a split for.
        X_val[col] = X_val[col].astype(pd.CategoricalDtype(categories=X_train[col].cat.categories))

    # scale_pos_weight tells LightGBM's loss function how much more a mistake
    # on a default (the minority class) should count, set to the actual
    # majority:minority ratio observed in the training split rather than a
    # guessed round number - the LightGBM-native equivalent of
    # class_weight="balanced" above.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # LightGBM's built-in early-stopping callback (lgb.early_stopping) was
    # found, empirically, to select the WORST validation round (iteration 1
    # out of hundreds, on a monotonically-improving AUC curve) as the "best"
    # one - but only when scale_pos_weight is set; removing scale_pos_weight
    # alone restores correct behavior. That points to a real bug in this
    # LightGBM release (4.7.0)'s early-stopping direction detection when
    # sample weighting is active, not a modeling mistake. Rather than trust
    # the automatic mechanism, the true best round is found manually from the
    # validation AUC curve, then a second, right-sized model is trained on
    # exactly that many rounds - this is the safe fallback the roadmap's
    # LightGBM docs describe manual early stopping as being built on anyway.
    max_rounds = 800
    probe_model = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        n_estimators=max_rounds,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
    )
    probe_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc")
    val_auc_curve = probe_model.evals_result_["valid_0"]["auc"]
    best_iteration = int(np.argmax(val_auc_curve)) + 1

    model = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        n_estimators=best_iteration,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="lightgbm"):
        mlflow.log_param("model", "LightGBM")
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 3))
        mlflow.log_param("n_estimators", best_iteration)
        mlflow.log_param("max_rounds_probed", max_rounds)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("num_leaves", 31)
        mlflow.log_param("n_features_raw", X_train.shape[1])

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_proba)
        mlflow.log_metric("best_iteration", best_iteration)
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, name="model")

    return metrics


def main():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_features()
    X_train, X_val, y_train, y_val = split_data(df)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Train default rate: {y_train.mean():.4f}, Val default rate: {y_val.mean():.4f}")

    lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    print("Logistic Regression (val):", lr_metrics)

    lgb_metrics = train_lightgbm(X_train, y_train, X_val, y_val)
    print("LightGBM (val):", lgb_metrics)


if __name__ == "__main__":
    main()
