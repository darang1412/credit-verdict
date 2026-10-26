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
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
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
        # ROC-AUC is dominated by the large non-default class at ~8% default
        # rate, so a strong ROC-AUC can still hide weak ranking within the
        # rare class this project actually cares about. PR-AUC (average
        # precision) is computed over precision/recall - both defined in
        # terms of the positive (default) class only - so it doesn't get that
        # free pass from the majority class, and is the more honest number
        # for judging whether this model is safe to deploy.
        "pr_auc": average_precision_score(y_true, y_proba),
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


def prepare_categoricals_for_search(X: pd.DataFrame) -> pd.DataFrame:
    """Cast every object column to pandas `category` dtype ONCE, over the
    whole training split, before cross-validation or a hyperparameter search.
    Unlike train_lightgbm's train/val casting below (which must guard against
    a category appearing only in validation), every StratifiedKFold fold
    produced from this same X_train is a subset of it - every category is
    guaranteed already known in every fold, so there's no unseen-category
    case to handle here."""
    X = X.copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")
    return X


def tune_lightgbm_hyperparameters(X_train, y_train, cv: StratifiedKFold) -> dict:
    """Phase 5 gap flagged by the roadmap review: hyperparameters were
    previously fixed at un-searched values (learning_rate=0.05, num_leaves=31,
    min_child_samples left at the library default) with only n_estimators
    ever actually chosen from data, via the manual best-iteration probe in
    train_lightgbm below. A small randomized search over the three
    hyperparameters that matter most for LightGBM on tabular data -
    num_leaves (tree complexity), learning_rate (step size),
    min_child_samples (leaf-level overfitting guard) - replaces "these were
    the docs' example values" with an actual, data-driven answer to "how did
    you pick these?".

    n_estimators is intentionally fixed and modest (200) for this search
    step, not searched or early-stopped: the search is choosing tree SHAPE,
    not final round count. The real round count is re-derived from scratch
    afterwards on the winning shape, via the same manual best-iteration probe
    train_lightgbm already uses - LightGBM's automatic early stopping is
    unreliable under scale_pos_weight on this LightGBM release (4.7.0), see
    train_lightgbm's docstring, so a fixed small n_estimators here (rather
    than trusting early stopping inside the search too) sidesteps that bug
    entirely instead of hitting it 24 more times.

    Scored on average precision (PR-AUC), not accuracy or raw AUC - see
    compute_metrics for why PR-AUC is the more honest metric for this
    project's ~8%-default-rate target.
    """
    X_search = prepare_categoricals_for_search(X_train)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # n_jobs=-1 lives on RandomizedSearchCV (parallelizes across the 8*3=24
    # candidate/fold fits), NOT on the LGBMClassifier fit inside each of them
    # (n_jobs=1). Setting both to -1 oversubscribes the machine - each of the
    # ~24 outer worker processes would ALSO try to grab every core for its
    # own LightGBM training, so they mostly contend with each other instead
    # of doing useful work (observed: ~15-18 python processes, each getting a
    # sliver of CPU time, taking far longer wall-clock than either extreme -
    # serial search, or this same outer-parallel/inner-serial split - would).
    base_model = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        n_estimators=200,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )
    param_distributions = {
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "min_child_samples": [10, 20, 50, 100],
    }
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=8,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=False,
    )
    search.fit(X_search, y_train)

    return {"best_params": search.best_params_, "best_cv_pr_auc": float(search.best_score_)}


def cross_validate_lightgbm(X_train, y_train, cv: StratifiedKFold, params: dict) -> dict:
    """StratifiedKFold sanity check that the LightGBM config chosen above
    (tuned num_leaves/learning_rate/min_child_samples, fixed n_estimators)
    isn't just a lucky single split - reports mean +/- std AUC and PR-AUC
    across folds, each of which preserves the ~8% default rate by
    construction (StratifiedKFold), so fold-to-fold differences reflect real
    variance in the model, not one fold randomly landing at a different
    default rate than another.

    This is a diagnostic ALONGSIDE, not a replacement for, the single
    stratified train/val split used for the model actually logged to MLflow
    and for Phase 6's calibration - both of those need one fixed, comparable
    validation set across phases, which k-fold CV doesn't give you.
    """
    X_cv = prepare_categoricals_for_search(X_train)
    aucs, pr_aucs = [], []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_cv, y_train), start=1):
        X_tr, X_va = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
        scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

        model = lgb.LGBMClassifier(
            objective="binary",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
        model.fit(X_tr, y_tr)
        y_proba = model.predict_proba(X_va)[:, 1]
        fold_auc = roc_auc_score(y_va, y_proba)
        fold_pr_auc = average_precision_score(y_va, y_proba)
        aucs.append(fold_auc)
        pr_aucs.append(fold_pr_auc)
        print(f"  fold {fold}: AUC={fold_auc:.4f}, PR-AUC={fold_pr_auc:.4f}")

    return {
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "cv_pr_auc_mean": float(np.mean(pr_aucs)),
        "cv_pr_auc_std": float(np.std(pr_aucs)),
    }


def train_lightgbm(
    X_train, y_train, X_val, y_val, num_leaves: int = 31, learning_rate: float = 0.05,
    min_child_samples: int = 20,
) -> tuple[dict, dict]:
    """Primary candidate. Categorical columns are cast to pandas' `category`
    dtype - LightGBM detects these automatically and splits on them directly
    (grouping category values by how they relate to the target), instead of
    needing them exploded into many one-hot columns first. It also handles
    NaN internally, learning per split which direction a missing value should
    go - so none of the imputation used for logistic regression above is
    needed, or wanted: imputing would throw away the "this was missing"
    signal Phase 3's EDA showed is itself informative for several columns
    (EXT_SOURCE missingness, bureau history).

    num_leaves/learning_rate/min_child_samples default to the values used
    before this project added tune_lightgbm_hyperparameters, but are meant to
    be called with that search's winning values from main() - kept as
    parameters (not hardcoded) so this function still works standalone."""
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
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
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
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        random_state=42,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="lightgbm"):
        mlflow.log_param("model", "LightGBM")
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 3))
        mlflow.log_param("n_estimators", best_iteration)
        mlflow.log_param("max_rounds_probed", max_rounds)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_leaves", num_leaves)
        mlflow.log_param("min_child_samples", min_child_samples)
        mlflow.log_param("n_features_raw", X_train.shape[1])

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_proba)
        mlflow.log_metric("best_iteration", best_iteration)
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, name="model")

    final_params = {
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_child_samples": min_child_samples,
        "n_estimators": best_iteration,
    }
    return metrics, final_params


def main():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_features()
    X_train, X_val, y_train, y_val = split_data(df)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Train default rate: {y_train.mean():.4f}, Val default rate: {y_val.mean():.4f}")

    lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    print("Logistic Regression (val):", lr_metrics)

    # Stratified, not plain, K-fold: at ~8% default rate an ordinary random
    # fold split can land noticeably off that rate by chance, adding
    # fold-to-fold noise that has nothing to do with which hyperparameters or
    # model are actually better.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    tuning_result = tune_lightgbm_hyperparameters(X_train, y_train, cv)
    print("LightGBM hyperparameter search:", tuning_result)
    with mlflow.start_run(run_name="lightgbm_hpo"):
        mlflow.log_params(tuning_result["best_params"])
        mlflow.log_metric("cv_pr_auc", tuning_result["best_cv_pr_auc"])

    lgb_metrics, final_params = train_lightgbm(
        X_train, y_train, X_val, y_val, **tuning_result["best_params"]
    )
    print("LightGBM (val):", lgb_metrics)

    print("Cross-validating the final LightGBM config (StratifiedKFold, 3 folds):")
    cv_metrics = cross_validate_lightgbm(X_train, y_train, cv, final_params)
    print("LightGBM CV summary:", cv_metrics)
    with mlflow.start_run(run_name="lightgbm_cv_check"):
        mlflow.log_params(final_params)
        mlflow.log_metrics(cv_metrics)


if __name__ == "__main__":
    main()
