"""
Phase 4 - Feature Engineering.

Turns the four raw Home Credit tables into a single feature table, one row per
applicant, keyed on SK_ID_CURR. The core technique throughout is the same one
described in the roadmap's Phase 4 resources: group each auxiliary table by
SK_ID_CURR and aggregate (count/mean/max/sum) its many rows-per-applicant down
to a handful of summary columns, then join those onto application_train.

This also bakes in the concrete action items from the Phase 3 EDA
(notebooks/01_eda_application.ipynb) rather than leaving them as loose notes:
  - DAYS_EMPLOYED's 365243 sentinel -> NaN + IS_DAYS_EMPLOYED_ANOMALY flag
  - HAS_BUREAU_HISTORY flag (missingness itself is informative here)
  - CREDIT/INCOME-style ratios instead of relying on raw income/credit alone

Deliberately targets ~50 engineered features (see the roadmap's Phase 4 note:
"don't chase all 10,000+ features some competition solutions built"), not an
exhaustive aggregation of every column in every table.

Leakage audit (checked against data/raw/HomeCredit_columns_description.csv
before writing any aggregation below - the question for every field is "would
this have been known at the moment THIS application was submitted?"):

  - bureau.csv time fields (DAYS_CREDIT, DAYS_CREDIT_ENDDATE,
    CREDIT_DAY_OVERDUE) are explicitly documented as measured relative to the
    current application ("days before current application", "at the time of
    application") - safe by definition, no audit judgment call needed.
  - bureau.csv "current"-labelled fields (CREDIT_ACTIVE, AMT_CREDIT_SUM,
    AMT_CREDIT_SUM_DEBT, AMT_CREDIT_SUM_OVERDUE, CNT_CREDIT_PROLONG) are
    documented only as "current" / "so far", without an explicit "as of the
    application date" qualifier - the safety of using these rests on Home
    Credit's documented process (the bureau report is pulled at application
    time), not on a per-column guarantee in the data dictionary. This is the
    standard reading used by every public solution referenced in the
    roadmap's Phase 3/4 resources, not a distinction unique to this project -
    flagged here explicitly rather than silently assumed.
  - previous_application.csv (NAME_CONTRACT_STATUS, DAYS_DECISION,
    AMT_APPLICATION, AMT_CREDIT, CNT_PAYMENT) covers this same applicant's
    OTHER, earlier applications - DAYS_DECISION is explicitly "relative to
    current application", and the table only contains previous applications
    by construction. Safe.
  - credit_card_balance.csv (MONTHS_BALANCE, AMT_BALANCE,
    AMT_CREDIT_LIMIT_ACTUAL, SK_DPD, AMT_DRAWINGS_CURRENT) is keyed on
    MONTHS_BALANCE, documented as "month of balance relative to application
    date" (0/negative = at or before it) - every row aggregated here is at or
    before the application, by the same definition. Safe.
  - installments_payments.csv and POS_CASH_balance.csv are NOT used in this
    feature table. Both carry a real look-ahead risk this audit specifically
    ruled out for v1: e.g. installments_payments.csv's
    DAYS_ENTRY_PAYMENT/AMT_PAYMENT reflect whether a payment was EVENTUALLY
    made and how late, which can encode outcome information about a loan that
    may not have fully resolved until after the current application's
    decision point. Revisit only with the same field-by-field check applied
    above, not by assuming the pattern transfers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Resolved relative to this file, not the current working directory - a
# relative string like "data/raw" only works if you happen to run the script
# from inside ml/pipeline/, which breaks the moment it's run from the repo
# root, a different cwd, or an IDE "Run" button that doesn't set cwd the same
# way. REPO_ROOT climbs out of ml/pipeline/ to the top-level data/ directory
# shared by every phase, rather than a local pipeline-only copy.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = str(REPO_ROOT / "data" / "raw")

# Columns that are only NaN after a left-join because the applicant has zero
# rows in that auxiliary table - i.e. the true value is 0, not "unknown".
# Ratio/mean columns are deliberately NOT included here: for those, NaN
# genuinely means "undefined" (e.g. an approval rate with a 0/0 denominator),
# and LightGBM handles NaN natively by learning the best split direction for
# missing values. Forcing a 0 there would misrepresent "no data" as "worst
# possible value", which is a different claim.
ZERO_FILL_COLS = [
    "BUREAU_COUNT",
    "BUREAU_ACTIVE_COUNT",
    "BUREAU_CNT_CREDIT_PROLONG_SUM",
    "PREV_COUNT",
    "PREV_APPROVED_COUNT",
    "PREV_REFUSED_COUNT",
    "CC_MONTHS_COUNT",
    "CC_LATE_PAYMENT_COUNT",
]


def load_raw(data_dir: str = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    app = pd.read_csv(f"{data_dir}/application_train.csv")
    bureau = pd.read_csv(f"{data_dir}/bureau.csv")
    previous = pd.read_csv(f"{data_dir}/previous_application.csv")
    credit_card = pd.read_csv(f"{data_dir}/credit_card_balance.csv")
    return app, bureau, previous, credit_card


def clean_and_engineer_application(app: pd.DataFrame) -> pd.DataFrame:
    """Applicant-level cleaning + ratio features (no auxiliary tables yet)."""
    app = app.copy()

    # EDA finding: 18% of applicants carry DAYS_EMPLOYED == 365243 (~1000
    # years) - a placeholder for "not currently employed" (mostly
    # pensioners), and that group defaults LESS often (5.4% vs 8.7%). Left
    # as a raw number, a model would read "employed for 1000 years" as a
    # huge, meaningless number and likely try to use it as if more days
    # employed always meant lower risk. Replacing it with NaN removes that
    # distortion; the flag preserves the (protective) signal instead of
    # discarding it.
    app["IS_DAYS_EMPLOYED_ANOMALY"] = (app["DAYS_EMPLOYED"] == 365243).astype(int)
    app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].replace(365243, np.nan)

    # DAYS_* columns are negative day-counts before the application date.
    # Converting to positive years is purely for readability - no change in
    # information content, but "42 years old" reads better than "-15,340".
    app["AGE_YEARS"] = -app["DAYS_BIRTH"] / 365.25
    app["EMPLOYED_YEARS"] = -app["DAYS_EMPLOYED"] / 365.25

    # EDA finding: raw AMT_INCOME_TOTAL correlates with TARGET at -0.004 -
    # essentially nothing. But intuitively, what should matter for repayment
    # risk is the *relationship* between the loan size and what the
    # applicant earns, not either number alone: a $500k loan means something
    # very different to someone earning $30k vs $300k. A tree model can in
    # principle discover a ratio like this from splits on the two raw
    # columns, but only approximately and only if it happens to split at the
    # right points - handing it the ratio directly is a much more reliable
    # way to expose that signal.
    app["CREDIT_INCOME_RATIO"] = app["AMT_CREDIT"] / app["AMT_INCOME_TOTAL"]
    app["ANNUITY_INCOME_RATIO"] = app["AMT_ANNUITY"] / app["AMT_INCOME_TOTAL"]
    # Credit amount divided by the annuity (periodic payment) approximates
    # how many payment periods the loan spans - a proxy for loan duration
    # that isn't a raw column in this dataset.
    app["CREDIT_ANNUITY_RATIO"] = app["AMT_CREDIT"] / app["AMT_ANNUITY"]
    app["INCOME_PER_FAMILY_MEMBER"] = app["AMT_INCOME_TOTAL"] / app["CNT_FAM_MEMBERS"]

    # EDA finding: EXT_SOURCE_1/2/3 are individually the strongest linear
    # predictors, but each is missing a very different amount (56% / 0.2% /
    # 20%). A row-wise mean over whichever of the three ARE present gives
    # the model one more-complete "consensus external score" on top of (not
    # instead of) the three raw columns, plus an explicit count of how many
    # were actually available for this applicant.
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    app["EXT_SOURCE_MEAN"] = app[ext_cols].mean(axis=1)
    app["EXT_SOURCE_STD"] = app[ext_cols].std(axis=1)
    app["EXT_SOURCE_COUNT_AVAILABLE"] = app[ext_cols].notnull().sum(axis=1)

    return app


def build_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    """One row per SK_ID_CURR, aggregated from every bureau-reported credit."""
    b = bureau.copy()
    b["IS_ACTIVE_CREDIT"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)

    agg = b.groupby("SK_ID_CURR").agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("IS_ACTIVE_CREDIT", "sum"),
        BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BUREAU_CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_CREDIT_DAY_OVERDUE_MEAN=("CREDIT_DAY_OVERDUE", "mean"),
        BUREAU_AMT_CREDIT_SUM_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_AMT_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUREAU_AMT_CREDIT_SUM_MAX=("AMT_CREDIT_SUM", "max"),
        BUREAU_AMT_CREDIT_SUM_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_AMT_CREDIT_SUM_DEBT_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"),
        BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_AMT_CREDIT_SUM_OVERDUE_MAX=("AMT_CREDIT_SUM_OVERDUE", "max"),
        BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BUREAU_CNT_CREDIT_PROLONG_SUM=("CNT_CREDIT_PROLONG", "sum"),
    ).reset_index()

    agg["BUREAU_ACTIVE_RATIO"] = agg["BUREAU_ACTIVE_COUNT"] / agg["BUREAU_COUNT"]
    # Fraction of this applicant's total bureau-reported credit that is
    # currently owed as debt - computed at the aggregate level (sum of debt
    # / sum of credit across all their bureau records), not as a per-row
    # ratio then averaged, since a per-row ratio would let a single small,
    # fully-drawn credit line skew the result as much as a large one.
    # A handful of applicants have AMT_CREDIT_SUM summing to exactly 0 across
    # all their bureau records (e.g. only closed/written-off lines with no
    # remaining credit amount) while still carrying nonzero debt - dividing
    # by that zero denominator produces inf, not a large-but-real ratio, so
    # it's replaced with NaN (undefined) rather than left to silently break
    # any downstream step that assumes finite values.
    agg["BUREAU_DEBT_CREDIT_RATIO"] = agg["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"] / agg[
        "BUREAU_AMT_CREDIT_SUM_SUM"
    ].replace(0, np.nan)

    # EDA finding: applicants with zero bureau records default at 10.1% vs
    # 7.7% for those with at least one - informative missingness, not a data
    # gap. This flag is built from the ORIGINAL table (before grouping)
    # rather than from BUREAU_COUNT after the merge, so it's explicit and
    # doesn't rely on a downstream fillna to mean the right thing.
    has_history = bureau[["SK_ID_CURR"]].drop_duplicates()
    has_history["HAS_BUREAU_HISTORY"] = 1
    agg = agg.merge(has_history, on="SK_ID_CURR", how="left")

    return agg


def build_previous_application_features(previous: pd.DataFrame) -> pd.DataFrame:
    """One row per SK_ID_CURR, aggregated from this applicant's own past
    applications with Home Credit specifically (distinct from bureau.csv,
    which covers credit at OTHER institutions)."""
    p = previous.copy()
    p["IS_APPROVED"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    p["IS_REFUSED"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)

    # How much of what they asked for they actually received. A lender
    # granting meaningfully less than requested is itself a signal that the
    # applicant looked riskier at the time - independent of whether that
    # particular application's status was ultimately "Approved". AMT_APPLICATION
    # is legitimately 0 for a large minority of rows in this table (a known
    # quirk of previous_application.csv, not a bug), so those denominators
    # are replaced with NaN rather than producing inf.
    p["APPLICATION_CREDIT_RATIO"] = p["AMT_CREDIT"] / p["AMT_APPLICATION"].replace(0, np.nan)

    agg = p.groupby("SK_ID_CURR").agg(
        PREV_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("IS_APPROVED", "sum"),
        PREV_REFUSED_COUNT=("IS_REFUSED", "sum"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_AMT_APPLICATION_MAX=("AMT_APPLICATION", "max"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_CREDIT_MAX=("AMT_CREDIT", "max"),
        PREV_APPLICATION_CREDIT_RATIO_MEAN=("APPLICATION_CREDIT_RATIO", "mean"),
        PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),  # closest to 0 = most recent
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
    ).reset_index()

    agg["PREV_APPROVAL_RATE"] = agg["PREV_APPROVED_COUNT"] / agg["PREV_COUNT"]
    agg["PREV_REFUSAL_RATE"] = agg["PREV_REFUSED_COUNT"] / agg["PREV_COUNT"]
    return agg


def build_credit_card_features(credit_card: pd.DataFrame) -> pd.DataFrame:
    """One row per SK_ID_CURR, aggregated from monthly credit-card balance
    snapshots on any Home Credit card(s) this applicant holds."""
    c = credit_card.copy()

    # Per-month utilization (balance as a fraction of the credit limit),
    # computed per row (per month) and then aggregated, since it's the
    # *trend and severity* of utilization across months that's the risk
    # signal here, not a single snapshot value.
    c["UTILIZATION"] = c["AMT_BALANCE"] / c["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    c["IS_LATE_PAYMENT"] = (c["SK_DPD"] > 0).astype(int)

    agg = c.groupby("SK_ID_CURR").agg(
        CC_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        CC_AMT_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_AMT_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_UTILIZATION_MEAN=("UTILIZATION", "mean"),
        CC_UTILIZATION_MAX=("UTILIZATION", "max"),
        CC_SK_DPD_MEAN=("SK_DPD", "mean"),
        CC_SK_DPD_MAX=("SK_DPD", "max"),
        CC_LATE_PAYMENT_COUNT=("IS_LATE_PAYMENT", "sum"),
        CC_AMT_DRAWINGS_CURRENT_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
        CC_CNT_DRAWINGS_CURRENT_MEAN=("CNT_DRAWINGS_CURRENT", "mean"),
    ).reset_index()

    agg["CC_LATE_PAYMENT_RATE"] = agg["CC_LATE_PAYMENT_COUNT"] / agg["CC_MONTHS_COUNT"]
    return agg


def build_feature_table(data_dir: str = DATA_DIR, save_path: str | None = None) -> pd.DataFrame:
    app, bureau, previous, credit_card = load_raw(data_dir)

    app = clean_and_engineer_application(app)
    bureau_feats = build_bureau_features(bureau)
    prev_feats = build_previous_application_features(previous)
    cc_feats = build_credit_card_features(credit_card)

    # Every aggregated table already has exactly one row per SK_ID_CURR, so
    # these left-joins can't duplicate application_train rows - each merge
    # only adds columns, never rows.
    features = app.merge(bureau_feats, on="SK_ID_CURR", how="left")
    features = features.merge(prev_feats, on="SK_ID_CURR", how="left")
    features = features.merge(cc_feats, on="SK_ID_CURR", how="left")

    # HAS_BUREAU_HISTORY and the count columns are NaN after the left-join
    # only because the applicant had zero matching rows in that table - a
    # real "0"/"no", not an unknown value, so filling them is correct here
    # (unlike the ratio/mean columns, which stay NaN on purpose - see
    # ZERO_FILL_COLS docstring above).
    features["HAS_BUREAU_HISTORY"] = features["HAS_BUREAU_HISTORY"].fillna(0).astype(int)
    for col in ZERO_FILL_COLS:
        features[col] = features[col].fillna(0)

    if save_path:
        features.to_parquet(save_path, index=False)

    return features


if __name__ == "__main__":
    table = build_feature_table(save_path=str(REPO_ROOT / "data" / "processed" / "features.parquet"))
    n_engineered = (
        table.shape[1]
        - pd.read_csv(f"{DATA_DIR}/application_train.csv", nrows=0).shape[1]
    )
    print(f"Feature table shape: {table.shape}")
    print(f"Engineered/aggregated columns added beyond raw application_train: {n_engineered}")
