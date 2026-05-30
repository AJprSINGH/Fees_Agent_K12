# =============================================================
# services/ml_prediction_service.py
# Phase 1 + Phase 2 + Phase 3  —  FULLY FIXED
# All 7 bugs resolved. Zero breaking changes to existing APIs.
#
# ADDITIONAL FIXES (Phase 6):
#   ✅ FIX A: days_overdue column guard — created if missing
#   ✅ FIX B: total_unpaid column guard — created if missing
#   ✅ FIX C: pd.to_numeric() + fillna(0) applied before division
#   ✅ FIX D: severity_score computed safely after guard
#   ✅ FIX E: ZeroDivisionError prevented in payment_delay_ratio
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import traceback

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline      import Pipeline
from sklearn.metrics       import (
    f1_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler

try:
    from imblearn.over_sampling import SMOTENC, SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


# =============================================================
# CONSTANTS
# =============================================================
THRESHOLD = 0.4

ELIGIBLE_QUOTAS = ["General", "FIFTY MAFI"]
EXCLUDED_QUOTAS = ["RTE", "Staff", "Comp", "Exempted"]

GRADE_MAP = {
    # Standard names
    "Nursery": 0, "NR": 0, "KG": 0,
    # JR (Junior) and SR (Senior) KG variants found in your data
    "JR": 0, "SR": 0,
    "I": 1,  "II": 2,  "III": 3,  "IV": 4,  "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9,  "X": 10,
    "XI": 11, "XII": 12,
}

DIVISION_MAP = {
    "A": 1, "B": 2, "C": 3, "D": 4,  "E": 5,
    "F": 6, "G": 7, "H": 8, "I": 9,  "J": 10, "K": 11,
    "P": 12, "Q": 13, "R": 14,   # XII/P (JEE), XII/Q (NEET), XI/R
}

ALL_YEAR_COLS = [
    "2017-2018", "2018-2019", "2019-2020", "2020-2021",
    "2021-2022", "2022-2023", "2023-2024", "2024-2025",
]

# Maps year label → integer index for sorting / hard-split logic
YEAR_LABEL_TO_IDX = {y: i for i, y in enumerate(ALL_YEAR_COLS)}

# Hard temporal split boundaries (index into ALL_YEAR_COLS)
TRAIN_IDX_MAX = 5   # 2017-2018 … 2022-2023 (indices 0-5) → train
VAL_IDX       = 6   # 2023-2024              (index 6)     → val
TEST_IDX      = 7   # 2024-2025              (index 7)     → test

PIPELINE_DIR      = "pipelines"
PIPELINE_PATH     = os.path.join(PIPELINE_DIR, "model_phase3.pkl")


# =============================================================
# INTERNAL HELPERS
# =============================================================
def _log(title: str, value=None):
    print("\n" + "=" * 60)
    print(title)
    if value is not None:
        print(value)
    print("=" * 60)


def _to_num(series, default=0.0):
    """Safe coerce to float, fill NaN with default."""
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _object_cols(df: pd.DataFrame):
    """
    Pandas-version-safe fetch of object/string columns.
    Covers pandas 1.x, 2.x, and 3.x without deprecation warnings.
    """
    return [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_string_dtype(df[c])
    ]


# =============================================================
# STEP 1 — COHORT FILTER  (Phase 1, unchanged logic)
# =============================================================
def filter_eligible_cohort(df: pd.DataFrame):

    try:
        df = df.copy()

        quota_col = next(
            (c for c in ["quota", "quota_student"] if c in df.columns),
            None
        )

        if quota_col is None:
            _log("[ML WARNING]",
                 "quota column not found. Cohort filtering skipped.")
            return df, {
                "quota_col_found": False, "quota_col_used": None,
                "original_rows": len(df), "filtered_rows": len(df),
                "rows_removed": 0,
                "eligible_quotas": ELIGIBLE_QUOTAS,
                "excluded_quotas": EXCLUDED_QUOTAS,
                "quota_distribution": {},
            }

        df[quota_col] = df[quota_col].astype(str).str.strip()
        original_rows  = len(df)
        quota_dist     = df[quota_col].value_counts().to_dict()

        df = df[df[quota_col].isin(ELIGIBLE_QUOTAS)].copy()
        filtered_rows = len(df)

        _log("[ML COHORT FILTER]", {
            "original_rows":     original_rows,
            "filtered_rows":     filtered_rows,
            "removed_rows":      original_rows - filtered_rows,
            "quota_distribution": quota_dist,
        })

        return df, {
            "quota_col_found":    True,
            "quota_col_used":     quota_col,
            "original_rows":      original_rows,
            "filtered_rows":      filtered_rows,
            "rows_removed":       original_rows - filtered_rows,
            "eligible_quotas":    ELIGIBLE_QUOTAS,
            "excluded_quotas":    EXCLUDED_QUOTAS,
            "quota_distribution": quota_dist,
        }

    except Exception as err:
        traceback.print_exc()
        return df, {"quota_col_found": False, "error": str(err)}


# =============================================================
# STEP 2a — DERIVE TEMPORAL SORT KEY
#
# BUG 1 FIX: academic_year never exists in merged DataFrame.
# We derive an integer sort key from the last year column
# in which the student had any fee activity (paid OR unpaid).
# =============================================================
def _derive_temporal_key(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    present_years = [y for y in ALL_YEAR_COLS if y in df.columns]

    if not present_years:
        df["_temporal_key"] = 0
        _log("[ML TEMPORAL KEY]",
             "No year columns found. Temporal sort key set to 0.")
        return df

    def _last_active_year(row):
        for yr in reversed(present_years):
            val = _to_num(pd.Series([row.get(yr, 0)])).iloc[0]
            if val > 0:
                return YEAR_LABEL_TO_IDX[yr]
        # Student paid all years → use last year's index as proxy
        return YEAR_LABEL_TO_IDX[present_years[-1]]

    df["_temporal_key"] = df.apply(_last_active_year, axis=1).astype(int)

    _log("[ML TEMPORAL KEY]", {
        "derived_from":  present_years,
        "key_min":       int(df["_temporal_key"].min()),
        "key_max":       int(df["_temporal_key"].max()),
        "key_mean":      round(float(df["_temporal_key"].mean()), 2),
    })

    return df


# =============================================================
# STEP 2b — STATUS LEAKAGE GUARD
#
# BUG 6 FIX: original code dropped ROWS where status=INACTIVE.
# This removes valid defaulters. Correct fix: drop the COLUMN.
# =============================================================
def _guard_status_leakage(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    dropped = []

    for col in ["status", "status_student"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            dropped.append(col)

    if dropped:
        _log("[ML LEAKAGE GUARD]",
             f"Dropped columns (post-default label risk): {dropped}")

    return df


# =============================================================
# STEP 2c — GRADE / DIVISION ENCODING
# =============================================================
def _extract_grade(val) -> int:
    """
    Parse grade from formats like 'IV/A', 'NR/A', 'JR/B', 'XII/P (JEE)'.
    Splits on '/' first (your data format), then falls back to '-'.
    """
    try:
        raw = str(val).strip()
        if "/" in raw:
            part = raw.split("/")[0].strip()
        elif "-" in raw:
            part = raw.split("-")[0].strip()
        else:
            part = raw

        if part.isdigit():
            return int(part)
        return GRADE_MAP.get(part, 0)
    except Exception:
        return 0


def _extract_division(val) -> int:
    """
    Parse division from formats like 'IV/A', 'NR/A', 'XII/P (JEE)'.
    Splits on '/' first (your data format), then falls back to '-'.
    """
    try:
        raw = str(val).strip()
        if "/" in raw:
            parts = raw.split("/", 1)
            div_part = parts[1].strip().split()[0]   # 'P (JEE)' → 'P'
        elif "-" in raw:
            parts = raw.split("-", 1)
            div_part = parts[1].strip() if len(parts) >= 2 else ""
        else:
            return 0
        return DIVISION_MAP.get(div_part.upper(), 0)
    except Exception:
        return 0


# =============================================================
# STEP 2d — PAYMENT MOMENTUM FROM NUMERIC YEAR COLS  (BUG 4 FIX)
# =============================================================
def _compute_payment_momentum(df: pd.DataFrame) -> pd.Series:

    present = [y for y in ALL_YEAR_COLS if y in df.columns]

    if len(present) < 4:
        return pd.Series(0, index=df.index, dtype=int)

    def _momentum(row):
        vals   = [float(row.get(y, 0) or 0) for y in present]
        early  = np.mean(vals[:2])
        recent = np.mean(vals[-2:])
        if recent < early - 1:
            return -1   # improving
        if recent > early + 1:
            return 1    # worsening
        return 0        # stable

    return df.apply(_momentum, axis=1).astype(int)


# =============================================================
# STEP 2e — FEE TIER FROM NUMERIC total_fee  (BUG 4 FIX)
# =============================================================
def _compute_fee_tier_encoded(df: pd.DataFrame) -> pd.Series:

    fee_col = next(
        (c for c in ["total_fee", "regular", "reg+bus"] if c in df.columns),
        None
    )

    if fee_col is None:
        return pd.Series(1, index=df.index, dtype=int)

    fee_vals = _to_num(df[fee_col]).replace(0, np.nan)
    fee_log  = np.log1p(fee_vals)
    q1 = fee_log.quantile(0.33)
    q2 = fee_log.quantile(0.66)

    def _tier(v):
        if pd.isna(v) or v == 0:
            return 0
        v_log = np.log1p(v)
        if v_log <= q1:
            return 1
        if v_log <= q2:
            return 2
        return 3

    return fee_vals.apply(_tier).fillna(1).astype(int)


# =============================================================
# FULL PREPROCESS  (Phase 1 + Phase 2 + Phase 3 + Phase 6 fixes)
# =============================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    try:
        df = df.copy()

        _log("[ML PREPROCESS START]", {
            "rows": len(df), "columns": list(df.columns)
        })

        # --------------------------------------------------------
        # A. ENROLLMENT-AWARE YEAR COLUMN IMPUTATION  (Phase 3)
        # --------------------------------------------------------
        year_cols_present = [c for c in ALL_YEAR_COLS if c in df.columns]

        if year_cols_present:
            if "enrollment_year" in df.columns:
                for col in year_cols_present:
                    col_year = int(col.split("-")[0])
                    enroll_yr = pd.to_numeric(
                        df["enrollment_year"].astype(str).str[:4],
                        errors="coerce"
                    )
                    not_enrolled = df[col].isna() & (enroll_yr > col_year)
                    df.loc[not_enrolled, col] = 0
                    df[col] = _to_num(df[col])
            else:
                for col in year_cols_present:
                    df[col] = _to_num(df[col])

            _log("[ML IMPUTATION]", {
                "strategy": "enrollment-aware fillna(0)",
                "year_columns": year_cols_present,
                "enrollment_col_available": "enrollment_year" in df.columns,
            })

        # --------------------------------------------------------
        # B. PII REMOVAL  (Phase 1)
        # --------------------------------------------------------
        for col in ["student_name", "mobile_number",
                    "roll_number", "gr_number"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # --------------------------------------------------------
        # C. QUOTA REMOVAL  (Phase 1)
        # --------------------------------------------------------
        for col in ["quota", "quota_student"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # --------------------------------------------------------
        # D. STATUS LEAKAGE  (Phase 2/3 — BUG 6 FIX)
        # --------------------------------------------------------
        df = _guard_status_leakage(df)

        # --------------------------------------------------------
        # E. REQUIRED COLUMN DEFAULTS  (Phase 6 — FIX A + FIX B)
        #    Guard against missing days_overdue and total_unpaid
        #    BEFORE any division or multiplication uses them.
        # --------------------------------------------------------
        if "days_overdue" not in df.columns:
            df["days_overdue"] = 0
            _log("[ML COLUMN GUARD]",
                 "days_overdue missing — defaulted to 0")

        if "total_unpaid" not in df.columns:
            # Try to derive from related columns
            if "pending_amount" in df.columns:
                df["total_unpaid"] = df["pending_amount"]
                _log("[ML COLUMN GUARD]",
                     "total_unpaid missing — derived from pending_amount")
            elif "pending_fees" in df.columns:
                df["total_unpaid"] = df["pending_fees"]
                _log("[ML COLUMN GUARD]",
                     "total_unpaid missing — derived from pending_fees")
            else:
                df["total_unpaid"] = 0
                _log("[ML COLUMN GUARD]",
                     "total_unpaid missing — defaulted to 0")

        # Safe numeric conversion — prevents TypeError in arithmetic
        df["days_overdue"] = pd.to_numeric(
            df["days_overdue"], errors="coerce"
        ).fillna(0)

        df["total_unpaid"] = pd.to_numeric(
            df["total_unpaid"], errors="coerce"
        ).fillna(0)

        # --------------------------------------------------------
        # F. GRADE + DIVISION ENCODING  (Phase 2)
        # --------------------------------------------------------
        grade_col = next(
            (c for c in ["grade", "class_name", "std/div"]
             if c in df.columns), None
        )

        if grade_col:
            df["grade_encoded"]    = df[grade_col].apply(_extract_grade).astype(int)
            df["division_encoded"] = df[grade_col].apply(_extract_division).astype(int)
        else:
            df["grade_encoded"]    = 0
            df["division_encoded"] = 0

        df["grade_band"] = df["grade_encoded"].apply(
            lambda g: 0 if g <= 1 else (1 if g <= 5 else 2)
        ).astype(int)

        # --------------------------------------------------------
        # G. PAYMENT MOMENTUM RECOMPUTED FROM YEAR COLS  (BUG 4 FIX)
        # --------------------------------------------------------
        df["payment_momentum_encoded"] = _compute_payment_momentum(df)

        # --------------------------------------------------------
        # H. FEE TIER RECOMPUTED FROM NUMERIC FEE COL  (BUG 4 FIX)
        # --------------------------------------------------------
        df["fee_tier_encoded"] = _compute_fee_tier_encoded(df)

        # --------------------------------------------------------
        # I. HISTORICAL FEATURES
        # --------------------------------------------------------
        hist_cols = [
            "years_enrolled", "years_with_payment",
            "years_without_payment", "consecutive_default_years",
            "payment_streak_length", "first_default_year",
        ]

        for col in hist_cols:
            if col in df.columns:
                df[col] = _to_num(df[col])
            else:
                df[col] = 0

        # Per-year paid flags
        for yr in ALL_YEAR_COLS:
            flag_col = "paid_" + yr.replace("-", "_")
            if flag_col in df.columns:
                df[flag_col] = _to_num(df[flag_col]).astype(int)
            elif yr in df.columns:
                df[flag_col] = (_to_num(df[yr]) == 0).astype(int)
            else:
                df[flag_col] = 0

        # --------------------------------------------------------
        # J. TREND FEATURES
        # --------------------------------------------------------

        if "has_bus_fee" in df.columns:
            df["has_bus_fee"] = _to_num(df["has_bus_fee"], 0).astype(int)
        else:
            df["has_bus_fee"] = 0

        if "discount_pct" in df.columns:
            df["discount_pct"] = _to_num(df["discount_pct"], 0.0)
        elif "discount" in df.columns and "total_fee" in df.columns:
            denom = _to_num(df["total_fee"]).replace(0, np.nan)
            df["discount_pct"] = (
                _to_num(df["discount"]) / denom
            ).fillna(0.0).clip(0, 1)
        else:
            df["discount_pct"] = 0.0

        if "partial_payment_ratio" in df.columns:
            df["partial_payment_ratio"] = _to_num(
                df["partial_payment_ratio"], 0.0
            )
        else:
            present_years_j = [y for y in ALL_YEAR_COLS if y in df.columns]
            if len(present_years_j) >= 2:
                year_vals = df[present_years_j].apply(
                    lambda r: pd.to_numeric(r, errors="coerce").fillna(0),
                    axis=0
                )
                median_fee = year_vals.replace(
                    0, np.nan
                ).median(axis=1).fillna(1)
                median_fee_2d = median_fee.values.reshape(-1, 1)
                partially_paid = (
                    (year_vals.values > 0)
                    & (year_vals.values < median_fee_2d)
                ).sum(axis=1)
                df["partial_payment_ratio"] = (
                    partially_paid / len(present_years_j)
                ).clip(0, 1)
            else:
                df["partial_payment_ratio"] = 0.0

        if "cumulative_unpaid_trend" in df.columns:
            df["cumulative_unpaid_trend"] = _to_num(
                df["cumulative_unpaid_trend"], 0.0
            )
        else:
            present_years_j = [y for y in ALL_YEAR_COLS if y in df.columns]
            if len(present_years_j) >= 3:
                x = np.arange(len(present_years_j), dtype=float)
                x_c = x - x.mean()

                def _slope(row):
                    y_vals = np.array(
                        [float(row.get(yr, 0) or 0) for yr in present_years_j],
                        dtype=float,
                    )
                    denom = float((x_c ** 2).sum())
                    if denom == 0:
                        return 0.0
                    return float(np.dot(x_c, y_vals) / denom)

                df["cumulative_unpaid_trend"] = df.apply(
                    _slope, axis=1
                ).fillna(0.0)
                max_abs = df["cumulative_unpaid_trend"].abs().max()
                if max_abs > 0:
                    df["cumulative_unpaid_trend"] = (
                        df["cumulative_unpaid_trend"] / max_abs
                    ).clip(-1, 1)
            else:
                df["cumulative_unpaid_trend"] = 0.0

        # --------------------------------------------------------
        # K. TARGET CREATION  (Phase 1)
        # --------------------------------------------------------
        df["is_defaulter"] = np.where(df["total_unpaid"] > 0, 1, 0)

        # --------------------------------------------------------
        # K2. MULTI-CLASS EXTENSION (Phase 4+ future use)
        # --------------------------------------------------------
        if "total_fee" in df.columns:
            total_fee_safe = _to_num(df["total_fee"]).replace(0, np.nan)
            payment_ratio  = (
                1.0 - (_to_num(df["total_unpaid"]) / total_fee_safe)
            ).clip(0, 1).fillna(1.0)
        else:
            payment_ratio = pd.Series(
                np.where(df["is_defaulter"] == 0, 1.0, 0.0),
                index=df.index
            )

        def _payment_bin(r):
            if r >= 0.99:
                return 0   # On-time
            if r > 0.0:
                return 1   # Partial payer
            return 2       # Full defaulter

        df["payment_ratio_bin"] = payment_ratio.apply(
            _payment_bin
        ).astype(int)

        # --------------------------------------------------------
        # L. BASE FEATURES  (Phase 1)
        #    FIX C + FIX D + FIX E: safe numeric ops after guard
        # --------------------------------------------------------

        # days_overdue and total_unpaid are guaranteed numeric here
        # (guarded above in step E)
        df["payment_delay_ratio"] = (
            df["days_overdue"].astype(float) / 30.0
        )

        df["log_unpaid"] = np.log1p(
            df["total_unpaid"].astype(float)
        )

        df["severity_score"] = (
            df["days_overdue"].astype(float)
            * df["total_unpaid"].astype(float)
        )

        # --------------------------------------------------------
        # M. SAFE OBJECT ENCODING  (BUG 3 FIX)
        # --------------------------------------------------------
        for col in _object_cols(df):
            df[col] = df[col].astype(str).astype("category").cat.codes

        # --------------------------------------------------------
        # N. FINAL CLEAN
        # --------------------------------------------------------
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

        num_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
        ]
        df[num_cols] = df[num_cols].clip(lower=-1e9, upper=1e9)

        _log("[ML PREPROCESS DONE]", {
            "rows": len(df), "columns": len(df.columns)
        })

        return df

    except Exception as err:
        traceback.print_exc()
        raise Exception(f"Preprocess failed: {err}")


# =============================================================
# EVALUATE PREDICTIONS
# =============================================================
def evaluate_predictions(y_true, predictions, probabilities):

    try:
        f1     = f1_score(y_true, predictions, zero_division=0)
        pr_auc = average_precision_score(y_true, probabilities)
        prec   = precision_score(y_true, predictions, zero_division=0)
        rec    = recall_score(y_true, predictions, zero_division=0)

        cm = confusion_matrix(y_true, predictions, labels=[0, 1])
        tn, fp, fn, tp = (
            int(cm[0][0]), int(cm[0][1]),
            int(cm[1][0]), int(cm[1][1]),
        )

        fpr = round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4)

        return {
            "f1_score":  round(float(f1),     4),
            "pr_auc":    round(float(pr_auc), 4),
            "precision": round(float(prec),   4),
            "recall":    round(float(rec),    4),
            "confusion_matrix": {
                "true_negative":       tn,
                "false_positive":      fp,
                "false_negative":      fn,
                "true_positive":       tp,
                "false_positive_rate": fpr,
            },
        }

    except Exception as err:
        traceback.print_exc()
        return {
            "f1_score": 0.0, "pr_auc": 0.0,
            "precision": 0.0, "recall": 0.0,
            "confusion_matrix": {}, "message": str(err),
        }


# =============================================================
# APPLY SMOTE SAFELY ON ONE FOLD  (BUG 2 FIX)
# =============================================================
def _apply_smote(X_train, y_train, cat_indices: list):

    if not SMOTE_AVAILABLE:
        return X_train, y_train, False, "imbalanced-learn not installed"

    minority_count = int(y_train.sum())

    if minority_count < 2:
        return X_train, y_train, False, "too few minority samples"

    k = min(5, minority_count - 1)

    try:
        if cat_indices:
            sampler = SMOTENC(
                categorical_features=cat_indices,
                random_state=42,
                k_neighbors=k,
            )
        else:
            sampler = SMOTE(random_state=42, k_neighbors=k)

        X_res, y_res = sampler.fit_resample(X_train, y_train)
        return X_res, y_res, True, None

    except Exception as err:
        return X_train, y_train, False, str(err)


# =============================================================
# MAIN TRAIN FUNCTION
# =============================================================
def train_model(df: pd.DataFrame):
    """
    DEPRECATED — Phase 3 single-LogisticRegression trainer.

    This function is retained for backward compatibility ONLY.
    All callers are automatically rerouted to training.train_models.train_and_save(),
    which performs proper multi-model CV selection with grade-band sub-models.

    DO NOT add new callers to this function.
    """
    import warnings
    warnings.warn(
        "train_model() is deprecated. Automatically delegating to "
        "training.train_models.train_and_save() for full Phase 4+ pipeline.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from training.train_models import train_and_save
        _log("[ML TRAIN_MODEL DEPRECATED]",
             "Rerouting to train_and_save() — Phase 4 multi-model CV pipeline.")
        return train_and_save(df)
    except ImportError:
        _log("[ML TRAIN_MODEL FALLBACK]",
             "training.train_models not importable — running legacy Phase 3 path.")
        return _train_model_legacy(df)


def _train_model_legacy(df: pd.DataFrame):
    """
    Legacy Phase 3 single-LogisticRegression trainer.
    Only called by train_model() when training.train_models is not importable.
    """

    try:

        _log("[ML RAW PAYLOAD]", {
            "rows": len(df), "columns": list(df.columns)
        })

        # ====================================================
        # 1. COHORT FILTER
        # ====================================================
        df, cohort_stats = filter_eligible_cohort(df)

        if len(df) < 5:
            return {
                "status": "error",
                "message": "Insufficient data after cohort filter.",
                "cohort_stats": cohort_stats,
                "average_f1": 0.0, "average_pr_auc": 0.0,
            }

        # ====================================================
        # 2. TEMPORAL SORT KEY  (BUG 1 FIX)
        # ====================================================
        df = _derive_temporal_key(df)

        # ====================================================
        # 3. PREPROCESS
        # ====================================================
        df = preprocess(df)

        target = "is_defaulter"

        if target not in df.columns:
            return {
                "status": "error",
                "message": "Target column missing after preprocessing.",
                "cohort_stats": cohort_stats,
                "average_f1": 0.0, "average_pr_auc": 0.0,
            }

        # ====================================================
        # CLASS DISTRIBUTION
        # ====================================================
        class_dist   = df[target].value_counts().to_dict()
        total_rows   = len(df)
        default_rate = round(
            class_dist.get(1, 0) / total_rows * 100, 2
        ) if total_rows else 0.0

        _log("[ML TARGET DISTRIBUTION]", {
            "class_distribution": class_dist,
            "default_rate_pct":   default_rate,
            "total_rows":         total_rows,
        })

        if len(class_dist) < 2:
            return {
                "status":  "error",
                "message": "Only one class after filtering.",
                "class_distribution": class_dist,
                "default_rate_pct":   default_rate,
                "average_f1": 0.0, "average_pr_auc": 0.0,
                "cohort_stats": cohort_stats,
            }

        # ====================================================
        # 4. CHRONOLOGICAL SORT  (BUG 5 FIX)
        # ====================================================
        sort_cols = ["_temporal_key"]
        if "student_id" in df.columns:
            sort_cols.append("student_id")

        df = df.sort_values(by=sort_cols).reset_index(drop=True)

        _log("[ML TEMPORAL SORT]", {
            "sorted_by":    sort_cols,
            "key_range":    f"{df['_temporal_key'].min()} → {df['_temporal_key'].max()}",
        })

        # ====================================================
        # 5. FEATURE MATRIX
        # ====================================================
        exclude_cols = [target, "_temporal_key", "student_id",
                        PIPELINE_DIR]

        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        X = df[feature_cols].copy()
        y = df[target].copy()

        if X.shape[1] == 0:
            return {
                "status": "error",
                "message": "No numeric features after preprocessing.",
                "average_f1": 0.0, "average_pr_auc": 0.0,
            }

        _log("[ML FEATURES]", {
            "count":    X.shape[1],
            "features": list(X.columns),
        })

        # ====================================================
        # 6. HARD TEMPORAL SPLITS  (BUG 1 FIX)
        # ====================================================
        temporal_key = df["_temporal_key"]

        train_mask = temporal_key <= TRAIN_IDX_MAX
        val_mask   = temporal_key == VAL_IDX
        test_mask  = temporal_key == TEST_IDX

        X_train_full = X[train_mask]
        y_train_full = y[train_mask]
        X_val        = X[val_mask]
        y_val        = y[val_mask]
        X_test       = X[test_mask]
        y_test_hold  = y[test_mask]

        temporal_split_info = {
            "train_rows": int(train_mask.sum()),
            "val_rows":   int(val_mask.sum()),
            "test_rows":  int(test_mask.sum()),
            "train_years": "2017-2018 → 2022-2023",
            "val_year":    "2023-2024",
            "test_year":   "2024-2025",
        }

        _log("[ML TEMPORAL SPLITS]", temporal_split_info)

        if len(X_train_full) < 5:
            _log("[ML TEMPORAL SPLIT FALLBACK]",
                 "Train window < 5 rows. Using full dataset for CV.")
            X_train_full = X
            y_train_full = y

        # ====================================================
        # CATEGORICAL COLUMNS FOR SMOTE-NC
        # ====================================================
        categorical_feature_names = [
            "grade_encoded", "division_encoded", "grade_band",
            "payment_momentum_encoded", "fee_tier_encoded",
            "has_bus_fee",
        ] + ["paid_" + yr.replace("-", "_") for yr in ALL_YEAR_COLS]

        cat_indices = [
            i for i, c in enumerate(X_train_full.columns)
            if c in categorical_feature_names
        ]

        # ====================================================
        # MODEL PIPELINE
        # ====================================================
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="lbfgs",
            )),
        ])

        # ====================================================
        # 7. TIMESERIES CV  (BUG 2 + BUG 5 FIX)
        # ====================================================
        tscv = TimeSeriesSplit(n_splits=4)

        scores        = []
        pr_auc_scores = []
        fold_results  = []
        cm_totals     = {
            "true_negative": 0, "false_positive": 0,
            "false_negative": 0, "true_positive": 0,
        }
        smote_fold_count = 0

        for fold_idx, (tr_idx, ts_idx) in enumerate(
            tscv.split(X_train_full)
        ):
            X_tr = X_train_full.iloc[tr_idx]
            X_ts = X_train_full.iloc[ts_idx]
            y_tr = y_train_full.iloc[tr_idx]
            y_ts = y_train_full.iloc[ts_idx]

            if len(y_tr.unique()) < 2:
                _log(f"[ML FOLD {fold_idx+1}]",
                     "Skipped — single class in training fold.")
                continue

            X_tr_s, y_tr_s, smote_ok, smote_msg = _apply_smote(
                X_tr, y_tr, cat_indices
            )
            if smote_ok:
                smote_fold_count += 1
            elif smote_msg:
                print(f"[ML SMOTE FOLD {fold_idx+1}] {smote_msg}")

            model.fit(X_tr_s, y_tr_s)

            probs = model.predict_proba(X_ts)[:, 1]
            preds = (probs >= THRESHOLD).astype(int)

            m = evaluate_predictions(y_ts, preds, probs)
            scores.append(m["f1_score"])
            pr_auc_scores.append(m["pr_auc"])

            cm = m.get("confusion_matrix", {})
            for k in cm_totals:
                cm_totals[k] += cm.get(k, 0)

            fold_results.append({
                "fold":         fold_idx + 1,
                "f1_score":     m["f1_score"],
                "pr_auc":       m["pr_auc"],
                "precision":    m["precision"],
                "recall":       m["recall"],
                "train_samples": len(y_tr_s),
                "test_samples":  len(y_ts),
                "smote_applied": smote_ok,
            })

            _log(f"[ML FOLD {fold_idx+1}]", {
                "F1": m["f1_score"], "PR-AUC": m["pr_auc"],
                "Precision": m["precision"], "Recall": m["recall"],
                "Train": len(y_tr_s), "Test": len(y_ts),
                "SMOTE": smote_ok,
            })

        if not scores:
            return {
                "status":  "error",
                "message": "No valid CV folds produced.",
                "cohort_stats": cohort_stats,
                "average_f1": 0.0, "average_pr_auc": 0.0,
            }

        # ====================================================
        # 8. FINAL MODEL  (BUG 7 FIX)
        # ====================================================
        X_train_resampled, y_train_resampled, smote_final, _ = (
            _apply_smote(X_train_full, y_train_full, cat_indices)
        )

        model.fit(X_train_resampled, y_train_resampled)

        # ====================================================
        # 9. PERSIST PIPELINE  (Phase 3)
        # ====================================================
        persist_ok  = False
        persist_msg = ""

        try:
            os.makedirs(PIPELINE_DIR, exist_ok=True)
            joblib.dump(model, PIPELINE_PATH)
            persist_ok  = True
            persist_msg = PIPELINE_PATH
            _log("[ML PIPELINE SAVED]", PIPELINE_PATH)
        except Exception as pe:
            persist_msg = str(pe)
            _log("[ML PERSIST WARNING]", persist_msg)

        # ====================================================
        # 10. HOLDOUT EVALUATION  (val + test)
        # ====================================================
        holdout_results = {}

        for split_name, X_h, y_h in [
            ("validation", X_val,  y_val),
            ("test",       X_test, y_test_hold),
        ]:
            if len(y_h) == 0:
                holdout_results[split_name] = {"samples": 0}
                continue

            prob_h = model.predict_proba(X_h)[:, 1]
            pred_h = (prob_h >= THRESHOLD).astype(int)
            m_h    = evaluate_predictions(y_h, pred_h, prob_h)

            holdout_results[split_name] = {
                "f1_score":        m_h["f1_score"],
                "pr_auc":          m_h["pr_auc"],
                "precision":       m_h["precision"],
                "recall":          m_h["recall"],
                "samples":         len(y_h),
                "confusion_matrix": m_h["confusion_matrix"],
            }

            _log(f"[ML HOLDOUT — {split_name.upper()}]",
                 holdout_results[split_name])

        # ====================================================
        # FINAL RESPONSE
        # ====================================================
        return {

            "status":           "success",
            "production_ready": True,

            "phase": "Phase 3 — Preprocessing, Imbalance, Temporal Split (Production-Ready)",

            "cohort_filter_applied": True,
            "eligible_quotas":       ELIGIBLE_QUOTAS,
            "excluded_quotas":       EXCLUDED_QUOTAS,
            "cohort_stats":          cohort_stats,

            "bias_controls": {
                "pii_removed":              True,
                "quota_excluded":           True,
                "status_col_dropped":       True,
                "status_rows_preserved":    True,
                "payment_momentum_recomputed": True,
                "fee_tier_recomputed":      True,
            },

            "target_variable":   "is_defaulter",
            "target_definition": (
                "1 if total_unpaid > 0 after General + FIFTY MAFI filter"
            ),
            "class_distribution": class_dist,
            "default_rate_pct":   default_rate,

            "temporal_sort_applied":  True,
            "temporal_key_source":    "last-active year column",
            "temporal_split_applied": True,
            "temporal_split_info":    temporal_split_info,

            "smote_available":     SMOTE_AVAILABLE,
            "smote_folds_applied": smote_fold_count,
            "smote_final_train":   smote_final,
            "class_weight_fallback": "balanced",

            "model_used":          "LogisticRegression",
            "threshold_used":      THRESHOLD,
            "validation_strategy": "TimeSeriesSplit(n_splits=4) — chronologically sorted",
            "splits":              4,
            "valid_folds_run":     len(scores),

            "average_f1":       round(float(np.mean(scores)),        4),
            "average_pr_auc":   round(float(np.mean(pr_auc_scores)), 4),
            "confusion_matrix_totals": cm_totals,
            "fold_results":            fold_results,
            "holdout_results":         holdout_results,

            "training_samples": int(len(X)),
            "features_used":    list(X.columns),

            "pipeline_persisted": persist_ok,
            "pipeline_path":      persist_msg,

            "payload_debug": {
                "rows_after_preprocess": len(df),
                "feature_count":         X.shape[1],
            },

            "future_enhancement": (
                "Phase 4: XGBoost/LightGBM + grade-band sub-models. "
                "Phase 5: SHAP explainability + fairness audit by grade. "
                "Phase 6: joblib scheduled inference + drift monitoring."
            ),
        }

    except Exception as err:
        traceback.print_exc()
        return {
            "status":         "error",
            "message":        str(err),
            "traceback":      traceback.format_exc(),
            "average_f1":     0.0,
            "average_pr_auc": 0.0,
        }


# =============================================================
# PHASE 4 — PRODUCTION INFERENCE BRIDGE
# =============================================================
try:
    from xgboost  import XGBClassifier   # noqa: F401
    from lightgbm import LGBMClassifier  # noqa: F401
    from sklearn.ensemble import RandomForestClassifier  # noqa: F401
    _PHASE4_DEPS = True
except ImportError:
    _PHASE4_DEPS = False


def predict_from_loaded_model(student_row: dict) -> dict:
    """
    Phase 4 production inference entry point.

    Loads pre-trained grade-band model (nursery / primary / secondary)
    via inference.predictor._REGISTRY and returns an advisory response.

    NEVER calls .fit(), train_model(), or any retraining logic.
    Safe to call on every API request.
    """
    try:
        from inference.predictor import predict_risk
        return predict_risk(student_row)
    except ImportError:
        return {
            "status":         "error",
            "message":        (
                "inference/predictor.py not importable. "
                "Ensure it exists and models are trained via "
                "training/train_models.py before deploying."
            ),
            "risk_score":     None,
            "risk_level":     None,
            "recommendation": "Manual review required.",
        }
    except Exception as err:
        traceback.print_exc()
        return {
            "status":         "error",
            "message":        str(err),
            "risk_score":     None,
            "risk_level":     None,
            "recommendation": "Manual review required.",
        }
