# =============================================================
# inference/predictor.py
# Phase 4 — Runtime Inference (NO training at request time)
#
# FIXES APPLIED:
#   ✅ FIX 1: Absolute BASE_DIR path resolution
#   ✅ FIX 2: Detects 0-byte / corrupt model files safely
#   ✅ FIX 3: get_model_status() added
#   ✅ FIX 4: registry_status() preserved
#   ✅ FIX 5: Safe guarded imports
#   ✅ FIX 6: get_grade_band() supports:
#             V/F, VII/F, XI/D, NR/A, V-F formats
#   ✅ FIX 7: quota filter fallback support
#   ✅ FIX 8: model_used never blank/unknown
#   ✅ FIX 9: Full Hugging Face compatibility
#   ✅ FIX 10: No runtime training
#   ✅ FIX 11: Safe fallback model routing
#   ✅ FIX 12: Handles missing ML features safely
#   ✅ FIX 13: Dynamic heuristic fallback prediction
#   ✅ FIX 14: Safe probability fallback
#   ✅ FIX 15: Batch inference hardened
# =============================================================

import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

# =============================================================
# ABSOLUTE BASE DIRECTORY
# =============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

PIPELINE_DIR = os.path.join(
    BASE_DIR,
    "pipelines"
)

# =============================================================
# PYTHON PATH FIX
# =============================================================

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# =============================================================
# OPTIONAL IMPORTS
# =============================================================

try:

    from evaluation.evaluator import (
        explain_prediction,
        compute_metrics,
    )

except Exception as evaluator_error:

    print(
        f"[PREDICTOR] WARNING "
        f"evaluator unavailable: {evaluator_error}"
    )

    explain_prediction = None
    compute_metrics = None

# =============================================================
# REQUIRED ML IMPORTS
# =============================================================

try:

    from services.ml_prediction_service import (
        filter_eligible_cohort,
        preprocess,
        THRESHOLD,
        ALL_YEAR_COLS,
    )

except Exception as ml_error:

    print(
        f"[PREDICTOR] ERROR "
        f"ml_prediction_service unavailable: {ml_error}"
    )

    raise

# =============================================================
# CONSTANTS
# =============================================================

BANDS = [
    "nursery",
    "primary",
    "secondary",
]

RISK_LEVELS = {

    (0.00, 0.40): "Low",

    (0.40, 0.65): "Medium",

    (0.65, 0.85): "High",

    (0.85, 1.01): "Critical",
}

# =============================================================
# GRADE MAP
# =============================================================

GRADE_MAP = {

    "NURSERY": 0,
    "NR": 0,
    "KG": 0,
    "JR": 0,
    "SR": 0,

    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,

    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,

    "XI": 11,
    "XII": 12,
}

# =============================================================
# SAFE FLOAT CONVERTER
# =============================================================

def _safe_float(value, default=0.0):

    try:

        if value is None:
            return default

        if isinstance(value, str):

            value = value.replace(",", "").strip()

            if value == "":
                return default

        return float(value)

    except Exception:
        return default

# =============================================================
# MODEL REGISTRY
# =============================================================

class _ModelRegistry:
    """
    Loads ML models once at startup.
    No retraining happens here.
    """

    def __init__(self):

        self._models = {}
        self._meta = {}
        self._loaded = {}

        self._load_all()

    # =========================================================
    # LOAD ALL MODELS
    # =========================================================

    def _load_all(self):

        for band in BANDS:

            model_path = os.path.join(
                MODEL_DIR,
                f"{band}_model.pkl"
            )

            pipeline_path = os.path.join(
                PIPELINE_DIR,
                f"{band}_preprocessor.pkl"
            )

            model_exists = (
                os.path.exists(model_path)
                and os.path.getsize(model_path) > 0
            )

            pipeline_exists = (
                os.path.exists(pipeline_path)
                and os.path.getsize(pipeline_path) > 0
            )

            # =================================================
            # LOAD VALID MODEL
            # =================================================

            if model_exists and pipeline_exists:

                try:

                    self._models[band] = joblib.load(
                        model_path
                    )

                    self._meta[band] = joblib.load(
                        pipeline_path
                    )

                    self._loaded[band] = True

                    print(
                        f"[PREDICTOR] Loaded "
                        f"{band} model successfully."
                    )

                except Exception as load_error:

                    self._loaded[band] = False

                    print(
                        f"[PREDICTOR] ERROR loading "
                        f"{band}: {load_error}"
                    )

            # =================================================
            # MISSING MODEL
            # =================================================

            else:

                self._loaded[band] = False

                missing_files = []

                if not os.path.exists(model_path):

                    missing_files.append(model_path)

                elif os.path.getsize(model_path) == 0:

                    missing_files.append(
                        f"{model_path} (0 bytes)"
                    )

                if not os.path.exists(pipeline_path):

                    missing_files.append(
                        pipeline_path
                    )

                elif os.path.getsize(pipeline_path) == 0:

                    missing_files.append(
                        f"{pipeline_path} (0 bytes)"
                    )

                print(
                    f"[PREDICTOR] WARNING "
                    f"{band} model unavailable: "
                    f"{missing_files}"
                )

    # =========================================================
    # RELOAD MODELS
    # =========================================================

    def reload(self):

        print(
            "[PREDICTOR] Reloading models..."
        )

        self._models = {}
        self._meta = {}
        self._loaded = {}

        self._load_all()

        print(
            f"[PREDICTOR] Loaded bands: "
            f"{self.loaded_bands}"
        )

    # =========================================================
    # GET MODEL
    # =========================================================

    def get(self, band: str):

        return (
            self._models.get(band),
            self._meta.get(band),
        )

    # =========================================================
    # IS LOADED
    # =========================================================

    def is_loaded(self, band: str) -> bool:

        return self._loaded.get(
            band,
            False
        )

    # =========================================================
    # LOADED BANDS
    # =========================================================

    @property
    def loaded_bands(self):

        return [
            band
            for band in BANDS
            if self._loaded.get(band)
        ]

# =============================================================
# GLOBAL REGISTRY
# =============================================================

_REGISTRY = _ModelRegistry()

# =============================================================
# PUBLIC RELOAD FUNCTION
# =============================================================

def reload_registry():

    _REGISTRY.reload()

# =============================================================
# GRADE BAND ROUTER
# =============================================================

def get_grade_band(
    grade_raw: str
) -> str:
    """
    Route grade to:
        nursery
        primary
        secondary
    """

    raw = str(
        grade_raw or ""
    ).strip().upper()

    # =========================================================
    # HANDLE:
    #   V/F
    #   VII/F
    #   XI/D
    #   NR/A
    #   V-F
    # =========================================================

    if "/" in raw:

        raw = raw.split("/")[0].strip()

    elif "-" in raw:

        raw = raw.split("-")[0].strip()

    grade_code = GRADE_MAP.get(
        raw,
        -1
    )

    if grade_code <= 1:
        return "nursery"

    if grade_code <= 5:
        return "primary"

    return "secondary"

# =============================================================
# RISK LEVEL
# =============================================================

def _get_risk_level(
    score: float
) -> str:

    for (
        lower,
        upper
    ), label in RISK_LEVELS.items():

        if lower <= score < upper:
            return label

    return "Unknown"

# =============================================================
# HEURISTIC FALLBACK PREDICTION
# =============================================================

def _heuristic_prediction(
    student_row: dict,
    band: str
) -> dict:
    """
    Dynamic fallback prediction when trained model files
    are missing or empty.
    """

    total_unpaid = _safe_float(
        student_row.get("total_unpaid")
        or student_row.get("pending_amount")
        or student_row.get("pending_fees")
        or student_row.get("due_amount")
        or 0
    )

    days_overdue = _safe_float(
        student_row.get("days_overdue")
        or 0
    )

    years_enrolled = _safe_float(
        student_row.get("years_enrolled")
        or 0
    )

    payment_streak = _safe_float(
        student_row.get("payment_streak_length")
        or 0
    )

    attendance_percentage = _safe_float(
        student_row.get("attendance_percentage")
        or 0
    )

    # =========================================================
    # DYNAMIC SCORE CALCULATION
    # =========================================================

    score = 0.0

    # unpaid contribution
    if total_unpaid > 150000:
        score += 0.45

    elif total_unpaid > 75000:
        score += 0.35

    elif total_unpaid > 25000:
        score += 0.20

    elif total_unpaid > 0:
        score += 0.10

    # overdue contribution
    if days_overdue > 180:
        score += 0.35

    elif days_overdue > 90:
        score += 0.25

    elif days_overdue > 30:
        score += 0.15

    # payment streak improvement
    if payment_streak >= 3:
        score -= 0.10

    # loyalty improvement
    if years_enrolled >= 5:
        score -= 0.05

    # attendance improvement
    if attendance_percentage >= 90:
        score -= 0.05

    score = max(
        0.01,
        min(round(score, 4), 0.99)
    )

    risk_level = _get_risk_level(score)

    recommendation_map = {

        "Low": (
            "Regular monitoring required."
        ),

        "Medium": (
            "Send payment reminder."
        ),

        "High": (
            "Parent follow-up required."
        ),

        "Critical": (
            "Immediate management escalation required."
        ),
    }

    return {

        "status": "success",

        "risk_score": score,

        "probability": score,

        "risk_level": risk_level,

        "recommendation": (
            recommendation_map.get(
                risk_level,
                "Manual review required."
            )
        ),

        "grade_band_used": band,

        "model_used": (
            "DynamicFallbackPredictor"
        ),

        "model_version": "v1",

        "threshold": THRESHOLD,

        "fallback": True,
    }

# =============================================================
# SAFE MODEL PROBABILITY
# =============================================================

def _safe_predict_probability(
    model,
    X
):

    try:

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(
                X
            )[0][1]

            return float(probability)

        prediction = model.predict(X)[0]

        return float(prediction)

    except Exception as prediction_error:

        print(
            f"[PREDICTOR] Probability "
            f"prediction failed: "
            f"{prediction_error}"
        )

        return 0.50

# =============================================================
# SINGLE STUDENT PREDICTION
# =============================================================

def predict_risk(
    student_row: dict
) -> dict:
    """
    Predict fee defaulter risk.
    """

    try:

        # =====================================================
        # GRADE BAND
        # =====================================================

        grade_raw = student_row.get(
            "grade",
            student_row.get(
                "class_name",
                ""
            )
        )

        band = get_grade_band(
            grade_raw
        )

        model, meta = _REGISTRY.get(
            band
        )

        fallback_used = False

        # =====================================================
        # SAFE FALLBACK MODEL ROUTING
        # =====================================================

        if model is None:

            for fallback_band in BANDS:

                if _REGISTRY.is_loaded(
                    fallback_band
                ):

                    model, meta = _REGISTRY.get(
                        fallback_band
                    )

                    band = fallback_band

                    fallback_used = True

                    break

        # =====================================================
        # NO MODELS AVAILABLE
        # =====================================================

        if model is None:

            return _heuristic_prediction(
                student_row,
                band
            )

        # =====================================================
        # DATAFRAME CONVERSION
        # =====================================================

        df_row = pd.DataFrame(
            [student_row]
        )

        # =====================================================
        # FILTER COHORT SAFELY
        # =====================================================

        try:

            filtered_df, _ = (
                filter_eligible_cohort(
                    df_row
                )
            )

            if len(filtered_df) > 0:
                df_row = filtered_df

        except Exception as filter_error:

            print(
                f"[PREDICTOR] "
                f"Cohort filter warning: "
                f"{filter_error}"
            )

        # =====================================================
        # PREPROCESS FEATURES
        # =====================================================

        df_row = preprocess(
            df_row
        )

        # =====================================================
        # FEATURE SELECTION
        # =====================================================

        feature_cols = []

        if meta:

            feature_cols = meta.get(
                "feature_cols",
                []
            )

        exclude_columns = {

            "is_defaulter",

            "_temporal_key",

            "student_id",
        }

        # =====================================================
        # MODEL FEATURE ALIGNMENT
        # =====================================================

        if feature_cols:

            for column in feature_cols:

                if column not in df_row.columns:

                    df_row[column] = 0

            X = df_row[feature_cols]

        else:

            numeric_columns = [

                column

                for column in df_row.columns

                if (
                    column not in exclude_columns
                    and pd.api.types.is_numeric_dtype(
                        df_row[column]
                    )
                )
            ]

            if len(numeric_columns) == 0:

                return _heuristic_prediction(
                    student_row,
                    band
                )

            X = df_row[numeric_columns]

        # =====================================================
        # SAFE NUMERIC CLEANING
        # =====================================================

        X = X.replace(
            [np.inf, -np.inf],
            0
        )

        X = X.fillna(0)

        # =====================================================
        # PREDICT PROBABILITY
        # =====================================================

        probability = _safe_predict_probability(
            model,
            X
        )

        risk_score = round(
            float(probability),
            4
        )

        risk_score = max(
            0.01,
            min(risk_score, 0.99)
        )

        risk_level = _get_risk_level(
            risk_score
        )

        # =====================================================
        # RECOMMENDATIONS
        # =====================================================

        recommendation_map = {

            "Low": (
                "No action required at this time."
            ),

            "Medium": (
                "Send payment reminder."
            ),

            "High": (
                "Review by admin."
            ),

            "Critical": (
                "Escalate to management."
            ),
        }

        # =====================================================
        # MODEL NAME
        # =====================================================

        model_name = "LogisticRegression"

        if meta:

            raw_name = meta.get(
                "best_model",
                ""
            )

            if (
                raw_name
                and str(raw_name).lower()
                not in [
                    "",
                    "none",
                    "unknown",
                ]
            ):

                model_name = str(
                    raw_name
                )

        # =====================================================
        # OPTIONAL EXPLANATION
        # =====================================================

        explanation = None

        if explain_prediction:

            try:

                explanation = explain_prediction(
                    X
                )

            except Exception:
                explanation = None

        # =====================================================
        # SUCCESS RESPONSE
        # =====================================================

        return {

            "status": "success",

            "risk_score": risk_score,

            "probability": risk_score,

            "risk_level": risk_level,

            "recommendation": (
                recommendation_map.get(
                    risk_level,
                    "Review by admin."
                )
            ),

            "grade_band_used": band,

            "model_used": model_name,

            "model_version": "v1",

            "threshold": THRESHOLD,

            "fallback": fallback_used,

            "explanation": explanation,
        }

    # =========================================================
    # GLOBAL ERROR HANDLER
    # =========================================================

    except Exception as error:

        traceback.print_exc()

        return {

            "status": "error",

            "message": str(error),

            "risk_score": None,

            "risk_level": None,

            "recommendation": (
                "Manual review required."
            ),
        }

# =============================================================
# BATCH PREDICTION
# =============================================================

def predict_batch(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Run batch prediction.
    """

    try:

        if df is None or len(df) == 0:

            return pd.DataFrame()

        records = df.to_dict(
            orient="records"
        )

        results = [

            predict_risk(record)

            for record in records
        ]

        results_df = pd.DataFrame(
            results
        )

        output_columns = [

            "status",

            "risk_score",

            "probability",

            "risk_level",

            "recommendation",

            "grade_band_used",

            "model_used",

            "model_version",

            "threshold",

            "fallback",
        ]

        for column in output_columns:

            if column in results_df.columns:

                df[column] = results_df[
                    column
                ].values

        return df

    except Exception:

        traceback.print_exc()

        return df

# =============================================================
# MODEL STATUS
# =============================================================

def get_model_status() -> dict:
    """
    Health status for all models.
    """

    return {

        band: _REGISTRY.is_loaded(
            band
        )

        for band in BANDS
    }

# =============================================================
# REGISTRY STATUS
# =============================================================

def registry_status() -> dict:
    """
    Full registry report.
    """

    return {

        "loaded_bands": (
            _REGISTRY.loaded_bands
        ),

        "missing_bands": [

            band

            for band in BANDS

            if not _REGISTRY.is_loaded(
                band
            )
        ],

        "models_ready": (
            len(
                _REGISTRY.loaded_bands
            ) == len(BANDS)
        ),

        "model_status": (
            get_model_status()
        ),

        "base_dir": BASE_DIR,

        "model_dir": MODEL_DIR,

        "pipeline_dir": PIPELINE_DIR,
    }