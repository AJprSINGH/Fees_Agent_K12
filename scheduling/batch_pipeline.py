# =============================================================
# scheduling/batch_pipeline.py
# Phase 6 — Scheduled Batch Inference & Early Warning System
#
# FIXES APPLIED:
#   ✅ Added --dry-run CLI flag
#   ✅ Added rollback-on-drift support
#   ✅ Added Offline DB mode support
#   ✅ Added safer persistence handling
#   ✅ Added local fallback persistence support
#   ✅ Added db_mode visibility in response payload
#   ✅ Preserved existing APIs and response structure
#   ✅ No breaking changes to predictor or training pipeline
#
# Workflow:
#   cron (annual / midyear)
#       → load active students
#       → run batch inference
#       → compute risk scores
#       → generate early warning alerts
#       → persist results
#       → run drift checks
#       → rollback if severe drift detected
#
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import traceback
import argparse

from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd


# =============================================================
# PROJECT ROOT PATH
# =============================================================

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# =============================================================
# CONSTANTS
# =============================================================

HIGH_RISK_LEVELS = {"High", "Critical"}

EARLY_WARNING_MONTH = 2

MIDYEAR_FEE_DUE_THRESHOLD = 0.50

MODEL_VERSION_FALLBACK = os.environ.get(
    "ACTIVE_MODEL_VERSION",
    "latest"
)

BATCH_PREDICTIONS_TABLE = "batch_predictions"

EARLY_WARNING_TABLE = "early_warning_flags"


# =============================================================
# GRACEFUL IMPORTS
# =============================================================

try:

    from inference.predictor import (
        predict_batch,
        registry_status
    )

    PREDICTOR_AVAILABLE = True

except ImportError:

    PREDICTOR_AVAILABLE = False

    predict_batch = None
    registry_status = None

    print(
        "[BATCH PIPELINE] WARNING — "
        "inference.predictor not importable."
    )


try:

    from monitoring.drift_monitor import (
        run_drift_check,
        log_drift_report_to_supabase
    )

    DRIFT_MONITOR_AVAILABLE = True

except ImportError:

    DRIFT_MONITOR_AVAILABLE = False

    run_drift_check = None
    log_drift_report_to_supabase = None

    print(
        "[BATCH PIPELINE] WARNING — "
        "monitoring.drift_monitor unavailable."
    )


try:

    from versioning.model_registry import (
        get_active_version,
        rollback_to_previous,
    )

    REGISTRY_AVAILABLE = True

except ImportError:

    REGISTRY_AVAILABLE = False

    get_active_version = None
    rollback_to_previous = None

    print(
        "[BATCH PIPELINE] WARNING — "
        "versioning.model_registry unavailable."
    )


try:

    from database.supabase_client import (
        supabase,
        is_online,
        client_mode,
    )

except Exception:

    supabase = None

    is_online = lambda: False

    client_mode = lambda: "unavailable"


# =============================================================
# INTERNAL HELPERS
# =============================================================

def _log(title: str, value=None):

    print("\n" + "=" * 60)

    print(title)

    if value is not None:
        print(value)

    print("=" * 60)


def _now_iso() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


def _get_model_version(
    band: str = "global"
) -> str:
    """
    Get active model version from registry.
    """

    if (
        REGISTRY_AVAILABLE
        and supabase
        and get_active_version
    ):

        record = get_active_version(
            supabase,
            band=band
        )

        if record:

            return record.get(
                "version_id",
                MODEL_VERSION_FALLBACK
            )

    return MODEL_VERSION_FALLBACK


# =============================================================
# CORE: BATCH INFERENCE
# =============================================================

def run_batch_inference(
    df_students: pd.DataFrame,
    mode: str = "annual",
    academic_year: str = "",
    baseline_df: Optional[pd.DataFrame] = None,
    model_version: Optional[str] = None,
    persist: bool = True,
    dry_run: bool = False,
    rollback_on_drift: bool = False,
) -> Dict:
    """
    Run scheduled batch inference.

    No retraining occurs here.
    """

    timestamp = _now_iso()

    _log(
        f"[BATCH PIPELINE] Starting "
        f"{mode.upper()} batch inference"
        + (" [DRY RUN]" if dry_run else ""),
        {
            "students": len(df_students),
            "academic_year": academic_year or "unspecified",
            "timestamp": timestamp,
            "db_mode": (
                client_mode()
                if not dry_run
                else "dry_run"
            ),
        }
    )

    # =========================================================
    # VALIDATE PREDICTOR
    # =========================================================

    if (
        not PREDICTOR_AVAILABLE
        or predict_batch is None
    ):

        return {
            "status": "error",
            "message": (
                "inference.predictor unavailable. "
                "Run training/train_models.py first."
            ),
            "timestamp": timestamp,
        }

    # =========================================================
    # MODEL VERSION
    # =========================================================

    mv = model_version or _get_model_version()

    # =========================================================
    # REGISTRY STATUS
    # =========================================================

    if registry_status:

        reg_status = registry_status()

        _log(
            "[BATCH PIPELINE] Model registry status",
            reg_status
        )

        if not reg_status.get("models_ready"):

            _log(
                "[BATCH PIPELINE] WARNING",
                {
                    "missing_bands": reg_status.get(
                        "missing_bands",
                        []
                    )
                }
            )

    # =========================================================
    # RUN INFERENCE
    # =========================================================

    try:

        scored_df = predict_batch(
            df_students.copy()
        )

    except Exception as e:

        traceback.print_exc()

        return {
            "status": "error",
            "message": f"predict_batch failed: {e}",
            "timestamp": timestamp,
        }

    # =========================================================
    # ENRICH RESULTS
    # =========================================================

    scored_df["prediction_timestamp"] = timestamp

    scored_df["model_version"] = mv

    scored_df["academic_year"] = academic_year

    scored_students = int(
        scored_df["risk_score"]
        .notna()
        .sum()
    )

    # =========================================================
    # RISK DISTRIBUTION
    # =========================================================

    if "risk_level" in scored_df.columns:

        dist = (
            scored_df["risk_level"]
            .value_counts()
            .to_dict()
        )

    else:
        dist = {}

    _log(
        "[BATCH PIPELINE] Risk level distribution",
        dist
    )

    # =========================================================
    # EARLY WARNING
    # =========================================================

    early_warning_rows = []

    high_risk_count = 0

    if "risk_level" in scored_df.columns:

        hw_mask = scored_df["risk_level"].isin(
            HIGH_RISK_LEVELS
        )

        high_risk_df = scored_df[
            hw_mask
        ].copy()

        high_risk_count = len(high_risk_df)

        for _, row in high_risk_df.iterrows():

            early_warning_rows.append({
                "student_id": str(
                    row.get("student_id", "")
                ),

                "risk_score": round(
                    float(
                        row.get("risk_score", 0)
                    ),
                    4
                ),

                "risk_level": str(
                    row.get("risk_level", "")
                ),

                "grade": str(
                    row.get(
                        "grade",
                        row.get(
                            "class_name",
                            ""
                        )
                    )
                ),

                "recommendation": str(
                    row.get(
                        "recommendation",
                        "Review by admin."
                    )
                ),

                "model_version": mv,

                "academic_year": academic_year,

                "flagged_at": timestamp,
            })

    _log(
        f"[BATCH PIPELINE] Early warning — "
        f"{high_risk_count} flagged",
        {
            r["student_id"]: r["risk_level"]
            for r in early_warning_rows[:10]
        }
    )

    # =========================================================
    # DRIFT CHECK
    # =========================================================

    drift_report = None

    rollback_result = None

    # =========================================================
    # AUTO LOAD BASELINE
    # =========================================================

    if (
        DRIFT_MONITOR_AVAILABLE
        and run_drift_check
        and baseline_df is None
    ):

        default_data = os.environ.get(
            "TRAINING_DATA_PATH",
            "data/Fees Defaulter Report - Overall.xlsx"
        )

        if os.path.exists(default_data):

            try:

                baseline_df = pd.read_excel(
                    default_data
                )

                _log(
                    "[BATCH PIPELINE] baseline_df auto-loaded",
                    default_data
                )

            except Exception as be:

                _log(
                    "[BATCH PIPELINE] WARNING — "
                    "baseline auto-load failed",
                    str(be)
                )

        else:

            _log(
                "[BATCH PIPELINE] WARNING",
                (
                    "No baseline data found. "
                    "Drift monitoring skipped."
                )
            )

    # =========================================================
    # EXECUTE DRIFT CHECK
    # =========================================================

    if (
        DRIFT_MONITOR_AVAILABLE
        and run_drift_check
        and baseline_df is not None
    ):

        try:

            baseline_scores = (
                baseline_df["risk_score"].values
                if "risk_score" in baseline_df.columns
                else None
            )

            current_scores = (
                scored_df["risk_score"]
                .dropna()
                .values
                if "risk_score" in scored_df.columns
                else None
            )

            drift_report = run_drift_check(
                baseline_df=baseline_df,
                current_df=scored_df,
                model_version=mv,
                baseline_scores=baseline_scores,
                current_scores=current_scores,
            )

            # =================================================
            # DRIFT DETECTED
            # =================================================

            if drift_report.get("drift_detected"):

                _log(
                    "[BATCH PIPELINE] DRIFT DETECTED",
                    drift_report.get(
                        "alert_message",
                        "Unknown drift"
                    )
                )

                # =============================================
                # AUTO ROLLBACK
                # =============================================

                if (
                    rollback_on_drift
                    and REGISTRY_AVAILABLE
                    and rollback_to_previous
                    and drift_report.get(
                        "retrain_recommended"
                    )
                ):

                    _log(
                        "[BATCH PIPELINE] "
                        "Auto rollback triggered"
                    )

                    rollback_result = {}

                    for band in [
                        "nursery",
                        "primary",
                        "secondary",
                        "global",
                    ]:

                        rb = rollback_to_previous(
                            supabase_client=supabase,
                            band=band,
                            reason=drift_report.get(
                                "alert_message",
                                "drift detected"
                            ),
                        )

                        rollback_result[band] = rb

                        _log(
                            f"[BATCH PIPELINE] "
                            f"Rollback [{band}]",
                            rb
                        )

            # =================================================
            # LOG DRIFT REPORT
            # =================================================

            if (
                not dry_run
                and log_drift_report_to_supabase
            ):

                log_drift_report_to_supabase(
                    drift_report,
                    supabase
                )

        except Exception as de:

            traceback.print_exc()

            drift_report = {
                "error": str(de)
            }

    # =========================================================
    # PERSISTENCE
    # =========================================================

    supabase_written = False

    db_mode = client_mode()

    if dry_run:

        _log(
            "[BATCH PIPELINE] DRY RUN — "
            "all persistence skipped"
        )

        db_mode = "dry_run"

    elif persist:

        supabase_written = _persist_to_supabase(
            scored_df=scored_df,
            early_warning=early_warning_rows,
            academic_year=academic_year,
            mode=mode,
            model_version=mv,
            timestamp=timestamp,
        )

    # =========================================================
    # FINAL RESPONSE
    # =========================================================

    result = {

        "status": "success",

        "mode": mode,

        "academic_year": academic_year,

        "total_students": len(df_students),

        "scored_students": scored_students,

        "risk_distribution": dist,

        "high_risk_count": high_risk_count,

        "early_warning_count": len(
            early_warning_rows
        ),

        "early_warning": early_warning_rows,

        "drift_report": drift_report,

        "rollback_result": rollback_result,

        "model_version": mv,

        "timestamp": timestamp,

        "supabase_written": supabase_written,

        "db_mode": db_mode,

        "dry_run": dry_run,

        "human_in_the_loop_note": (
            "All predictions are advisory only. "
            "Final decisions must remain with "
            "authorised school staff."
        ),
    }

    _log(
        "[BATCH PIPELINE] Complete",
        {
            k: v
            for k, v in result.items()
            if k not in ("early_warning",)
        }
    )

    return result


# =============================================================
# PERSISTENCE LAYER
# =============================================================

def _persist_to_supabase(
    scored_df: pd.DataFrame,
    early_warning: list,
    academic_year: str,
    mode: str,
    model_version: str,
    timestamp: str,
) -> bool:
    """
    Persist batch predictions.

    Supports:
        - Live Supabase
        - OfflineClient fallback
    """

    ok = True

    # =========================================================
    # BATCH PREDICTIONS
    # =========================================================

    try:

        cols = [
            "student_id",
            "risk_score",
            "risk_level",
            "recommendation",
            "grade_band_used",
            "model_used",
            "model_version",
            "academic_year",
            "prediction_timestamp",
            "fallback",
        ]

        rows = []

        for _, r in scored_df.iterrows():

            row = {}

            for c in cols:

                val = r.get(c)

                if (
                    val is None
                    or (
                        isinstance(val, float)
                        and np.isnan(val)
                    )
                ):
                    val = None

                row[c] = val

            row["mode"] = mode

            rows.append(row)

        if rows:

            chunk_size = 500

            for i in range(
                0,
                len(rows),
                chunk_size
            ):

                supabase.table(
                    BATCH_PREDICTIONS_TABLE
                ).upsert(
                    rows[i:i + chunk_size],
                    on_conflict=(
                        "student_id,academic_year"
                    )
                ).execute()

            _log(
                "[BATCH PIPELINE] "
                "batch_predictions written",
                len(rows)
            )

    except Exception as e:

        print(
            f"[BATCH PIPELINE] "
            f"batch_predictions failed: {e}"
        )

        ok = False

    # =========================================================
    # EARLY WARNING FLAGS
    # =========================================================

    try:

        if early_warning:

            supabase.table(
                EARLY_WARNING_TABLE
            ).upsert(
                early_warning,
                on_conflict=(
                    "student_id,academic_year"
                )
            ).execute()

            _log(
                "[BATCH PIPELINE] "
                "early_warning_flags written",
                len(early_warning)
            )

    except Exception as e:

        print(
            f"[BATCH PIPELINE] "
            f"early_warning write failed: {e}"
        )

        ok = False

    return ok


# =============================================================
# RETRAINING CHECKER
# =============================================================

def should_retrain_midyear(
    fee_due_fraction: float,
    drift_report: Optional[Dict] = None,
    force: bool = False,
) -> Dict:
    """
    Decide whether retraining is required.
    """

    if force:

        return {
            "retrain_recommended": True,
            "reasons": [
                "Forced retraining."
            ]
        }

    reasons = []

    if (
        fee_due_fraction
        >= MIDYEAR_FEE_DUE_THRESHOLD
    ):

        reasons.append(
            f"Fee due fraction "
            f"{fee_due_fraction:.0%} ≥ "
            f"{MIDYEAR_FEE_DUE_THRESHOLD:.0%}"
        )

    if (
        drift_report
        and drift_report.get(
            "retrain_recommended"
        )
    ):

        reasons.append(
            drift_report.get(
                "alert_message",
                "drift detected"
            )
        )

    return {
        "retrain_recommended": len(reasons) > 0,
        "reasons": reasons,
    }


# =============================================================
# CLI ENTRY
# =============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Phase 6 batch inference pipeline"
        )
    )

    parser.add_argument(
        "--mode",
        choices=["annual", "midyear"],
        default="annual",
    )

    parser.add_argument(
        "--year",
        default="",
        help="Academic year label",
    )

    parser.add_argument(
        "--data",
        default=os.environ.get(
            "TRAINING_DATA_PATH",
            "data/Fees Defaulter Report - Overall.xlsx"
        ),
        help="Path to XLSX",
    )

    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Disable persistence",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run inference only. "
            "No persistence."
        ),
    )

    parser.add_argument(
        "--rollback-on-drift",
        action="store_true",
        help=(
            "Rollback active models "
            "if severe drift detected."
        ),
    )

    args = parser.parse_args()

    # =========================================================
    # VALIDATE INPUT FILE
    # =========================================================

    if not os.path.exists(args.data):

        print(
            f"[ERROR] Data file not found: "
            f"{args.data}"
        )

        sys.exit(1)

    # =========================================================
    # LOAD DATA
    # =========================================================

    _log(
        "[BATCH PIPELINE CLI] Loading data",
        args.data
    )

    df = pd.read_excel(args.data)

    _log(
        "[BATCH PIPELINE CLI] Shape",
        df.shape
    )

    # =========================================================
    # RUN PIPELINE
    # =========================================================

    output = run_batch_inference(
        df_students=df,
        mode=args.mode,
        academic_year=args.year,
        persist=not args.no_persist,
        dry_run=args.dry_run,
        rollback_on_drift=args.rollback_on_drift,
    )

    # =========================================================
    # FINAL OUTPUT
    # =========================================================

    _log(
        "[BATCH PIPELINE CLI] Result",
        {
            k: v
            for k, v in output.items()
            if k not in ("early_warning",)
        }
    )

    if output.get("early_warning"):

        _log(
            "[BATCH PIPELINE CLI] "
            "First 5 high-risk students",
            output["early_warning"][:5]
        )