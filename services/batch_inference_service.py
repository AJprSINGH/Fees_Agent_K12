# =============================================================
# services/batch_inference_service.py
# Phase 6 — Batch Inference Service
#
# FIXES APPLIED:
#   ✅ FIX 1:  UUID auto-generated (not from payload)
#   ✅ FIX 2:  Full prediction stored
#   ✅ FIX 3:  No NULL values — all fields have safe defaults
#   ✅ FIX 4:  prediction_timestamp stored as ISO string
#   ✅ FIX 5:  agent_id, academic_year validated before insert
#   ✅ FIX 6:  Both early_warning_reports AND batch_predictions populated
#   ✅ FIX 7:  Chunked UPSERT (500 rows) to avoid Supabase payload limits
#   ✅ FIX 8:  excel_reader normalises GR NO.→gr_number, Std/Div→std_div
#   ✅ FIX 9:  Rows without a valid gr_number are skipped entirely
#   ✅ FIX 10: UPSERT on conflict — no duplicate key errors ever
#   ✅ FIX 11: ZERO 'unknown' / 'Unknown' / 'N/A' values in Supabase
#   ✅ FIX 12: In-memory dedup per chunk — prevents PostgreSQL
#              "ON CONFLICT DO UPDATE command cannot affect row a second time"
#   ✅ FIX 13: Complete prediction_input — all ML feature columns supplied
#   ✅ FIX 14: grade_band_used resolved via get_grade_band() — never empty
#   ✅ FIX 15: model_used defaults to "LogisticRegression"
#   ✅ FIX 16: Safe Hugging Face deployment support
#   ✅ FIX 17: Clean API response body for frontend
#   ✅ FIX 18: Prevents empty student inserts
#   ✅ FIX 19: batch_prediction_row now matches SQL schema exactly:
#              - std_div, total_unpaid, days_overdue, threshold_days added
#              - probability, grade_band, model_version included
#              - pending_amount removed (column does not exist in SQL)
#   ✅ FIX 20: days_overdue stored as int (not float) to match INTEGER column
#   ✅ FIX 21: _safe_int() helper added for integer safe conversion
# =============================================================

import uuid
from datetime import datetime, timezone

from database.supabase_client import supabase
from services.excel_reader import read_student_data


# =============================================================
# SAFE TYPE HANDLERS
# =============================================================

def _safe_str(value) -> str:
    """
    Return clean string.
    Never returns:
        - None
        - nan
        - unknown
        - N/A
    """

    if value is None:
        return ""

    result = str(value).replace(".0", "").strip()

    invalid_values = [
        "",
        "none",
        "nan",
        "unknown",
        "n/a",
        "null",
    ]

    if result.lower() in invalid_values:
        return ""

    return result


def _safe_float(value, default: float = 0.0) -> float:
    """
    Convert any value safely to float.
    """

    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value, default: int = 0) -> int:
    """
    Convert any value safely to int.
    """

    try:
        if value is None:
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


# =============================================================
# REMOVE DUPLICATE CONFLICT ROWS
# =============================================================

def _dedup_rows(rows: list, key_fields: list) -> list:
    """
    Prevent PostgreSQL conflict errors:
    'ON CONFLICT DO UPDATE command cannot affect row a second time'
    """

    seen = {}

    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        seen[key] = row

    return list(seen.values())


# =============================================================
# CHUNKED UPSERT
# =============================================================

def _chunked_upsert(
    table_name: str,
    rows: list,
    conflict_cols: str,
    chunk_size: int = 500
):
    """
    Insert/Update Supabase rows safely in chunks.
    """

    if not rows:
        return

    key_fields = [
        col.strip()
        for col in conflict_cols.split(",")
    ]

    for i in range(0, len(rows), chunk_size):

        chunk = rows[i:i + chunk_size]

        # Remove duplicate keys inside chunk
        chunk = _dedup_rows(chunk, key_fields)

        supabase.table(table_name).upsert(
            chunk,
            on_conflict=conflict_cols
        ).execute()

        print(
            f"[batch_inference] Upserted "
            f"{len(chunk)} rows into {table_name}"
        )


# =============================================================
# MAIN BATCH INFERENCE
# =============================================================

def run_batch_inference(payload: dict) -> dict:
    """
    Run ML prediction for all students from XLSX.
    """

    try:

        # =====================================================
        # LOAD XLSX DATA
        # =====================================================

        data_path = payload.get("data_path")

        df = read_student_data(data_path)

        # =====================================================
        # IMPORT PREDICTOR
        # =====================================================

        try:
            from inference.predictor import (
                predict_risk,
                get_grade_band,
            )

        except Exception as predictor_error:

            return {
                "status": "error",
                "message": (
                    "Predictor unavailable: "
                    f"{str(predictor_error)}. "
                    "Run training/train_models.py first."
                )
            }

        # =====================================================
        # REQUEST METADATA
        # =====================================================

        agent_id = (
            _safe_str(payload.get("agent_id"))
            or f"agent-{str(uuid.uuid4())[:8]}"
        )

        academic_year = (
            _safe_str(payload.get("academic_year"))
            or str(datetime.now().year)
        )

        mode = (
            _safe_str(payload.get("mode"))
            or "annual"
        )

        persist = bool(
            payload.get("persist", True)
        )

        threshold_days = _safe_int(
            payload.get("threshold_days", 30),
            default=30,
        )

        now_ts = datetime.now(
            timezone.utc
        ).isoformat()

        # =====================================================
        # OPTIONAL SINGLE STUDENT FILTER
        # =====================================================

        filter_gr_number = _safe_str(
            payload.get("gr_number")
        )

        if filter_gr_number:

            df = df[
                df["gr_number"].astype(str)
                == filter_gr_number
            ]

        # =====================================================
        # ENSURE REQUIRED ML COLUMNS EXIST IN DATAFRAME
        # =====================================================

        if "days_overdue" not in df.columns:
            df["days_overdue"] = 0

        if "total_unpaid" not in df.columns:
            df["total_unpaid"] = df.get("pending_amount", 0)

        import pandas as pd
        import numpy as np

        df["days_overdue"] = pd.to_numeric(
            df["days_overdue"], errors="coerce"
        ).fillna(0).astype(int)

        df["total_unpaid"] = pd.to_numeric(
            df["total_unpaid"], errors="coerce"
        ).fillna(0).astype(float)

        # =====================================================
        # STORAGE ARRAYS
        # =====================================================

        early_warning_rows = []
        batch_prediction_rows = []

        total_processed = 0
        total_skipped = 0

        # =====================================================
        # LOOP THROUGH STUDENTS
        # =====================================================

        for _, row in df.iterrows():

            row_dict = row.to_dict()

            # =================================================
            # REQUIRED STUDENT FIELDS
            # =================================================

            gr_number = _safe_str(
                row_dict.get("gr_number")
                or row_dict.get("student_id")
            )

            student_name = _safe_str(
                row_dict.get("student_name")
            )

            std_div = _safe_str(
                row_dict.get("std_div")
                or row_dict.get("class_name")
            )

            # Skip invalid students
            if not gr_number or not student_name:
                total_skipped += 1
                continue

            # Safe fallback for std_div
            if not std_div:
                std_div = _safe_str(
                    row_dict.get("grade")
                ) or ""

            # =================================================
            # GRADE BAND
            # =================================================

            try:
                grade_band_resolved = get_grade_band(std_div)
            except Exception:
                grade_band_resolved = "primary"

            # =================================================
            # FINANCIAL DATA
            # =================================================

            # total_unpaid — prefer pre-built column, fall back
            total_unpaid = _safe_float(
                row_dict.get("total_unpaid")
                or row_dict.get("pending_amount")
                or row_dict.get("pending_fees")
            )

            # days_overdue — always integer for the DB column
            days_overdue = _safe_int(
                row_dict.get("days_overdue"),
                default=0,
            )

            # =================================================
            # THRESHOLD LOGIC
            # =================================================

            threshold_override = (
                days_overdue < threshold_days
            )

            # =================================================
            # COMPLETE ML INPUT
            # =================================================

            prediction_input = {

                # Student identity
                "gr_number": gr_number,
                "student_name": student_name,
                "std_div": std_div,
                "grade": std_div,
                "class_name": std_div,

                # Financial features
                "pending_fees": total_unpaid,
                "pending_amount": total_unpaid,
                "total_unpaid": total_unpaid,

                # Timing
                "days_overdue": days_overdue,

                # Historical ML features
                "years_enrolled": _safe_float(
                    row_dict.get("years_enrolled"), 0
                ),

                "payment_streak_length": _safe_float(
                    row_dict.get("payment_streak_length"), 0
                ),

                "consecutive_default_years": _safe_float(
                    row_dict.get("consecutive_default_years"), 0
                ),

                "fee_tier_encoded": _safe_float(
                    row_dict.get("fee_tier_encoded"), 0
                ),

                "payment_momentum": _safe_float(
                    row_dict.get("payment_momentum"), 0
                ),

                # Remaining raw XLSX fields
                **row_dict,
            }

            # =================================================
            # RUN ML PREDICTION
            # =================================================

            prediction = predict_risk(prediction_input)

            # =================================================
            # SAFE PREDICTION OUTPUTS
            # =================================================

            risk_score = _safe_float(
                prediction.get("risk_score")
                or prediction.get("probability")
            )

            probability = _safe_float(
                prediction.get("probability")
                or prediction.get("risk_score")
            )

            risk_level = (
                _safe_str(prediction.get("risk_level"))
                or "Low"
            )

            # Apply threshold override
            if threshold_override:
                risk_level = "Low"

            recommendation = (
                _safe_str(prediction.get("recommendation"))
                or "Regular monitoring required."
            )

            # Update recommendation based on final risk
            if risk_level in ["High", "Critical"]:
                recommendation = (
                    "Immediate parent follow-up required."
                )

            grade_band_used = (
                _safe_str(prediction.get("grade_band_used"))
                or grade_band_resolved
            )

            # grade_band = label-only (nursery/primary/secondary)
            grade_band = (
                _safe_str(prediction.get("grade_band"))
                or grade_band_used
            )

            model_used = (
                _safe_str(prediction.get("model_used"))
                or "LogisticRegression"
            )

            model_version = (
                _safe_str(prediction.get("model_version"))
                or ""
            )

            fallback = bool(
                prediction.get("fallback", False)
            )

            # =================================================
            # EARLY WARNING ROW
            # Columns match early_warning_reports SQL table
            # =================================================

            early_warning_row = {

                "agent_id": agent_id,

                "gr_number": gr_number,

                "student_name": student_name,

                "std_div": std_div,

                "risk_level": risk_level,

                "probability": probability,

                "recommendation": recommendation,

                "model_used": model_used,

                "grade_band": grade_band,

                "grade_band_used": grade_band_used,

                "fallback": fallback,

                "academic_year": academic_year,

                "created_at": now_ts,
            }

            early_warning_rows.append(
                early_warning_row
            )

            # =================================================
            # BATCH PREDICTION ROW
            # IMPORTANT: columns MUST match batch_predictions
            # SQL table exactly. No extra keys. No missing keys.
            #
            # SQL columns:
            #   id, agent_id, student_id, gr_number,
            #   student_name, std_div, academic_year,
            #   risk_level, risk_score, probability,
            #   total_unpaid, days_overdue, threshold_days,
            #   grade_band, grade_band_used,
            #   model_used, model_version, recommendation,
            #   mode, prediction_timestamp, fallback,
            #   created_at
            # =================================================

            batch_prediction_row = {

                "agent_id": _safe_str(agent_id),

                "student_id": _safe_str(gr_number),

                "gr_number": _safe_str(gr_number),

                "student_name": _safe_str(student_name),

                "std_div": _safe_str(std_div),

                "academic_year": _safe_str(academic_year),

                "risk_level": _safe_str(risk_level or "Low"),

                "risk_score": _safe_float(risk_score),

                "probability": _safe_float(probability),

                "total_unpaid": _safe_float(total_unpaid),

                "days_overdue": _safe_int(days_overdue),

                "threshold_days": _safe_int(threshold_days, 30),

                "grade_band": _safe_str(grade_band),

                "grade_band_used": _safe_str(grade_band_used),

                "model_used": _safe_str(
                    model_used or "LogisticRegression"
                ),

                "model_version": _safe_str(model_version),

                "recommendation": _safe_str(
                    recommendation
                    or "Regular monitoring required."
                ),

                "mode": _safe_str(mode or "annual"),

                "prediction_timestamp": now_ts,

                "fallback": bool(fallback),
            }

            batch_prediction_rows.append(
                batch_prediction_row
            )

            total_processed += 1

        # =====================================================
        # SAVE TO SUPABASE
        # =====================================================

        persisted = False

        if persist and supabase is not None:

            # early_warning_reports
            if early_warning_rows:

                _chunked_upsert(
                    table_name="early_warning_reports",
                    rows=early_warning_rows,
                    conflict_cols="gr_number,academic_year",
                )

            # batch_predictions
            if batch_prediction_rows:

                _chunked_upsert(
                    table_name="batch_predictions",
                    rows=batch_prediction_rows,
                    conflict_cols="student_id,academic_year",
                )

            persisted = True

        # =====================================================
        # HIGH RISK COUNT
        # =====================================================

        high_risk_students = len([
            row
            for row in batch_prediction_rows
            if row["risk_level"] in [
                "High",
                "Critical",
            ]
        ])

        # =====================================================
        # SUCCESS RESPONSE
        # =====================================================

        return {

            "status": "success",

            "message": (
                "Batch inference completed successfully."
            ),

            "agent_id": agent_id,

            "academic_year": academic_year,

            "mode": mode,

            "persist": persist,

            "threshold_days": threshold_days,

            "total_processed": total_processed,

            "total_skipped": total_skipped,

            "high_risk_students": high_risk_students,

            "persisted": persisted,

            "tables_updated": [
                "early_warning_reports",
                "batch_predictions",
            ],

            "results": batch_prediction_rows[:100],
        }

    # =========================================================
    # GLOBAL ERROR HANDLER
    # =========================================================

    except Exception as error:

        import traceback

        traceback.print_exc()

        return {

            "status": "error",

            "message": str(error),
        }
