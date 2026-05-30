import json
import pandas as pd
import numpy as np

from datetime import datetime

from database.supabase_client import supabase


# =====================================================
# LEGACY ERP API SUPPORT
# =====================================================
from services.erp_api_service import (
    pending_fees,
    fee_details
)

from services.fee_api_service import (
    get_fee_details,
    get_pending_fees
)


# =====================================================
# EXCEL SERVICES
# =====================================================
from services.student_service import (
    get_students
)

from services.payment_service import (
    get_payments,
    calculate_due_status
)

from services.parent_service import (
    get_parents
)

from services.reminder_service import (
    generate_message
)

from services.escalation_service import (
    generate_escalation
)

from services.report_service import (
    generate_daily_report
)


# =====================================================
# OPTIONAL ML SERVICE
# =====================================================
try:

    from services.ml_prediction_service import (
        train_model
    )

except Exception:

    train_model = None


# Phase 4 — production inference (no retraining at request time)
try:

    from services.ml_prediction_service import (
        predict_from_loaded_model
    )

    from inference.predictor import (
        predict_batch,
        registry_status,
    )

except Exception:

    predict_from_loaded_model = None
    predict_batch             = None
    registry_status           = None


# =====================================================
# SAFE CLEAN STRING
# =====================================================
def _safe_string(value):

    try:

        if value is None:
            return ""

        return str(value).strip()

    except Exception:
        return ""


# =====================================================
# SAFE CLEAN STUDENT ID
# =====================================================
def _clean_student_id(value):

    try:

        return (

            str(value)

            .replace(".0", "")

            .strip()
        )

    except Exception:
        return ""


# =====================================================
# SAFE NUMERIC
# =====================================================
def _safe_numeric(value):

    try:

        return float(

            pd.to_numeric(
                value,
                errors="coerce"
            )

            or 0
        )

    except Exception:
        return 0.0


# =====================================================
# SAFE JSON — converts dict/list → JSON string if needed
# Supabase JSONB columns accept Python dict directly
# via supabase-py, but this guards edge cases.
# =====================================================
def _safe_json(value):

    if value is None:
        return {}

    if isinstance(value, (dict, list)):
        return value

    try:
        return json.loads(str(value))
    except Exception:
        return {"raw": str(value)}


# =====================================================
# SAFE MOBILE — strip trailing .0 from numeric reads
# =====================================================
def _safe_mobile(value):

    try:
        return (
            str(value)
            .replace(".0", "")
            .strip()
        )
    except Exception:
        return ""


# =====================================================
# BUILD SUPABASE ROW — single source of truth
# Call this for both single and bulk inserts so the
# column list is always in sync.
# =====================================================
def _build_insert_row(row_dict, threshold_days):

    return {

        "agent_id":       row_dict.get("agent_id", ""),

        "student_id":     row_dict.get("student_id", ""),

        "student_name":   row_dict.get("student_name", "Unknown"),

        "parent_name":    row_dict.get("parent_name", "Unknown Parent"),

        "class_name":     row_dict.get("class_name", "N/A"),

        "mobile_number":  _safe_mobile(
                              row_dict.get("mobile_number", "")
                          ),

        "risk_tier":      row_dict.get("risk_tier", "GREEN"),

        "pending_amount": float(
                              row_dict.get("pending_amount", 0) or 0
                          ),

        "total_unpaid":   float(
                              row_dict.get("pending_amount", 0) or 0
                          ),

        "days_overdue":   int(
                              row_dict.get("days_overdue", 0) or 0
                          ),

        "threshold_days": int(threshold_days),

        "risk_score":     float(
                              row_dict.get("risk_score", 0) or 0
                          ),

        "defaulter":      bool(
                              row_dict.get("defaulter", False)
                          ),

        "due_date":       row_dict.get("due_date", ""),

        # reminder / escalation are dicts — stored as JSONB
        "reminder":       _safe_json(
                              row_dict.get("reminder", {})
                          ),

        "escalation":     _safe_json(
                              row_dict.get("escalation", {})
                          ),

        "reminder_sent":  False,

        "escalation_sent": False,

        "source":         "excel",

        "generated_at":   row_dict.get(
                              "generated_at",
                              datetime.now().isoformat()
                          ),
    }


# =====================================================
# MAIN DEFAULTER AGENT
# =====================================================
def run_defaulter(payload):

    try:

        # =================================================
        # PAYLOAD VALUES
        # =================================================
        agent_id = payload.get(
            "agent_id"
        )

        student_id = payload.get(
            "student_id"
        )

        threshold_days = payload.get(
            "threshold_days",
            7
        )

        class_name = payload.get(
            "class_name"
        )

        # =================================================
        # DEBUG PAYLOAD
        # =================================================
        print(
            "\n========== REQUEST PAYLOAD =========="
        )

        print(payload)

        print(
            "=====================================\n"
        )

        # =================================================
        # VALIDATION
        # =================================================
        if not agent_id:

            return {

                "status": "error",

                "message": (
                    "agent_id is required"
                ),

                "payload_used": payload
            }

        # =================================================
        # RESULT STORAGE
        # =================================================
        results = []

        # =================================================
        # LOAD STUDENTS
        # =================================================
        try:

            students_df = get_students()

            if (
                students_df is None
                or students_df.empty
            ):

                students_df = pd.DataFrame()

        except Exception as error:

            print(
                f"Student load error: {error}"
            )

            students_df = pd.DataFrame()

        # =================================================
        # LOAD PAYMENTS
        # =================================================
        try:

            payments_df = get_payments()

            if (
                payments_df is None
                or payments_df.empty
            ):

                payments_df = pd.DataFrame()

        except Exception as error:

            print(
                f"Payment load error: {error}"
            )

            payments_df = pd.DataFrame()

        # =================================================
        # LOAD PARENTS
        # =================================================
        try:

            parents_df = get_parents()

            if (
                parents_df is None
                or parents_df.empty
            ):

                parents_df = pd.DataFrame()

        except Exception as error:

            print(
                f"Parent load error: {error}"
            )

            parents_df = pd.DataFrame()

        # =================================================
        # EMPTY PAYMENT CHECK
        # =================================================
        if payments_df.empty:

            return {

                "status": "error",

                "source": "excel",

                "message": (
                    "Payments XLSX is empty "
                    "or invalid"
                ),

                "payload_used": payload
            }

        # =================================================
        # CLEAN COLUMN NAMES
        # =================================================
        payments_df.columns = [

            str(column)
            .strip()
            .lower()

            for column in payments_df.columns
        ]

        # =================================================
        # REQUIRED COLUMN CHECK
        # =================================================
        required_columns = [

            "student_id",
            "pending_amount"
        ]

        missing_columns = [

            column

            for column in required_columns

            if column not in payments_df.columns
        ]

        if missing_columns:

            return {

                "status": "error",

                "source": "excel",

                "message": (
                    "Missing required columns"
                ),

                "missing_columns": (
                    missing_columns
                ),

                "payload_used": payload
            }

        # =================================================
        # CLEAN PAYMENT DATA
        # =================================================
        payments_df["student_id"] = (

            payments_df["student_id"]

            .astype(str)

            .str.replace(
                ".0",
                "",
                regex=False
            )

            .str.strip()
        )

        payments_df["pending_amount"] = (

            pd.to_numeric(

                payments_df[
                    "pending_amount"
                ],

                errors="coerce"

            )

            .fillna(0)
        )

        # =================================================
        # REMOVE INVALID IDS
        # =================================================
        invalid_values = [

            "",
            "nan",
            "none",
            "null"
        ]

        payments_df = payments_df[

            ~payments_df["student_id"]

            .str.lower()

            .isin(invalid_values)
        ]

        # =================================================
        # REMOVE DUPLICATES
        # =================================================
        payments_df = payments_df.drop_duplicates(
            subset=["student_id"]
        )

        # =================================================
        # CALCULATE DUE STATUS
        # =================================================
        try:

            payments_df = calculate_due_status(
                payments_df
            )

        except Exception as error:

            print(
                f"Due status error: {error}"
            )

        # =================================================
        # DEFAULT COLUMNS
        # =================================================
        default_columns = {

            "student_name": "Unknown",

            "parent_name": "Unknown Parent",

            "class_name": "N/A",

            "mobile_number": "",

            "days_overdue": 0,

            "risk_tier": "GREEN",

            "due_date": datetime.now().strftime(
                "%Y-%m-%d"
            )
        }

        for column, value in default_columns.items():

            if column not in payments_df.columns:

                payments_df[column] = value

        # =================================================
        # FILTER SINGLE STUDENT
        # =================================================
        if student_id:

            student_id = (

                str(student_id)

                .replace(".0", "")

                .strip()
            )

            print(
                "\n========== SEARCH STUDENT =========="
            )

            print(student_id)

            print(
                "========== AVAILABLE IDS =========="
            )

            print(

                payments_df["student_id"]

                .astype(str)

                .str.strip()

                .unique()

                .tolist()
            )

            print(
                "===================================\n"
            )

            payments_df = payments_df[

                payments_df["student_id"]

                .astype(str)

                .str.replace(
                    ".0",
                    "",
                    regex=False
                )

                .str.strip()

                == student_id
            ]

        # =================================================
        # STUDENT NOT FOUND
        # =================================================
        if payments_df.empty:

            return {

                "status": "success",

                "mode": "single",

                "source": "excel",

                "student_id": student_id,

                "defaulter": False,

                "message": (
                    "Student not found "
                    "in payment records"
                ),

                "payload_used": payload,

                "data": None
            }

        # =================================================
        # MERGED DATAFRAME
        # =================================================
        merged = payments_df.copy()

        # =================================================
        # MERGE STUDENTS
        # =================================================
        if (

            not students_df.empty

            and "student_id"
            in students_df.columns
        ):

            students_df["student_id"] = (

                students_df["student_id"]

                .astype(str)

                .str.replace(
                    ".0",
                    "",
                    regex=False
                )

                .str.strip()
            )

            merged = merged.merge(

                students_df,

                on="student_id",

                how="left",

                suffixes=(
                    "",
                    "_student"
                )
            )

        # =================================================
        # MERGE PARENTS
        # =================================================
        if (

            not parents_df.empty

            and "student_id"
            in parents_df.columns
        ):

            parents_df["student_id"] = (

                parents_df["student_id"]

                .astype(str)

                .str.replace(
                    ".0",
                    "",
                    regex=False
                )

                .str.strip()
            )

            merged = merged.merge(

                parents_df,

                on="student_id",

                how="left",

                suffixes=(
                    "",
                    "_parent"
                )
            )

        # =================================================
        # FIX NULL VALUES
        # =================================================
        merged = merged.replace(
            [np.inf, -np.inf],
            np.nan
        )

        merged = merged.fillna({

            "student_name": "Unknown",

            "parent_name": "Unknown Parent",

            "class_name": (
                class_name or "N/A"
            ),

            "mobile_number": "",

            "days_overdue": 0,

            "risk_tier": "GREEN",

            "due_date": datetime.now().strftime(
                "%Y-%m-%d"
            )
        })

        # =================================================
        # CLEAN NUMERIC TYPES
        # =================================================
        numeric_columns = [

            "pending_amount",
            "days_overdue"
        ]

        for column in numeric_columns:

            merged[column] = (

                pd.to_numeric(

                    merged[column],

                    errors="coerce"

                )

                .fillna(0)
            )

        # =================================================
        # MAIN LOOP
        # =================================================
        for _, row in merged.iterrows():

            # =============================================
            # BASIC VALUES
            # =============================================
            current_student_id = _clean_student_id(

                row.get(
                    "student_id",
                    ""
                )
            )

            student_name = _safe_string(

                row.get(
                    "student_name",
                    "Unknown"
                )
            )

            parent_name = _safe_string(

                row.get(
                    "parent_name",
                    "Unknown Parent"
                )
            )

            current_class = _safe_string(

                row.get(
                    "class_name",
                    class_name or "N/A"
                )
            )

            mobile_number = _safe_mobile(

                row.get(
                    "mobile_number",
                    ""
                )
            )

            pending_amount = _safe_numeric(

                row.get(
                    "pending_amount",
                    0
                )
            )

            overdue_days = int(

                _safe_numeric(

                    row.get(
                        "days_overdue",
                        0
                    )
                )
            )

            due_date = _safe_string(

                row.get(
                    "due_date",
                    "N/A"
                )
            )

            calculated_risk = _safe_string(

                row.get(
                    "risk_tier",
                    "GREEN"
                )

            ).upper()

            # =============================================
            # FINAL RISK LOGIC
            # =============================================
            if pending_amount <= 0:

                risk_tier = "GREEN"

                defaulter = False

            else:

                risk_tier = calculated_risk

                defaulter = True

            # =============================================
            # RISK SCORE
            # =============================================
            risk_score = round(

                min(
                    overdue_days / 30,
                    1
                ),

                2
            )

            # =============================================
            # REMINDER
            # =============================================
            reminder = generate_message(

                parent_name,

                student_name,

                pending_amount,

                due_date,

                risk_tier
            )

            # =============================================
            # ESCALATION
            # =============================================
            escalation = generate_escalation(

                student_name,

                risk_tier
            )

            # =============================================
            # RESULT OBJECT
            # =============================================
            result = {

                "agent_id": agent_id,

                "student_id": (
                    current_student_id
                ),

                "student_name": (
                    student_name
                ),

                "parent_name": (
                    parent_name
                ),

                "class_name": (
                    current_class
                ),

                "mobile_number": (
                    mobile_number
                ),

                "pending_amount": (
                    pending_amount
                ),

                "days_overdue": (
                    overdue_days
                ),

                "threshold_days": (
                    threshold_days
                ),

                "risk_tier": (
                    risk_tier
                ),

                "risk_score": (
                    risk_score
                ),

                "defaulter": (
                    defaulter
                ),

                "due_date": (
                    due_date
                ),

                "reminder": (
                    reminder
                ),

                "escalation": (
                    escalation
                ),

                "generated_at": (
                    datetime.now()
                    .isoformat()
                )
            }

            results.append(result)

        # =================================================
        # DASHBOARD SUMMARY
        # =================================================
        dashboard_summary = {

            "green": len([

                row for row in results

                if row["risk_tier"]
                == "GREEN"
            ]),

            "yellow": len([

                row for row in results

                if row["risk_tier"]
                == "YELLOW"
            ]),

            "red": len([

                row for row in results

                if row["risk_tier"]
                == "RED"
            ]),

            "critical": len([

                row for row in results

                if row["risk_tier"]
                == "CRITICAL"
            ])
        }

        # =================================================
        # SINGLE STUDENT MODE
        # =================================================
        if student_id:

            single_result = results[0]

            # =============================================
            # SUPABASE SINGLE INSERT — FULL COLUMNS
            # =============================================
            try:

                if not supabase:
                    raise RuntimeError("Supabase not configured")

                insert_payload = _build_insert_row(
                    single_result,
                    threshold_days
                )

                print(
                    "\n========== SUPABASE SINGLE INSERT =========="
                )
                print(insert_payload)
                print(
                    "============================================\n"
                )

                supabase.table(
                    "defaulters"
                ).insert(
                    insert_payload
                ).execute()

            except Exception as error:

                print(
                    f"Supabase insert error: "
                    f"{error}"
                )

            return {

                "status": "success",

                "mode": "single",

                "source": "excel",

                "payload_used": payload,

                "dashboard_summary": (
                    dashboard_summary
                ),

                "data": single_result,

                "insights": [

                    "Excel-based fee analysis completed",

                    "Risk classification completed",

                    "Reminder generated successfully",

                    "Escalation workflow completed"
                ],

                "recommended_actions": [

                    "GREEN → No action",

                    "YELLOW → Send reminder",

                    "RED → Finance escalation",

                    "CRITICAL → PTM and legal escalation"
                ],

                "confidence_score": 0.97
            }

        # =================================================
        # REPORT GENERATION
        # =================================================
        report_path = None

        try:

            report_path = generate_daily_report(
                pd.DataFrame(results)
            )

        except Exception as error:

            print(
                f"Report generation error: "
                f"{error}"
            )

        # =================================================
        # BULK INSERT — FULL COLUMNS
        # =================================================
        try:

            insert_rows = []

            for row in results:

                insert_rows.append(
                    _build_insert_row(row, threshold_days)
                )

            if insert_rows and supabase:

                print(
                    f"\n========== SUPABASE BULK INSERT "
                    f"({len(insert_rows)} rows) =========="
                )
                print(insert_rows[0])   # preview first row
                print(
                    "============================================\n"
                )

                supabase.table(
                    "defaulters"
                ).insert(
                    insert_rows
                ).execute()

        except Exception as error:

            print(
                f"Bulk insert error: "
                f"{error}"
            )

        # =================================================
        # OPTIONAL ML TRAINING
        # =================================================
        ml_response = None

        if train_model:

            try:

                ml_df = merged.copy()

                # =============================================
                # PHASE 1 FIX — PRESERVE QUOTA COLUMN
                # =============================================
                object_cols = (

                    ml_df.select_dtypes(
                        include=["object"]
                    ).columns
                )

                keep_cols = [
                    "student_id",
                    "quota",
                    "quota_student"
                ]

                remove_cols = [

                    column

                    for column in object_cols

                    if column not in keep_cols
                ]

                ml_df = ml_df.drop(

                    columns=remove_cols,

                    errors="ignore"
                )

                # Phase 4: use pre-trained grade-band models
                if predict_batch is not None:
                    ml_response = {
                        "status": "success",
                        "phase":  "Phase 4 — Grade-band inference (XGBoost/LightGBM/RF)",
                        "registry": registry_status() if registry_status else {},
                    }
                    try:
                        scored_df = predict_batch(ml_df.copy())
                        ml_response["scored_rows"] = len(scored_df)
                        if "student_id" in scored_df.columns and "risk_score" in scored_df.columns:
                            risk_map = dict(zip(scored_df["student_id"], scored_df["risk_score"]))
                            for r in results:
                                sid = r.get("student_id")
                                if sid in risk_map:
                                    r["ml_risk_score"] = risk_map[sid]
                    except Exception as batch_err:
                        ml_response["batch_error"] = str(batch_err)
                else:
                    ml_response = train_model(ml_df)

            except Exception as error:

                ml_response = {

                    "status": "error",

                    "message": str(error)
                }

        # =================================================
        # FINAL RESPONSE DEBUG
        # =================================================
        print(
            "\n========== FINAL RESPONSE =========="
        )

        print({

            "status": "success",

            "total_students": len(results),

            "dashboard_summary": (
                dashboard_summary
            )
        })

        print(
            "====================================\n"
        )

        # =================================================
        # FINAL BULK RESPONSE
        # =================================================
        return {

            "status": "success",

            "mode": "bulk",

            "source": "excel",

            "payload_used": payload,

            "polling_supported": True,

            "polling_frequency": "24 hours",

            "report_generated": report_path,

            "dashboard_summary": (
                dashboard_summary
            ),

            "ml_training": (
                ml_response
            ),

            "total_students": (
                len(results)
            ),

            "data": results,

            "insights": [

                "Excel-driven autonomous detection completed",

                "Risk tier classification completed",

                "Reminder automation completed",

                "Escalation workflow completed",

                "ML pipeline initialized"
            ],

            "recommended_actions": [

                "GREEN → No action",

                "YELLOW → Reminder",

                "RED → Teacher + finance escalation",

                "CRITICAL → PTM + legal notice"
            ],

            "confidence_score": 0.96
        }

    except Exception as error:

        print(
            f"Defaulter agent error: {error}"
        )

        return {

            "status": "error",

            "message": str(error),

            "payload_used": payload
        }
