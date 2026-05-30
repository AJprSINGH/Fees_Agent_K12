import sys
import os
import io
import json

from pathlib import Path

# =====================================================
# PROJECT ROOT PATH
# =====================================================
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# =====================================================
# FASTAPI
# =====================================================
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# =====================================================
# ROUTES
# =====================================================
from routes import (

    student,

    fee,

    email,

    k12_routes
)

# =====================================================
# APSCHEDULER  (Phase 6 — scheduled retraining)
# =====================================================
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    print("[SCHEDULER] APScheduler not installed — periodic retraining disabled. "
          "Add apscheduler to requirements.txt to enable.")

_scheduler = None


def _run_scheduled_training():
    """
    Annual retraining job — runs train_save_and_register() on the
    default dataset and writes every band to the model registry.
    Designed to be triggered by APScheduler at the start of each
    academic year (default: 1 April 00:00 UTC).
    """
    DATA_PATH = os.environ.get(
        "TRAINING_DATA_PATH",
        "data/Fees Defaulter Report - Overall.xlsx"
    )

    print("\n[SCHEDULER] Annual retraining job started.")

    if not os.path.exists(DATA_PATH):
        print(f"[SCHEDULER] Training data not found at {DATA_PATH} — job aborted.")
        return

    try:
        import pandas as pd
        from training.train_models import train_save_and_register

        df_raw = pd.read_excel(DATA_PATH)
        print(f"[SCHEDULER] Loaded {len(df_raw)} rows from {DATA_PATH}")

        result = train_save_and_register(
            df            = df_raw,
            dataset_range = "2017-2018 to 2024-2025",
            notes         = "APScheduler annual retrain",
            auto_promote  = True,
        )

        bands = result.get("bands", {})
        print(f"[SCHEDULER] Retraining complete — bands: "
              f"{[b for b, r in bands.items() if r.get('status') == 'success']}")
        print(f"[SCHEDULER] Registry: {result.get('registry', {})}")

    except Exception as e:
        import traceback
        print(f"[SCHEDULER] Retraining job FAILED: {e}")
        traceback.print_exc()


# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(

    title="K12 ERP Agent APIs",

    version="2.0.0",

    docs_url="/",

    redoc_url="/redoc"
)

# =====================================================
# CORS MIDDLEWARE
# Required so Laravel frontend (different origin)
# can POST multipart/form-data to this FastAPI server.
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production: e.g. ["https://your-laravel.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# STARTUP CHECK
# =====================================================
@app.on_event("startup")
def startup_check():

    print(
        "========== STARTUP XLSX CHECK =========="
    )

    data_dir = Path("data")

    # =================================================
    # DATA FOLDER CHECK
    # =================================================
    if not data_dir.exists():

        print(
            "DATA FOLDER NOT FOUND"
        )

    else:

        xlsx_files = list(
            data_dir.glob("*.xlsx")
        )

        if not xlsx_files:

            print(
                "NO XLSX FILES FOUND"
            )

        else:

            for file in xlsx_files:

                print(
                    "FOUND XLSX:",
                    file.name
                )

    print(
        "========================================"
    )

    # =================================================
    # PHASE 6 — APSCHEDULER  (annual retraining cron)
    # Fires on 1 April at 00:00 UTC each year.
    # Override RETRAIN_CRON env var to customise:
    #   e.g. RETRAIN_CRON="0 0 1 6 *"  → 1 June
    # =================================================
    global _scheduler

    if SCHEDULER_AVAILABLE:
        cron_expr = os.environ.get("RETRAIN_CRON", "0 0 1 4 *")  # April 1 00:00 UTC
        try:
            parts = cron_expr.split()
            trigger = CronTrigger(
                minute=parts[0],
                hour=parts[1],
                day=parts[2],
                month=parts[3],
                day_of_week=parts[4] if len(parts) > 4 else "*",
            )
            _scheduler = BackgroundScheduler(timezone="UTC")
            _scheduler.add_job(
                _run_scheduled_training,
                trigger=trigger,
                id="annual_retrain",
                replace_existing=True,
            )
            _scheduler.start()
            print(f"[SCHEDULER] Annual retraining scheduled — cron: {cron_expr}")
        except Exception as sched_err:
            print(f"[SCHEDULER] Failed to start scheduler: {sched_err}")
    else:
        print("[SCHEDULER] APScheduler not available — install with: pip install apscheduler")


# =====================================================
# BASIC APIs
# =====================================================
app.include_router(

    student.router,

    prefix="/api/student",

    tags=["Student"]
)

app.include_router(

    fee.router,

    prefix="/api/fee",

    tags=["Fees"]
)

app.include_router(

    email.router,

    prefix="/api/email",

    tags=["Email"]
)

# =====================================================
# K12 AI AGENT APIs
# =====================================================
app.include_router(

    k12_routes.router,

    prefix="/k12/agents",

    tags=["K12 Agents"]
)

# =====================================================
# PROCESS EXCEL — Laravel DataTable Integration
# Accepts a .xlsx file uploaded via multipart/form-data
# from the Laravel frontend (SheetJS-generated Blob).
# Reads it into a pandas DataFrame, runs defaulter
# agent logic, and returns a JSON summary.
#
# Laravel JS usage:
#   const formData = new FormData();
#   formData.append('file', blob, 'datatable_export.xlsx');
#   formData.append('meta', JSON.stringify({...}));
#   fetch('/api/process-excel', { method: 'POST', body: formData });
#
# IMPORTANT: Do NOT set Content-Type manually in JS —
# let the browser set the multipart boundary automatically.
# =====================================================
@app.post("/api/process-excel", tags=["Excel Integration"])
async def process_excel(
    file: UploadFile = File(...),
    meta: str = Form(None)
):
    """
    Receive a .xlsx file from the Laravel DataTable export,
    parse it with pandas, and return row count + column names
    + a 5-row preview.  Pass agent_id in meta JSON to also
    run the defaulter agent on the uploaded data.
    """
    # ─── 1. Validate file type ──────────────────────────────
    allowed_types = [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/octet-stream",  # some browsers send this for .xlsx
    ]
    filename = file.filename or ""
    if (
        file.content_type not in allowed_types
        and not filename.lower().endswith((".xlsx", ".xls"))
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {file.content_type}. "
                "Please upload a .xlsx or .xls file."
            ),
        )

    # ─── 2. Read bytes → pandas DataFrame ──────────────────
    try:
        import pandas as pd
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        df = pd.read_excel(io.BytesIO(contents), engine="openpyxl")
        df = df.dropna(how="all").dropna(axis=1, how="all")

        if df.empty:
            raise HTTPException(status_code=422, detail="Excel file has no data rows.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse Excel file: {e}")

    # ─── 3. Parse optional metadata ─────────────────────────
    meta_dict = {}
    if meta:
        try:
            meta_dict = json.loads(meta)
        except Exception:
            meta_dict = {"raw": meta}

    # ─── 4. Optionally run defaulter agent ──────────────────
    agent_result = None
    agent_id = meta_dict.get("agent_id")
    if agent_id:
        try:
            from services.k12_agents import defaulter as defaulter_agent
            agent_payload = {
                "agent_id": agent_id,
                "student_id": meta_dict.get("student_id"),
                "threshold_days": int(meta_dict.get("threshold_days", 7)),
                "class_name": meta_dict.get("class_name"),
            }
            agent_result = defaulter_agent.run_defaulter(agent_payload)
        except Exception as agent_err:
            agent_result = {"status": "error", "message": str(agent_err)}

    # ─── 5. Build and return response ───────────────────────
    excel_data = []
    try:
        # Replace NaN values with empty strings
        clean_df = df.fillna("")

        # Convert the full Excel file into JSON response
        excel_data = clean_df.to_dict(orient="records")

    except Exception as e:
        excel_data = []

    return {
        "status": "success",
        "filename": filename,
        "rows_received": len(df),
        "columns": df.columns.tolist(),

        # Complete Excel Data
        "excel_data": excel_data,

        "meta_received": meta_dict,
        "agent_result": agent_result,
    }



@app.get("/")
def root():

    return {

        "status": "success",

        "message": "K12 ERP Agent APIs running successfully",

        "version": "2.0.0",

        "docs": "/",

        "redoc": "/redoc",

        "available_modules": [

            "Student APIs",

            "Fees APIs",

            "Email APIs",

            "K12 AI Agents",

            "Excel Integration (POST /api/process-excel)"
        ]
    }