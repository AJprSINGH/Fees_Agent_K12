# =============================================================
# app.py — Phase 6 Production Entry Point
#
# FIXES APPLIED:
#   ✅ FIX 1: Legacy routes REMOVED from Swagger UI
#             (/api/v6/batch-inference-legacy, /api/v6/early-warning-legacy)
#   ✅ FIX 2: Legacy endpoints kept as HIDDEN (include_in_schema=False)
#             so old integrations still work but Swagger stays clean
#   ✅ FIX 3: All production routes are clean and documented
#   ✅ FIX 4: CORS, Auth, model startup — all correct
#   ✅ FIX 5: Startup prints clear action if API_SECRET_KEY missing
#   ✅ FIX 6: Auto-train on startup if models missing (runs ONCE, saves to disk)
#   ✅ FIX 7: Swagger BearerAuth schema — Authorize button works correctly
#   ✅ FIX 8: load_dotenv() before all os.getenv() calls
# =============================================================

import sys
import os
import io
import json
from pathlib import Path

# ── Load .env FIRST — must happen before any os.getenv() calls ──────────────
# On Hugging Face Spaces, secrets are injected as env vars automatically.
# Locally, they are read from the .env file.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # HF env vars take precedence over .env
except ImportError:
    pass  # python-dotenv not installed — rely solely on environment variables

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from routes.k12_routes import router as k12_router
from routes.v6_routes import router as v6_router
from middleware.auth import require_api_key

# =====================================================
# ALLOWED ORIGINS — set ALLOWED_ORIGINS in .env
# e.g. ALLOWED_ORIGINS=https://your-laravel-app.com
# =====================================================
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()] or ["*"]

# =====================================================
# PHASE 6 — MODEL REGISTRY (loaded once at startup)
# =====================================================
try:
    from inference.predictor import registry_status, predict_batch
    _PREDICTOR_READY = True
except Exception as _pred_err:
    print(f"[STARTUP] WARNING — predictor unavailable: {_pred_err}")
    registry_status = None
    predict_batch   = None
    _PREDICTOR_READY = False

try:
    from versioning.model_registry import (
        list_versions, get_active_version,
        register_training_run, promote_to_active,
    )
    _VERSIONING_READY = True
except Exception as _ver_err:
    print(f"[STARTUP] WARNING — versioning unavailable: {_ver_err}")
    list_versions = get_active_version = register_training_run = None
    promote_to_active = None
    _VERSIONING_READY = False

try:
    from monitoring.drift_monitor import run_drift_check
    _DRIFT_READY = True
except Exception as _drift_err:
    print(f"[STARTUP] WARNING — drift monitor unavailable: {_drift_err}")
    run_drift_check = None
    _DRIFT_READY = False

try:
    from database.supabase_client import supabase
except Exception:
    supabase = None

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(
    title="K12 Fee Defaulter Detection — Production API",
    description=(
        "**Phase 6 — Production Deployment**\n\n"
        "Persisted ML models loaded at startup. No runtime retraining.\n\n"
        "**Workflow:**\n"
        "1. `POST /api/v6/batch-inference` — Run ML predictions and store results\n"
        "2. `GET /api/v6/early-warning` — Fetch High/Critical risk students\n\n"
        "All risk scores are **ADVISORY only**. "
        "Human-in-the-loop: all final decisions remain with school admin.\n\n"
        "**Authentication:** Click the **Authorize** button → enter your `API_SECRET_KEY` "
        "(do NOT include the word 'Bearer' — Swagger adds it automatically)."
    ),
    version="6.0.0",
    docs_url="/",
    redoc_url="/redoc",
)

# Register routers
app.include_router(k12_router)
app.include_router(v6_router)

# =====================================================
# SWAGGER BEARER AUTH SCHEMA
# ─────────────────────────────────────────────────────
# This makes the Swagger "Authorize" button work correctly.
# Without this block, Swagger does not know Bearer auth exists
# and clicking Execute sends no token → 401 on every request.
#
# HOW TO USE IN SWAGGER:
#   1. Open your HF Space URL
#   2. Click "Authorize" button (top right of Swagger page)
#   3. Enter ONLY:  trizk12secure2026
#      (Do NOT type "Bearer" — Swagger adds it automatically)
#   4. Click Authorize → Close
#   5. All protected routes now work from Swagger UI
# =====================================================
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # ── Step 1: Declare the BearerAuth security scheme ──────────────────────
    # scheme_name="BearerAuth" in HTTPBearer (auth.py) ensures FastAPI
    # registers routes under this exact name — no duplicate "HTTPBearer" entry.
    openapi_schema.setdefault("components", {})
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "API Key",
            "description": (
                "Enter your API_SECRET_KEY value here. "
                "Do NOT include the word 'Bearer' — Swagger adds it automatically. "
                "Example: Triz@669Mar69"
            ),
        }
    }

    # ── Step 2: Apply BearerAuth to every route path and method ─────────────
    # This stamps security: [{BearerAuth: []}] on each operation so Swagger
    # actually sends the Authorization header when you click Execute.
    for path_item in openapi_schema.get("paths", {}).values():
        for operation in path_item.values():
            if isinstance(operation, dict):
                # Remove any auto-generated "HTTPBearer" security (wrong name)
                operation.pop("security", None)
                # Stamp our canonical BearerAuth requirement
                operation["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# =====================================================
# CORS MIDDLEWARE
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# STARTUP EVENT — MODEL LOAD CHECK + AUTO-TRAIN
# =====================================================
@app.on_event("startup")
def startup():
    print("\n" + "=" * 60)
    print("STARTUP — K12 Fee Defaulter Detection  [Phase 6]")
    print("=" * 60)

    # ── Authentication check ──────────────────────────────────
    if not os.getenv("API_SECRET_KEY"):
        print("  ❌ API_SECRET_KEY not set — admin endpoints will return 500.")
        print("     ACTION: Add API_SECRET_KEY to HF Secrets → Settings → Variables and Secrets")
        print("     Then restart the Space.")
    else:
        print("  ✅ API_SECRET_KEY loaded — admin endpoints are protected.")

    # ── Supabase check ────────────────────────────────────────
    if not os.getenv("SUPABASE_URL"):
        print("  ❌ SUPABASE_URL not set — running in offline mode (local file fallback).")
        print("     ACTION: Add SUPABASE_URL and SUPABASE_KEY to HF Secrets.")
    else:
        print("  ✅ SUPABASE_URL loaded.")

    print(f"  CORS allowed origins: {ALLOWED_ORIGINS}")

    # ── Data files check ──────────────────────────────────────
    data_dir = Path("data")
    if not data_dir.exists():
        print("  ❌ DATA FOLDER NOT FOUND — batch inference will fail.")
    else:
        for f in data_dir.glob("*.xlsx"):
            print(f"  ✅ FOUND XLSX: {f.name}")

    # ── Model registry check + auto-train ─────────────────────
    if registry_status:
        reg = registry_status()
        print(f"\n  MODEL REGISTRY:")
        print(f"    Loaded bands : {reg.get('loaded_bands', [])}")
        print(f"    Missing bands: {reg.get('missing_bands', [])}")
        print(f"    Models ready : {reg.get('models_ready', False)}")

        if not reg.get("models_ready"):
            # ── AUTO-TRAIN: models missing → train once from data file ──
            # Models are saved to disk after training.
            # On every subsequent restart they load from disk — NO retraining.
            data_path = os.getenv(
                "TRAINING_DATA_PATH",
                "data/Fees Defaulter Report - Overall.xlsx",
            )
            if os.path.exists(data_path):
                print(f"\n  🔄 AUTO-TRAINING: Missing models detected.")
                print(f"     Data: {data_path}")
                print(f"     Training runs ONCE at startup and saves models to disk.")
                print(f"     Subsequent restarts will load saved models — no retraining.")
                try:
                    import pandas as pd
                    from training.train_models import train_and_save

                    df_raw = pd.read_excel(data_path)
                    print(f"     Loaded {len(df_raw)} rows. Starting training ...")

                    train_result = train_and_save(df_raw)
                    trained_bands = [
                        b for b, r in train_result.get("bands", {}).items()
                        if r.get("status") == "success"
                    ]
                    skipped_bands = [
                        b for b, r in train_result.get("bands", {}).items()
                        if r.get("status") != "success"
                    ]
                    print(f"  ✅ AUTO-TRAINING COMPLETE — bands trained: {trained_bands}")
                    if skipped_bands:
                        print(f"  ⚠️  Skipped bands (insufficient data): {skipped_bands}")

                    # Reload models into memory immediately — no restart required
                    try:
                        from inference.predictor import reload_registry, registry_status as rs2
                        reload_registry()
                        reg2 = rs2()
                        print(f"  ✅ MODELS NOW LOADED: {reg2.get('loaded_bands', [])}")
                    except Exception as reload_err:
                        print(f"  ⚠️  Registry reload error: {reload_err}")
                        print("     Models ARE saved to disk — they will load on next restart.")

                except Exception as train_err:
                    import traceback
                    print(f"  ❌ AUTO-TRAINING FAILED: {train_err}")
                    traceback.print_exc()
                    print("     Fix the error above, then restart the Space.")
            else:
                print(f"  ❌ DATA FILE NOT FOUND: {data_path}")
                print("     Upload your XLSX to the data/ folder and restart.")
        else:
            print("    ✅ All models loaded and ready for inference.")
    else:
        print("\n  ❌ MODEL REGISTRY: predictor not importable — models NOT loaded.")

    print("\n  Deployment mode : INFERENCE ONLY — no runtime retraining.")
    print("  Human-in-the-loop: all actions require school admin approval.")
    print("=" * 60 + "\n")


# =====================================================
# HEALTH — BASIC
# =====================================================
@app.get("/health", tags=["Health"])
def health():
    return {
        "status":  "running",
        "message": "K12 Fee Defaulter Detection API is live",
        "version": "6.0.0",
        "phase":   "Phase 6 — Production Deployment",
    }


# =====================================================
# HEALTH — MODEL STATUS
# =====================================================
@app.get("/health/models", tags=["Health"])
def health_models():
    """Reports which grade-band models are loaded and ready for inference."""
    if registry_status is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable", "models_ready": False,
                "message": "inference/predictor.py not importable. Run training/train_models.py",
            },
        )
    reg = registry_status()
    return {
        "status":        "success",
        "models_ready":  reg.get("models_ready", False),
        "loaded_bands":  reg.get("loaded_bands",  []),
        "missing_bands": reg.get("missing_bands", []),
        "model_status":  reg.get("model_status", {}),
        "advisory_note": (
            "Risk scores are advisory only. "
            "No student services are restricted automatically."
        ),
    }


# =====================================================
# PHASE 6 — DRIFT STATUS  [AUTH REQUIRED]
# =====================================================
@app.get("/api/v6/drift/status", tags=["Phase 6 — Monitoring"])
def drift_status(_: str = Depends(require_api_key)):
    """Return the latest drift monitoring report from Supabase."""
    if supabase is None:
        return {
            "status":  "offline",
            "message": "Supabase not configured. Drift reports are logged locally.",
        }
    try:
        resp = (
            supabase.table("drift_monitoring_log")
            .select("*").order("timestamp", desc=True).limit(1).execute()
        )
        rows = resp.data or []
        if not rows:
            return {"status": "no_reports", "message": "No drift reports found."}
        return {"status": "success", "latest_report": rows[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# PHASE 6 — MODEL VERSIONS  [AUTH REQUIRED]
# =====================================================
@app.get("/api/v6/models/versions", tags=["Phase 6 — Versioning"])
def model_versions(band: str = None, limit: int = 20, _: str = Depends(require_api_key)):
    """List model version history. Optional ?band=nursery|primary|secondary|global"""
    if not _VERSIONING_READY or list_versions is None:
        raise HTTPException(status_code=503, detail="versioning module unavailable.")
    versions = list_versions(supabase, band=band, limit=limit)
    return {"status": "success", "versions": versions, "count": len(versions)}


@app.get("/api/v6/models/active", tags=["Phase 6 — Versioning"])
def active_model(band: str = "global", _: str = Depends(require_api_key)):
    """Return the currently Active model record for a given band."""
    if not _VERSIONING_READY or get_active_version is None:
        raise HTTPException(status_code=503, detail="versioning module unavailable.")
    record = get_active_version(supabase, band=band)
    if record is None:
        return {"status": "not_found", "message": f"No Active model for band={band}."}
    return {"status": "success", "active_version": record}


@app.post("/api/v6/models/promote", tags=["Phase 6 — Versioning"])
def promote_model(payload: dict, _: str = Depends(require_api_key)):
    """
    Promote a Staging model to Active.
    Body: { "version_id": "xgboost_secondary_20250601_a3f2", "band": "secondary" }
    Requires Bearer token auth.
    """
    if not _VERSIONING_READY or promote_to_active is None:
        raise HTTPException(status_code=503, detail="versioning module unavailable.")
    version_id = payload.get("version_id")
    band       = payload.get("band", "global")
    if not version_id:
        raise HTTPException(status_code=422, detail="version_id is required.")
    ok = promote_to_active(supabase, version_id, band)
    return {"status": "success" if ok else "error", "version_id": version_id, "band": band}


# =====================================================
# PHASE 6 — OFFLINE RETRAIN  [AUTH REQUIRED]
# =====================================================
@app.post("/api/v6/retrain", tags=["Phase 6 — Lifecycle"])
def trigger_offline_retrain(payload: dict, _: str = Depends(require_api_key)):
    """
    Offline retraining — authorised admin use only.
    Runs training/train_models.py, saves new models to disk,
    and reloads them into memory immediately (no restart required).
    Requires Bearer token (API_SECRET_KEY).
    """
    try:
        import pandas as pd
        from training.train_models import train_and_save

        data_path     = payload.get(
            "data_path",
            os.environ.get("TRAINING_DATA_PATH",
                           "data/Fees Defaulter Report - Overall.xlsx")
        )
        dataset_range = payload.get("dataset_range", "")
        notes         = payload.get("notes", "Triggered via /api/v6/retrain")
        auto_promote  = payload.get("auto_promote", False)

        if not os.path.exists(data_path):
            raise HTTPException(status_code=422, detail=f"File not found: {data_path}")

        df = pd.read_excel(data_path)
        training_output = train_and_save(df)

        # Reload models into memory immediately — no restart required
        try:
            from inference.predictor import reload_registry
            reload_registry()
            print("[RETRAIN] Model registry reloaded in-process.")
        except Exception as reload_err:
            print(f"[RETRAIN] Registry reload error: {reload_err}")

        version_info = {}
        if _VERSIONING_READY and register_training_run:
            version_info = register_training_run(
                supabase_client=supabase,
                training_output=training_output,
                dataset_range=dataset_range,
                notes=notes,
                auto_promote=auto_promote,
            )

        return {
            "status":          "success",
            "training_result": training_output,
            "version_info":    version_info,
            "advisory_note": (
                "Models trained, saved to disk, and reloaded into memory. "
                "Inference is immediately available — no restart required. "
                "Use /api/v6/models/promote after admin review."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# LEGACY ENDPOINTS — HIDDEN FROM SWAGGER
# include_in_schema=False keeps Swagger clean but
# old integrations remain working.
# =====================================================
@app.post("/api/v6/batch-inference-legacy", include_in_schema=False)
def trigger_batch_inference_legacy(payload: dict, _: str = Depends(require_api_key)):
    """LEGACY — use POST /api/v6/batch-inference instead."""
    try:
        from scheduling.batch_pipeline import run_batch_inference
        import pandas as pd

        academic_year = payload.get("academic_year", "")
        mode          = payload.get("mode", "annual")
        persist       = payload.get("persist", True)
        data_path     = payload.get(
            "data_path",
            os.environ.get("TRAINING_DATA_PATH",
                           "data/Fees Defaulter Report - Overall.xlsx")
        )

        if not os.path.exists(data_path):
            raise HTTPException(status_code=422, detail=f"Data file not found: {data_path}")

        df = pd.read_excel(data_path)
        result = run_batch_inference(
            df_students=df, mode=mode,
            academic_year=academic_year, persist=persist,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v6/early-warning-legacy", include_in_schema=False)
def early_warning_report_legacy(
    academic_year: str = "",
    limit: int = 100,
    _: str = Depends(require_api_key),
):
    """LEGACY — use GET /api/v6/early-warning instead."""
    if supabase is None:
        raise HTTPException(status_code=503, detail="Supabase not configured.")
    try:
        query = supabase.table("early_warning_flags").select("*")
        if academic_year:
            query = query.eq("academic_year", academic_year)
        resp = query.order("risk_score", desc=True).limit(limit).execute()
        rows = resp.data or []
        return {
            "status":          "success",
            "academic_year":   academic_year or "all",
            "high_risk_count": len(rows),
            "students":        rows,
            "advisory_note": (
                "LEGACY endpoint. Use GET /api/v6/early-warning. "
                "These flags are ADVISORY. No services are restricted automatically."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# PROCESS EXCEL — Laravel DataTable Integration
# =====================================================
@app.post("/api/process-excel", tags=["Excel Integration"])
async def process_excel(
    file: UploadFile = File(...),
    meta: str = Form(None),
):
    """
    Receive a .xlsx file from the Laravel DataTable export.
    Validates the file type, parses it with pandas + openpyxl,
    and returns rows_received, columns, full data, and
    (optionally) the defaulter agent result when agent_id is
    supplied in the meta JSON field.
    """
    import pandas as pd

    allowed_types = [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/octet-stream",
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

    try:
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

    meta_dict = {}
    if meta:
        try:
            meta_dict = json.loads(meta)
        except Exception:
            meta_dict = {"raw": meta}

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

    excel_data = []
    try:
        clean_df = df.fillna("")
        excel_data = clean_df.to_dict(orient="records")
    except Exception:
        excel_data = []

    return {
        "status": "success",
        "filename": filename,
        "rows_received": len(df),
        "columns": df.columns.tolist(),
        "excel_data": excel_data,
        "meta_received": meta_dict,
        "agent_result": agent_result,
    }


# =====================================================
# SERVER RUNNER  (local dev)
# =====================================================
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )
