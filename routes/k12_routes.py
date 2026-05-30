from fastapi import (
    APIRouter,
    Body,
    Request
)

from models.k12_agent_models import (
    DefaulterInput,
    FraudInput,
    ForecastInput,
    NachInput,
    StructureInput
)

from services.k12_agents import (
    defaulter,
    fraud,
    forecast,
    nach
)

from database.supabase_client import supabase

router = APIRouter(
    prefix="/k12/agents",
    tags=["K12 Agents"]
)


# =====================================================
# UC-FEE-01  DEFAULTER API
# BUG FIX: was calling undefined run_defaulter()
# instead of returning the already-computed response.
# =====================================================
@router.post("/defaulter/run")
async def run_defaulter_api(
    payload: DefaulterInput
):
    try:
        payload_data = payload.dict()

        # GR Number support: use gr_number as student_id if student_id not set
        if payload_data.get("gr_number") and not payload_data.get("student_id"):
            payload_data["student_id"] = payload_data["gr_number"]

        print("========== API PAYLOAD ==========")
        print(payload_data)
        print("=================================")

        # Run agent and return its result directly
        response = defaulter.run_defaulter(payload_data)
        return response

    except Exception as error:
        print(f"Defaulter API error: {error}")
        return {
            "status": "error",
            "api": "UC-FEE-01",
            "message": str(error),
            "payload_used": payload.dict()
        }


# =====================================================
# FRAUD API
# =====================================================
@router.post("/fraud/analyze")
def fraud_check(data: FraudInput):
    try:
        payload_data = data.dict()
        result = fraud.run(payload_data)
        return {
            "status": "success",
            "message": "Fraud analysis completed",
            "payload_used": payload_data,
            "response": result
        }
    except Exception as error:
        print(f"Fraud API error: {error}")
        return {
            "status": "error",
            "message": str(error),
            "payload_used": data.dict()
        }


# =====================================================
# FORECAST API
# =====================================================
@router.post("/forecast/run")
def forecast_run(data: ForecastInput):
    try:
        payload_data = data.dict()
        result = forecast.run(payload_data)
        return {
            "status": "success",
            "message": "Forecast completed",
            "payload_used": payload_data,
            "response": result
        }
    except Exception as error:
        print(f"Forecast API error: {error}")
        return {
            "status": "error",
            "message": str(error),
            "payload_used": data.dict()
        }


# =====================================================
# NACH API
# =====================================================
@router.post("/nach/process")
def process_nach(data: NachInput):
    try:
        payload_data = data.dict()
        result = nach.run(payload_data)
        return {
            "status": "success",
            "message": "NACH processing completed",
            "payload_used": payload_data,
            "response": result
        }
    except Exception as error:
        print(f"NACH API error: {error}")
        return {
            "status": "error",
            "message": str(error),
            "payload_used": data.dict()
        }


# =====================================================
# FEE STRUCTURE API
# =====================================================
@router.post("/fee-structure/generate")
def generate_structure_api(data: StructureInput):
    try:
        payload_data = data.dict()
        estimated_students = 100
        per_student_fee = round(data.target_revenue / estimated_students, 2)
        structure = {
            "target_revenue": data.target_revenue,
            "estimated_students": estimated_students,
            "per_student_fee": per_student_fee
        }

        try:
            if supabase:
                supabase.table("revenue_targets").insert({
                    "agent_id": data.agent_id,
                    "target_revenue": data.target_revenue
                }).execute()
        except Exception as error:
            print(f"Revenue insert error: {error}")

        return {
            "status": "success",
            "message": "Fee structure generated",
            "payload_used": payload_data,
            "response": structure
        }
    except Exception as error:
        print(f"Fee structure API error: {error}")
        return {
            "status": "error",
            "message": str(error),
            "payload_used": data.dict()
        }


# =====================================================
# SIMPLE APIS
# =====================================================
@router.get("/defaulter/simple-run")
def run_defaulter_simple():
    try:
        payload = {
            "agent_id": "demo-agent",
            "student_id": None,
            "threshold_days": 7
        }
        response = defaulter.run_defaulter(payload)
        return {
            "status": "success",
            "payload_used": payload,
            "response": response
        }
    except Exception as error:
        return {"status": "error", "message": str(error)}


@router.post("/fraud/run")
def run_fraud_simple(data: list = Body(...)):
    try:
        result = fraud.run(data)
        return {"status": "success", "payload_used": data, "response": result}
    except Exception as error:
        return {"status": "error", "message": str(error), "payload_used": data}


@router.post("/forecast/simple-run")
def run_forecast_simple(data: dict):
    try:
        result = forecast.run(data)
        return {"status": "success", "payload_used": data, "response": result}
    except Exception as error:
        return {"status": "error", "message": str(error), "payload_used": data}


@router.post("/nach/run")
async def run_nach_simple(request: Request):
    try:
        data = await request.json()
        result = nach.run(data)
        return {"status": "success", "payload_used": data, "response": result}
    except Exception as error:
        return {"status": "error", "message": str(error)}


# =====================================================
# GET DEFAULTERS
# =====================================================
@router.get("/defaulters")
def get_defaulters(agent_id: str):
    try:
        if not supabase:
            return {"status": "error", "message": "Supabase not configured"}
        data = (
            supabase.table("defaulters")
            .select("*")
            .eq("agent_id", agent_id)
            .execute()
            .data
        )
        return {"status": "success", "agent_id": agent_id, "total_records": len(data), "data": data}
    except Exception as error:
        return {"status": "error", "message": str(error), "agent_id": agent_id}


@router.get("/payments")
def get_payments_api(agent_id: str):
    try:
        if not supabase:
            return {"status": "error", "message": "Supabase not configured"}
        data = (
            supabase.table("payments")
            .select("*")
            .eq("agent_id", agent_id)
            .execute()
            .data
        )
        return {"status": "success", "agent_id": agent_id, "total_records": len(data), "data": data}
    except Exception as error:
        return {"status": "error", "message": str(error), "agent_id": agent_id}


@router.get("/forecast")
def get_forecast(agent_id: str):
    try:
        if not supabase:
            return {"status": "error", "message": "Supabase not configured"}
        data = (
            supabase.table("forecasts")
            .select("*")
            .eq("agent_id", agent_id)
            .execute()
            .data
        )
        return {"status": "success", "agent_id": agent_id, "total_records": len(data), "data": data}
    except Exception as error:
        return {"status": "error", "message": str(error), "agent_id": agent_id}


@router.get("/nach")
def get_nach(agent_id: str):
    try:
        if not supabase:
            return {"status": "error", "message": "Supabase not configured"}
        data = (
            supabase.table("nach_records")
            .select("*")
            .eq("agent_id", agent_id)
            .execute()
            .data
        )
        return {"status": "success", "agent_id": agent_id, "total_records": len(data), "data": data}
    except Exception as error:
        return {"status": "error", "message": str(error), "agent_id": agent_id}


@router.get("/fee-structure")
def get_fee_structure(agent_id: str):
    try:
        if not supabase:
            return {"status": "error", "message": "Supabase not configured"}
        data = (
            supabase.table("revenue_targets")
            .select("*")
            .eq("agent_id", agent_id)
            .execute()
            .data
        )
        return {"status": "success", "agent_id": agent_id, "total_records": len(data), "data": data}
    except Exception as error:
        return {"status": "error", "message": str(error), "agent_id": agent_id}


# =============================================================
# Phase 5 — ML Evaluation & Explainability Endpoints
# =============================================================

@router.get("/ml/status")
def ml_model_status():
    """Registry health check — which grade-band models are loaded."""
    try:
        from inference.predictor import registry_status
        return {"status": "success", **registry_status()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/ml/explain")
def ml_explain_student(data: dict = Body(...)):
    """
    Phase 5: Return SHAP explanation for a single student.

    Body: flat student feature dict (same shape as defaulter payload).

    Response includes:
      risk_score, risk_level, recommendation,
      top_factors (SHAP), explanation_summary,
      confusion_matrix (from most recent holdout eval),
      calibration info.
    """
    try:
        from inference.predictor import predict_risk
        result = predict_risk(data)
        return {"status": "success", "prediction": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/ml/fairness")
async def ml_fairness_audit(request: Request):
    """
    Phase 5: Run a full fairness audit over a batch of students.

    Body: {"students": [ ...list of student dicts... ]}

    Returns per-grade-band FPR/FNR, disparate impact flags,
    and temporal stability summary.
    """
    try:
        import pandas as pd
        from inference.predictor import predict_batch
        from evaluation.evaluator import fairness_audit, compute_metrics

        body     = await request.json()
        students = body.get("students", [])

        if not students:
            return {"status": "error", "message": "No students provided."}

        df = pd.DataFrame(students)

        # Run batch inference (adds risk_score column)
        scored = predict_batch(df.copy())

        if "risk_score" not in scored.columns or "is_defaulter" not in scored.columns:
            return {
                "status": "error",
                "message": (
                    "Batch must include is_defaulter column for fairness audit. "
                    "risk_score is computed automatically."
                ),
            }

        y_true  = scored["is_defaulter"].fillna(0).astype(int)
        y_proba = scored["risk_score"].fillna(0).astype(float).values

        meta_cols = [c for c in ["grade_encoded", "division_encoded", "grade_band"]
                     if c in scored.columns]
        df_meta = scored[meta_cols].reset_index(drop=True)

        audit = fairness_audit(y_true, y_proba, df_meta)

        return {"status": "success", "fairness_audit": audit}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
