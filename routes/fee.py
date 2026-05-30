from fastapi import APIRouter, HTTPException
from database.supabase_client import supabase
from models.fee import FeeCreate

router = APIRouter()

# =========================
# CREATE FEE (UPDATED)
# =========================
@router.post("/create")
def create_fee(data: FeeCreate):
    try:
        payload = {
            "agent_id": data.agent_id,
            "student_id": data.student_id,
            "total_amount": data.total_amount,
            "paid_amount": data.paid_amount,
            "pending_amount": data.pending_amount,
            "status": data.status
        }

        res = supabase.table("fees").insert(payload).execute()

        return {
            "status": "success",
            "data": res.data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# FILTER BY STUDENT + AGENT (UNCHANGED)
# =========================
@router.get("/student/{student_id}")
def get_fee(student_id: str, agent_id: str):
    try:
        res = supabase.table("fees") \
            .select("*") \
            .eq("student_id", student_id) \
            .eq("agent_id", agent_id) \
            .execute()

        if not res.data:
            raise HTTPException(status_code=404, detail="Fee record not found")

        return {
            "status": "success",
            "count": len(res.data),
            "data": res.data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))