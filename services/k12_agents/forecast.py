from database.supabase_client import supabase
from typing import List


# ===============================
#  REVENUE FORECAST FUNCTION
# ===============================
def run(data):
    try:
        # =========================
        # FLEXIBLE INPUT SUPPORT
        # =========================
        history = []
        agent_id = None

        # Support dict and object
        if isinstance(data, dict):
            history = data.get("history", [])
            agent_id = data.get("agent_id")
        else:
            history = getattr(data, "history", [])
            agent_id = getattr(data, "agent_id", None)

        # =========================
        # VALIDATION
        # =========================
        if not history or len(history) == 0:
            return {
                "status": "error",
                "message": "History data is required"
            }

        # =========================
        # CALCULATION (FINAL LOGIC)
        # =========================
        total = sum(history)
        avg = total / len(history)

        # Growth logic (20%)
        next_month_prediction = round(avg * 1.2, 2)

        prediction = {
            "next_month": next_month_prediction,
            "average": round(avg, 2),
            "months_considered": len(history)
        }

        # =========================
        # DB SAVE (FIXED TABLE)
        # =========================
        if agent_id:
            supabase.table("forecasts").insert({
                "agent_id": agent_id,
                "history": history,
                "months": len(history)
            }).execute()

        # =========================
        # FINAL RESPONSE
        # =========================
        return {
            "status": "success",
            "payload_used": {
                "agent_id": agent_id,
                "history_count": len(history)
            },
            "data": prediction,
            "forecast": round(avg, 2),  # backward compatibility
            "insights": [
                "Average-based revenue forecast",
                "20% growth applied for next month prediction"
            ],
            "recommended_actions": [
                "Increase collection efficiency",
                "Monitor monthly trends",
                "Adjust fee strategy if needed"
            ],
            "confidence_score": 0.9
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ===============================
#  TARGET REVENUE STRUCTURE
# ===============================
def generate_structure(agent_id: str, target_revenue: float, students: int = 100):
    try:
        # =========================
        # VALIDATION
        # =========================
        if not agent_id:
            return {
                "status": "error",
                "message": "agent_id is required"
            }

        if target_revenue <= 0:
            return {
                "status": "error",
                "message": "Target revenue must be greater than 0"
            }

        if students <= 0:
            return {
                "status": "error",
                "message": "Student count must be greater than 0"
            }

        # =========================
        # CALCULATION
        # =========================
        recommended_fee = round(target_revenue / students, 2)

        structure = {
            "agent_id": agent_id,
            "target_revenue": target_revenue,
            "students": students,
            "recommended_fee_per_student": recommended_fee
        }

        # =========================
        # DB SAVE (UNCHANGED - VALID)
        # =========================
        supabase.table("revenue_structures").insert(structure).execute()

        # =========================
        # FINAL RESPONSE
        # =========================
        return {
            "status": "success",
            "payload_used": {
                "agent_id": agent_id,
                "target_revenue": target_revenue,
                "students": students
            },
            "data": structure,
            "insights": [
                "Revenue distributed evenly across students",
                "Helps in pricing strategy planning"
            ],
            "recommended_actions": [
                "Review affordability before applying fee",
                "Segment students if needed",
                "Adjust based on actual collections"
            ],
            "confidence_score": 0.95
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }