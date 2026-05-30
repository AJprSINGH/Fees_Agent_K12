from fastapi import APIRouter, HTTPException
from database.supabase_client import supabase
from models.email import EmailSend

router = APIRouter()


#  SEND EMAIL 
@router.post("/send")
def send_email(data: EmailSend):
    payload = {
        "agent_id": data.agent_id,
        "email": data.email,
        "subject": data.subject,
        "status": "sent"
    }

    try:
        response = supabase.table("email_logs").insert(payload).execute()

        return {
            "status": "sent",
            "data": response.data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#  GET EMAIL LOGS BY AGENT
@router.get("/{agent_id}")
def get_email(agent_id: str):
    try:
        response = supabase.table("email_logs") \
            .select("*") \
            .eq("agent_id", agent_id) \
            .execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="No email logs found")

        return {
            "status": "success",
            "count": len(response.data),
            "data": response.data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))