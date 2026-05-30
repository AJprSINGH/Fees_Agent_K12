from fastapi import APIRouter

router = APIRouter()

@router.post("/lead/reply")
def reply(data: dict):
    return {"reply": "sent"}

@router.get("/lead/reply/{agent_id}")
def get_reply(agent_id: str):
    return {"agent_id": agent_id}