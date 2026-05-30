from fastapi import APIRouter
from database.supabase_client import supabase

router = APIRouter()

@router.get("/{agent_name}")
def get_logs(agent_name: str):
    return supabase.table("agent_logs").select("*").eq("agent_name", agent_name).execute().data