from database.supabase_client import supabase

def log_agent(agent_name, input_data, output_data, status="success"):
    payload = {
        "agent_name": agent_name,
        "input": input_data,
        "output": output_data,
        "status": status
    }

    supabase.table("agent_logs").insert(payload).execute()