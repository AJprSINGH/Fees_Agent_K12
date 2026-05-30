from pydantic import BaseModel, Field

class AgentRegister(BaseModel):
    agent_id: str = Field(example="uuid-from-supabase")   
    agent_name: str = Field(example="defaulter_agent")
    agent_type: str = Field(example="finance")
    description: str = Field(example="Detect fee defaulters")