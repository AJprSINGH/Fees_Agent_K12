from pydantic import BaseModel, Field

class FeeCreate(BaseModel):
    agent_id: str = Field(example="uuid-from-supabase")  
    student_id: str = Field(example="student-uuid")
    total_amount: float = Field(example=50000)
    paid_amount: float = Field(example=20000)
    pending_amount: float = Field(example=30000)
    status: str = Field(example="pending")