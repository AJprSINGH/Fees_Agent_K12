from pydantic import BaseModel, Field

class EmailSend(BaseModel):
    agent_id: str = Field(example="uuid-from-supabase")  
    email: str = Field(example="test@gmail.com")
    subject: str = Field(example="Fee Reminder")
    status: str = Field(example="sent")