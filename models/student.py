from pydantic import BaseModel, Field

class StudentCreate(BaseModel):
    agent_id: str = Field(example="uuid-from-supabase")  
    student_id: str = Field(example="student-uuid")
    name: str = Field(example="karan vekariya")
    mobile: str = Field(example="9510845927")
    class_name: str = Field(example="10")