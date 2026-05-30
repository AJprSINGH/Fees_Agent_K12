from fastapi import APIRouter, HTTPException
from database.supabase_client import supabase
from models.student import StudentCreate
from utils.excel_loader import load_students

router = APIRouter()


# =========================================
# CREATE STUDENT
# =========================================
@router.post("/create")
def create_student(data: StudentCreate):

    try:

        # =========================================
        # CHECK EXISTING STUDENT
        # =========================================
        existing_student = (
            supabase
            .table("students")
            .select("*")
            .eq("student_id", data.student_id)
            .execute()
        )

        if existing_student.data:

            return {
                "status": "error",
                "message": "Student already exists",
                "student_id": data.student_id
            }

        # =========================================
        # INSERT PAYLOAD
        # =========================================
        payload = {
            "agent_id": data.agent_id,
            "student_id": data.student_id,
            "name": data.name,
            "mobile": data.mobile,
            "class_name": data.class_name
        }

        # =========================================
        # INSERT DATA
        # =========================================
        response = (
            supabase
            .table("students")
            .insert(payload)
            .execute()
        )

        return {
            "status": "success",
            "message": "Student created successfully",
            "data": response.data
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================================
# GET ALL STUDENTS FROM SUPABASE
# =========================================
@router.get("/all")
def get_all_students(agent_id: str):

    try:

        response = (
            supabase
            .table("students")
            .select("*")
            .eq("agent_id", agent_id)
            .execute()
        )

        return {
            "status": "success",
            "count": len(response.data),
            "data": response.data
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================================
# GET ONE STUDENT
# =========================================
@router.get("/{student_id}")
def get_one_student(student_id: str):

    try:

        response = (
            supabase
            .table("students")
            .select("*")
            .eq("student_id", student_id)
            .execute()
        )

        if not response.data:

            return {
                "status": "error",
                "message": "Student not found",
                "student_id": student_id
            }

        return {
            "status": "success",
            "data": response.data[0]
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================================
# GET STUDENTS FROM EXCEL
# =========================================
@router.get("/excel/all")
def get_students_from_excel():

    try:

        # =========================================
        # LOAD STUDENTS EXCEL
        # =========================================
        students_response = load_students()

        # =========================================
        # ERROR HANDLING
        # =========================================
        if students_response["status"] != "success":

            return students_response

        # =========================================
        # GET DATAFRAME
        # =========================================
        students_df = students_response["data"]

        # =========================================
        # SUCCESS RESPONSE
        # =========================================
        return {
            "status": "success",
            "message": "Students fetched successfully",
            "total_students": len(students_df),
            "students": students_df.to_dict(
                orient="records"
            )
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )