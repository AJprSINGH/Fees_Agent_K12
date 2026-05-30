# ==========================================
# GENERATE ESCALATION FUNCTION
# ==========================================
def generate_escalation(
    student_name,
    risk
):

    try:

        # ==========================================
        # SAFE DEFAULT VALUES
        # ==========================================
        student_name = (
            str(student_name).strip()
            if student_name
            else "Student"
        )

        risk = (
            str(risk).upper().strip()
            if risk
            else "GREEN"
        )

        # ==========================================
        # RED RISK ESCALATION
        # ==========================================
        if risk == "RED":

            return {
                "status": "success",
                "risk_tier": "RED",
                "teacher_escalation": True,
                "finance_head": True,
                "ptm_required": False,
                "notice_required": False,
                "priority": "HIGH",
                "message": (
                    f"Red escalation generated "
                    f"for {student_name}"
                )
            }

        # ==========================================
        # CRITICAL RISK ESCALATION
        # ==========================================
        if risk == "CRITICAL":

            return {
                "status": "success",
                "risk_tier": "CRITICAL",
                "teacher_escalation": True,
                "finance_head": True,
                "ptm_required": True,
                "notice_required": True,
                "priority": "URGENT",
                "message": (
                    f"Critical escalation generated "
                    f"for {student_name}"
                )
            }

        # ==========================================
        # DEFAULT / GREEN / YELLOW
        # ==========================================
        return {
            "status": "success",
            "risk_tier": risk,
            "teacher_escalation": False,
            "finance_head": False,
            "ptm_required": False,
            "notice_required": False,
            "priority": "NORMAL",
            "message": "No escalation required"
        }

    except Exception as error:

        raise Exception(
            f"Error while generating escalation: {str(error)}"
        )