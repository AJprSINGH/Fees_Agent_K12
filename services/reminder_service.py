# ==========================================
# GENERATE MESSAGE FUNCTION
# ==========================================
def generate_message(
    parent_name,
    student_name,
    amount,
    due_date,
    risk
):

    try:

        # ==========================================
        # SAFE DEFAULT VALUES
        # ==========================================
        parent_name = (
            str(parent_name).strip()
            if parent_name
            else "Parent"
        )

        student_name = (
            str(student_name).strip()
            if student_name
            else "Student"
        )

        amount = (
            float(amount)
            if amount
            else 0
        )

        due_date = (
            str(due_date)
            if due_date
            else "N/A"
        )

        risk = (
            str(risk).upper().strip()
            if risk
            else "GREEN"
        )

        # ==========================================
        # YELLOW RISK
        # ==========================================
        if risk == "YELLOW":

            return {
                "status": "success",
                "risk_tier": "YELLOW",
                "channel": [
                    "SMS",
                    "EMAIL",
                    "WHATSAPP"
                ],
                "message": (
                    f"Hi {parent_name}, "
                    f"fee payment of ₹{amount:.2f} "
                    f"for {student_name} "
                    f"is pending. "
                    f"Due date: {due_date}. "
                    f"Please pay on time."
                )
            }

        # ==========================================
        # RED RISK
        # ==========================================
        if risk == "RED":

            return {
                "status": "success",
                "risk_tier": "RED",
                "channel": [
                    "SMS",
                    "EMAIL"
                ],
                "message": (
                    f"Reminder: {student_name}'s "
                    f"fee of ₹{amount:.2f} "
                    f"is overdue. "
                    f"Kindly clear the dues immediately."
                )
            }

        # ==========================================
        # CRITICAL RISK
        # ==========================================
        if risk == "CRITICAL":

            return {
                "status": "success",
                "risk_tier": "CRITICAL",
                "channel": [
                    "SMS",
                    "EMAIL",
                    "WHATSAPP"
                ],
                "message": (
                    f"Urgent Notice: Outstanding fee "
                    f"of ₹{amount:.2f} "
                    f"for {student_name}. "
                    f"Please contact the school finance desk immediately."
                )
            }

        # ==========================================
        # GREEN / DEFAULT
        # ==========================================
        return {
            "status": "success",
            "risk_tier": "GREEN",
            "channel": [],
            "message": "No action required"
        }

    except Exception as error:

        raise Exception(
            f"Error while generating message: {str(error)}"
        )