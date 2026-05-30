import pandas as pd

from utils.excel_loader import load_students


# ==========================================
# LOAD STUDENT DATA
# ==========================================
def get_students() -> pd.DataFrame:

    try:

        # ==========================================
        # LOAD STUDENTS FROM EXCEL LOADER
        # ==========================================
        response = load_students()

        # ==========================================
        # HANDLE LOAD ERROR
        # ==========================================
        if response["status"] != "success":

            print(
                f"STUDENT LOAD ERROR: "
                f"{response['message']}"
            )

            return pd.DataFrame()

        # ==========================================
        # GET DATAFRAME
        # ==========================================
        df = response["data"]

        # ==========================================
        # EMPTY DATAFRAME SAFETY
        # ==========================================
        if df is None or df.empty:

            print("STUDENT DATAFRAME EMPTY")

            return pd.DataFrame()

        # ==========================================
        # REQUIRED COLUMNS
        # ==========================================
        required_columns = [
            "student_id",
            "student_name",
            "class_name",
            "roll_number",
            "quota",
            "status",
            "mobile_number"
        ]

        # ==========================================
        # CREATE MISSING COLUMNS
        # ==========================================
        for column in required_columns:

            if column not in df.columns:

                df[column] = ""

        # ==========================================
        # SAFE TYPE CLEANUP
        # ==========================================
        df["student_id"] = (
            df["student_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df["student_name"] = (
            df["student_name"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df["class_name"] = (
            df["class_name"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df["roll_number"] = (
            df["roll_number"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df["quota"] = (
            df["quota"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df["status"] = (
            df["status"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df["mobile_number"] = (
            df["mobile_number"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        # ==========================================
        # INVALID VALUES
        # ==========================================
        invalid_values = [
            "",
            "nan",
            "none",
            "null"
        ]

        # ==========================================
        # REMOVE INVALID STUDENT IDS
        # ==========================================
        df = df[
            ~df["student_id"]
            .str.lower()
            .isin(invalid_values)
        ]

        # ==========================================
        # DEFAULT VALUES
        # ==========================================
        df.loc[
            df["student_name"] == "",
            "student_name"
        ] = "Unknown Student"

        df.loc[
            df["class_name"] == "",
            "class_name"
        ] = "Unknown Class"

        df.loc[
            df["status"] == "",
            "status"
        ] = "ACTIVE"

        # ==========================================
        # REMOVE DUPLICATES
        # ==========================================
        df = df.drop_duplicates(
            subset=["student_id"]
        )

        # ==========================================
        # RESET INDEX
        # ==========================================
        df = df.reset_index(
            drop=True
        )

        # ==========================================
        # FINAL DATAFRAME
        # ==========================================
        return df[[
            "student_id",
            "student_name",
            "class_name",
            "roll_number",
            "quota",
            "status",
            "mobile_number"
        ]]

    except Exception as error:

        print(
            f"Error in get_students: "
            f"{str(error)}"
        )

        return pd.DataFrame()