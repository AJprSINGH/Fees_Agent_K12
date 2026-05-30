import pandas as pd

from utils.excel_loader import load_parents


# ==========================================
# LOAD PARENT DATA
# ==========================================
def get_parents() -> pd.DataFrame:

    try:

        # ==========================================
        # LOAD PARENTS FROM EXCEL LOADER
        # ==========================================
        response = load_parents()

        # ==========================================
        # ERROR HANDLING
        # ==========================================
        if response["status"] != "success":

            print(
                f"PARENT LOAD ERROR: "
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

            print("PARENT DATAFRAME EMPTY")

            return pd.DataFrame()

        # ==========================================
        # REQUIRED COLUMNS
        # ==========================================
        required_columns = [
            "student_id",
            "parent_name",
            "mobile_number"
        ]

        # ==========================================
        # CREATE MISSING COLUMNS
        # ==========================================
        for column in required_columns:

            if column not in df.columns:

                df[column] = ""

        # ==========================================
        # SAFE TYPE CONVERSION
        # ==========================================
        df["student_id"] = (
            df["student_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df["parent_name"] = (
            df["parent_name"]
            .fillna("Parent")
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
        # CLEAN INVALID VALUES
        # ==========================================
        invalid_values = [
            "",
            "nan",
            "none",
            "null"
        ]

        # ==========================================
        # FIX PARENT NAME
        # ==========================================
        df.loc[
            df["parent_name"]
            .str.lower()
            .isin(invalid_values),
            "parent_name"
        ] = "Parent"

        # ==========================================
        # REMOVE INVALID STUDENT IDS
        # ==========================================
        df = df[
            ~df["student_id"]
            .str.lower()
            .isin(invalid_values)
        ]

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
        # FINAL RESPONSE DATAFRAME
        # ==========================================
        return df[[
            "student_id",
            "parent_name",
            "mobile_number"
        ]]

    except Exception as error:

        print(
            f"Error in get_parents: "
            f"{str(error)}"
        )

        return pd.DataFrame()