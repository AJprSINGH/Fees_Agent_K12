import pandas as pd
from pathlib import Path
from typing import Dict, Any
import os


# ==========================================
# BASE DIRECTORY CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

PAYMENTS_FILE = DATA_DIR / "Fees Defaulter Report - Overall.xlsx"
STUDENTS_FILE = DATA_DIR / "students.xlsx"
FEES_FILE     = DATA_DIR / "fee_structure.xlsx"
PARENTS_FILE  = DATA_DIR / "parents.xlsx"

# NEW FILES
SIBLINGS_FILE      = DATA_DIR / "siblings.xlsx"
COMMUNICATION_FILE = DATA_DIR / "communication.xlsx"


# ==========================================
# COMMON EXCEL LOADER
# ==========================================
def load_excel(
    file_path: Path,
    name: str = "file"
) -> Dict[str, Any]:

    try:

        if not Path(file_path).exists():
            return {
                "status": "error",
                "message": f"{name} XLSX not found",
                "path": str(file_path),
                "total_records": 0,
                "columns": [],
                "data": None
            }

        if Path(file_path).stat().st_size == 0:
            return {
                "status": "error",
                "message": f"{name} XLSX file is empty (0 bytes)",
                "path": str(file_path),
                "total_records": 0,
                "columns": [],
                "data": None
            }

        dataframe = pd.read_excel(
            file_path,
            engine="openpyxl",
            sheet_name=0
        )

        dataframe = dataframe.dropna(how="all")
        dataframe = dataframe.dropna(axis=1, how="all")

        # Strip + lowercase only (no underscore replace here)
        # so payment_service rename map still matches
        dataframe.columns = [
            str(col).strip().lower()
            for col in dataframe.columns
        ]

        if dataframe.empty:
            return {
                "status": "error",
                "message": f"{name} XLSX has no data rows",
                "path": str(file_path),
                "total_records": 0,
                "columns": [],
                "data": None
            }

        dataframe = dataframe.fillna("")

        return {
            "status": "success",
            "message": f"{name} XLSX loaded successfully",
            "path": str(file_path),
            "total_records": len(dataframe),
            "columns": list(dataframe.columns),
            "data": dataframe
        }

    except Exception as error:
        return {
            "status": "error",
            "message": f"Error while loading {name} XLSX: {str(error)}",
            "path": str(file_path),
            "total_records": 0,
            "columns": [],
            "data": None
        }


# ==========================================
# LOAD PAYMENTS
# BUG FIX: was returning raw DataFrame
# causing KeyError: 'status' in payment_service.
# Now returns standard {"status", "data"} dict.
# ==========================================
def load_payments() -> Dict[str, Any]:

    print("========== EXCEL DEBUG ==========")
    print("Reading file:", PAYMENTS_FILE)
    print("File exists:", os.path.exists(PAYMENTS_FILE))

    result = load_excel(PAYMENTS_FILE, "Payments")

    if result["status"] == "success":
        print("SUCCESS: XLSX Loaded")
        print(result["data"].head())
        print(result["data"].columns.tolist())
    else:
        print("ERROR LOADING XLSX:", result["message"])

    return result


def load_students():
    return load_excel(STUDENTS_FILE, "Students")

def load_fees():
    return load_excel(FEES_FILE, "FeeStructure")

def load_parents():
    return load_excel(PARENTS_FILE, "Parents")

def load_siblings():
    return load_excel(SIBLINGS_FILE, "Siblings")

def load_communication():
    return load_excel(COMMUNICATION_FILE, "Communication")
