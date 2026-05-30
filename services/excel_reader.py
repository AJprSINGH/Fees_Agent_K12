# =============================================================
# services/excel_reader.py
# Phase 6 — XLSX student data reader
#
# FIXES APPLIED:
#   ✅ FIX 1:  Supports both original filename (spaces) and renamed version
#   ✅ FIX 2:  Falls back through multiple known paths automatically
#   ✅ FIX 3:  Strips dots, slashes, brackets from column names
#   ✅ FIX 4:  Explicit rename map → gr_number, std_div, pending_fees
#   ✅ FIX 5:  drop_duplicates on gr_number — prevents ON CONFLICT batch errors
#   ✅ FIX 6:  Auto-create missing required columns
#   ✅ FIX 7:  Safe numeric conversion
#   ✅ FIX 8:  Added ML feature compatibility columns
#   ✅ FIX 9:  Supports Hugging Face deployment paths
#   ✅ FIX 10: Better logging and debugging support
#   ✅ FIX 11: days_overdue and total_unpaid always present as numeric
#   ✅ FIX 12: student_id column derived from gr_number for ML pipeline
# =============================================================

import os
import re
import pandas as pd

# =============================================================
# POSSIBLE XLSX FILE PATHS
# =============================================================

_CANDIDATE_PATHS = [
    "data/Fees Defaulter Report - Overall.xlsx",
    "data/fees_defaulter_report.xlsx",
    "Fees Defaulter Report - Overall.xlsx",
]

# =============================================================
# REQUIRED COLUMN STANDARDIZATION
# =============================================================

REQUIRED_COLUMNS = {
    "gr_no": "gr_number",
    "gr_no_": "gr_number",
    "gr_number": "gr_number",
    "student_name": "student_name",
    "student": "student_name",
    "name": "student_name",
    "std_div": "std_div",
    "std_division": "std_div",
    "class_div": "std_div",
    "total_fees": "total_fees",
    "fees": "total_fees",
    "total_paid": "total_paid",
    "paid_amount": "total_paid",
    "pending_amount": "pending_amount",
    "pending_fees": "pending_amount",
    "total_unpaid": "pending_amount",
    "days_overdue": "days_overdue",
    "quota": "quota",
    "roll_no": "roll_no",
    "mobile_no": "mobile_no",
    "sr": "sr_no",
}

# =============================================================
# NORMALIZE COLUMN NAME
# =============================================================

def _normalize_column(col: str) -> str:
    """
    Normalize XLSX column names safely.
    """

    col = str(col).strip().lower()

    # Replace spaces and special chars with _
    col = re.sub(r"[\s./\\()\-+]+", "_", col)

    # Remove duplicate underscores
    col = re.sub(r"_+", "_", col)

    # Remove starting/ending _
    col = col.strip("_")

    return col


# =============================================================
# MAIN XLSX READER
# =============================================================

def read_student_data(file_path: str = None) -> pd.DataFrame:
    """
    Read and normalize student XLSX data safely.
    Supports:
        - Hugging Face deployment
        - Dynamic paths
        - Dirty Excel headers
        - Missing columns
        - Duplicate GR numbers
    """

    # =========================================================
    # BUILD FILE SEARCH PATHS
    # =========================================================

    candidates = []

    if file_path:
        candidates.append(file_path)

    env_path = os.getenv("TRAINING_DATA_PATH", "")

    if env_path:
        candidates.append(env_path)

    candidates.extend(_CANDIDATE_PATHS)

    # Remove duplicate paths
    candidates = list(dict.fromkeys(candidates))

    # =========================================================
    # FIND VALID XLSX FILE
    # =========================================================

    resolved = None

    for path in candidates:
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                resolved = path
                break
        except Exception:
            continue

    # =========================================================
    # FILE NOT FOUND
    # =========================================================

    if resolved is None:
        raise FileNotFoundError(
            "Student data XLSX file not found.\n\n"
            "Tried paths:\n"
            + "\n".join(f"  • {p}" for p in candidates)
            + "\n\n"
            "Fix:\n"
            "1. Place XLSX file inside data/ folder\n"
            "2. OR set TRAINING_DATA_PATH in .env\n"
            "3. OR upload XLSX properly in Hugging Face"
        )

    print(f"[excel_reader] Reading XLSX file: {resolved}")

    # =========================================================
    # READ XLSX
    # =========================================================

    try:
        df = pd.read_excel(
            resolved,
            engine="openpyxl",
        )

    except Exception as e:
        raise RuntimeError(
            f"Failed to read XLSX file: {resolved}\nError: {str(e)}"
        )

    # =========================================================
    # NORMALIZE COLUMN NAMES
    # =========================================================

    original_columns = list(df.columns)

    df.columns = [_normalize_column(col) for col in df.columns]

    # =========================================================
    # RENAME COLUMNS
    # =========================================================

    rename_map = {}

    for col in df.columns:
        if col in REQUIRED_COLUMNS:
            # Skip renaming total_unpaid -> pending_amount to avoid conflict
            if col == "total_unpaid" and "pending_amount" in df.columns:
                continue
            rename_map[col] = REQUIRED_COLUMNS[col]

    df.rename(columns=rename_map, inplace=True)

    print("[excel_reader] Original columns:")
    print(original_columns)

    print("[excel_reader] Normalized columns:")
    print(list(df.columns))

    # =========================================================
    # CREATE MISSING REQUIRED COLUMNS
    # =========================================================

    required_final_columns = [
        "gr_number",
        "student_name",
        "std_div",
        "total_fees",
        "total_paid",
        "pending_amount",
        "days_overdue",
        "quota",
    ]

    for col in required_final_columns:

        if col not in df.columns:

            if col in [
                "gr_number",
                "student_name",
                "std_div",
                "quota",
            ]:
                df[col] = ""

            else:
                df[col] = 0

            print(f"[excel_reader] Created missing column: {col}")

    # =========================================================
    # SAFE TYPE CONVERSION
    # =========================================================

    # String columns
    string_cols = [
        "gr_number",
        "student_name",
        "std_div",
        "quota",
    ]

    for col in string_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Numeric columns
    numeric_cols = [
        "total_fees",
        "total_paid",
        "total_unpaid",
        "days_overdue",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0)
        
    # Ensure pending_amount also exists as an alias for total_unpaid
    if "total_unpaid" in df.columns and "pending_amount" not in df.columns:
        df["pending_amount"] = df["total_unpaid"]

    # =========================================================
    # REMOVE DUPLICATE GR NUMBERS
    # =========================================================

    if "gr_number" in df.columns:

        before = len(df)

        df = df.drop_duplicates(
            subset=["gr_number"],
            keep="first",
        )

        removed = before - len(df)

        if removed > 0:
            print(
                f"[excel_reader] Removed "
                f"{removed} duplicate gr_number rows."
            )

    # =========================================================
    # ADD / ENSURE ML FEATURE COLUMNS
    # FIX 11: days_overdue and total_unpaid MUST be numeric
    #         columns for the ML pipeline — ensure they exist.
    # =========================================================

    # total_unpaid — the canonical ML column name
    # Always derived from pending_amount for consistency
    df["total_unpaid"] = pd.to_numeric(
        df.get("pending_amount", 0),
        errors="coerce",
    ).fillna(0)

    # days_overdue — ensure safe int after potential coerce
    df["days_overdue"] = pd.to_numeric(
        df["days_overdue"],
        errors="coerce",
    ).fillna(0).astype(int)

    # FIX 12: student_id — ML pipeline expects this column
    if "student_id" not in df.columns:
        df["student_id"] = df["gr_number"]

    # ML pipeline features with safe defaults
    if "years_enrolled" not in df.columns:
        df["years_enrolled"] = 1

    if "payment_streak_length" not in df.columns:
        df["payment_streak_length"] = 0

    if "consecutive_default_years" not in df.columns:
        df["consecutive_default_years"] = 0

    if "fee_tier_encoded" not in df.columns:
        df["fee_tier_encoded"] = 0

    if "payment_momentum" not in df.columns:
        df["payment_momentum"] = 0

    # =========================================================
    # FINAL LOGGING
    # =========================================================

    print(
        f"[excel_reader] Successfully loaded "
        f"{len(df)} student records."
    )

    print(
        f"[excel_reader] Final columns: "
        f"{list(df.columns)}"
    )

    # =========================================================
    # RETURN CLEAN DATAFRAME
    # =========================================================

    return df
