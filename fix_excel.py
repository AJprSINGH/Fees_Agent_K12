# =============================================================
# fix_excel.py — One-time XLSX repair script
# Run ONCE before deploying to fix any corrupt/empty xlsx files.
# Usage: python fix_excel.py
# =============================================================

import pandas as pd

files = [
    "data/communication.xlsx",
    "data/siblings.xlsx",
    "reports/daily_defaulter_report.xlsx",
    "data/students.xlsx",
    "data/payments.xlsx",
    "data/parents.xlsx",
    "data/fee_structure.xlsx",
]

for file in files:
    try:
        df = pd.read_excel(file, engine="openpyxl")
        df.to_excel(file, index=False, engine="openpyxl")
        print(f"✅ Fixed: {file}  ({len(df)} rows)")
    except FileNotFoundError:
        print(f"⚠️  Skipped (not found): {file}")
    except Exception as e:
        print(f"❌ Error: {file} — {e}")

print("\nDone. All fixable files have been re-saved.")
