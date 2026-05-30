from pathlib import Path
from datetime import datetime

import pandas as pd

# ==========================================
# BASE DIRECTORY
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================
# REPORT DIRECTORY
# ==========================================
REPORT_DIR = BASE_DIR / "reports"

# Create reports folder if not exists
REPORT_DIR.mkdir(
    exist_ok=True
)


# ==========================================
# GENERATE DAILY REPORT
# ==========================================
def generate_daily_report(df):

    try:

        # ==========================================
        # DATAFRAME VALIDATION
        # ==========================================
        if df is None:
            raise ValueError(
                "Dataframe is required"
            )

        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                "Invalid dataframe format"
            )

        if df.empty:
            raise ValueError(
                "Dataframe is empty"
            )

        # ==========================================
        # REPORT FILE NAME
        # ==========================================
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        report_file = (
            REPORT_DIR /
            f"daily_defaulter_report_{timestamp}.xlsx"
        )

        # ==========================================
        # EXPORT REPORT
        # ==========================================
        df.to_excel(
            report_file,
            index=False,
            engine="xlsxwriter"
        )

        # ==========================================
        # SUCCESS RESPONSE
        # ==========================================
        return {
            "status": "success",
            "message": "Daily report generated successfully",
            "report_name": report_file.name,
            "report_path": str(report_file),
            "total_records": len(df)
        }

    except Exception as error:

        raise Exception(
            f"Error while generating report: {str(error)}"
        )