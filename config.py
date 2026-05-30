import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # =========================================================
    # AUTHENTICATION — Phase 6 Admin APIs
    # =========================================================
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "")

    # =========================================================
    # ERP CONFIG
    # =========================================================
    ERP_BASE_URL: str = os.getenv(
        "ERP_BASE_URL",
        "https://erp.triz.co.in"
    )

    ERP_TOKEN: str = os.getenv(
        "ERP_TOKEN",
        "your-token"
    )

    # =========================================================
    # EXCEL FILE PATH
    # =========================================================
    EXCEL_PATH: str = os.getenv(
        "DEFAULTER_EXCEL_PATH",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "Fees_Defaulter_Report_-_Overall.xlsx"
        )
    )

    # =========================================================
    # REPORT OUTPUT DIRECTORY
    # =========================================================
    REPORT_DIR: str = os.getenv(
        "REPORT_DIR",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "reports"
        )
    )

    # =========================================================
    # RISK THRESHOLDS - DAYS
    # =========================================================
    THRESHOLD_YELLOW_DAYS: int = int(
        os.getenv("THRESHOLD_YELLOW_DAYS", 1)
    )

    THRESHOLD_RED_DAYS: int = int(
        os.getenv("THRESHOLD_RED_DAYS", 365)
    )

    THRESHOLD_CRITICAL_DAYS: int = int(
        os.getenv("THRESHOLD_CRITICAL_DAYS", 730)
    )

    # =========================================================
    # RISK THRESHOLDS - AMOUNT
    # =========================================================
    THRESHOLD_YELLOW_AMOUNT: float = float(
        os.getenv("THRESHOLD_YELLOW_AMOUNT", 1)
    )

    THRESHOLD_RED_AMOUNT: float = float(
        os.getenv("THRESHOLD_RED_AMOUNT", 50000)
    )

    THRESHOLD_CRITICAL_AMOUNT: float = float(
        os.getenv("THRESHOLD_CRITICAL_AMOUNT", 100000)
    )

    # =========================================================
    # Pydantic Settings Config
    # =========================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()