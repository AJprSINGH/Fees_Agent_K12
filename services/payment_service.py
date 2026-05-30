import pandas as pd
import numpy as np

from datetime import datetime

from utils.excel_loader import load_payments


# =====================================================
# ALL HISTORICAL YEAR COLUMNS (2017-18 → 2024-25)
# =====================================================
ALL_YEAR_COLS = [
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
]

YEAR_COLS = [
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
]

YEAR_DAYS_MAP = {
    "2021-2022": 1460,
    "2022-2023": 1095,
    "2023-2024": 730,
    "2024-2025": 365,
}


def _oldest_unpaid_year(row):
    try:
        for year in YEAR_COLS:
            if year not in row.index:
                continue
            value = pd.to_numeric(row.get(year, 0), errors="coerce")
            if pd.notna(value) and value > 0:
                return year
        return None
    except Exception:
        return None


def _derive_days_overdue(row):
    try:
        total_unpaid   = float(row.get("total_unpaid",   0) or 0)
        pending_amount = float(row.get("pending_amount", 0) or 0)
        effective_unpaid = max(total_unpaid, pending_amount)
        if effective_unpaid <= 0:
            return 0
        oldest_year = row.get("oldest_unpaid_year")
        if oldest_year and oldest_year in YEAR_DAYS_MAP:
            return YEAR_DAYS_MAP[oldest_year]
        return 30
    except Exception:
        return 0


def _assign_risk_tier(days_overdue: int, total_unpaid: float) -> str:
    try:
        if total_unpaid <= 0:
            return "GREEN"
        if days_overdue > 730 or total_unpaid > 100000:
            return "CRITICAL"
        if days_overdue > 365 or total_unpaid > 50000:
            return "RED"
        return "YELLOW"
    except Exception:
        return "GREEN"


# =====================================================
# PHASE 2 — HISTORICAL PAYMENT BEHAVIOUR FEATURES
# =====================================================
def _build_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    present_years = [y for y in ALL_YEAR_COLS if y in df.columns]

    if not present_years:
        for col in ["years_enrolled", "years_with_payment", "years_without_payment",
                    "consecutive_default_years", "payment_streak_length", "first_default_year"]:
            df[col] = 0
        return df

    for y in present_years:
        df[y] = pd.to_numeric(df[y], errors="coerce").fillna(0)

    # per-year paid flags
    for y in present_years:
        flag_col = "paid_" + y.replace("-", "_")
        df[flag_col] = (df[y] == 0).astype(int)

    def _row_history(row):
        unpaid_flags = [1 if float(row.get(y, 0) or 0) > 0 else 0 for y in present_years]
        enrolled   = len(unpaid_flags)
        paid_yrs   = unpaid_flags.count(0)
        unpaid_yrs = unpaid_flags.count(1)

        max_consec = cur = 0
        for f in unpaid_flags:
            cur = cur + 1 if f == 1 else 0
            max_consec = max(max_consec, cur)

        max_streak = cur = 0
        for f in unpaid_flags:
            cur = cur + 1 if f == 0 else 0
            max_streak = max(max_streak, cur)

        first_default = next((i + 1 for i, f in enumerate(unpaid_flags) if f == 1), 0)

        return pd.Series({
            "years_enrolled":            enrolled,
            "years_with_payment":        paid_yrs,
            "years_without_payment":     unpaid_yrs,
            "consecutive_default_years": max_consec,
            "payment_streak_length":     max_streak,
            "first_default_year":        first_default,
        })

    hist = df.apply(_row_history, axis=1)
    df = pd.concat([df, hist], axis=1)
    return df


# =====================================================
# PHASE 2 — TREND-BASED FINANCIAL FEATURES
# =====================================================
def _build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    present_years = [y for y in ALL_YEAR_COLS if y in df.columns]

    # resolve total_fee
    total_fee_col = next((c for c in ["reg+bus", "regular", "total"] if c in df.columns), None)
    if total_fee_col:
        df["total_fee"] = pd.to_numeric(df[total_fee_col], errors="coerce").fillna(0)
    else:
        df["total_fee"] = (
            pd.to_numeric(df.get("total_paid",    pd.Series([0]*len(df))), errors="coerce").fillna(0)
            + pd.to_numeric(df.get("total_unpaid", pd.Series([0]*len(df))), errors="coerce").fillna(0)
        )

    df["partial_payment_ratio"] = np.where(
        df["total_fee"] > 0, df["total_paid"] / df["total_fee"], 0.0
    ).clip(0, 1)

    df["discount_pct"] = np.where(
        df["total_fee"] > 0, df["discount"] / df["total_fee"], 0.0
    ).clip(0, 1)

    def _unpaid_slope(row):
        vals = [float(row.get(y, 0) or 0) for y in present_years]
        if len(vals) < 2:
            return 0.0
        try:
            return float(np.polyfit(np.arange(len(vals), dtype=float), vals, 1)[0])
        except Exception:
            return 0.0

    df["cumulative_unpaid_trend"] = df.apply(_unpaid_slope, axis=1)

    def _momentum(row):
        vals = [float(row.get(y, 0) or 0) for y in present_years]
        if len(vals) < 4:
            return "stable"
        early  = np.mean(vals[:2])
        recent = np.mean(vals[-2:])
        if recent < early - 1:
            return "improving"
        if recent > early + 1:
            return "worsening"
        return "stable"

    df["payment_momentum"] = df.apply(_momentum, axis=1)

    # bus fee flag
    bus_col = next((c for c in ["bus", "bus fee", "bus_fee"] if c in df.columns), None)
    if bus_col:
        df["has_bus_fee"] = (pd.to_numeric(df[bus_col], errors="coerce").fillna(0) > 0).astype(int)
    else:
        df["has_bus_fee"] = 0

    # fee tier (quartile-based)
    try:
        q1 = df["total_fee"].quantile(0.33)
        q2 = df["total_fee"].quantile(0.66)
        df["fee_tier"] = df["total_fee"].apply(
            lambda v: "Low" if v <= q1 else ("Mid" if v <= q2 else "High")
        )
    except Exception:
        df["fee_tier"] = "Mid"

    return df


# =====================================================
# LOAD PAYMENT DATA — public API (unchanged signature)
# =====================================================
def get_payments() -> pd.DataFrame:

    try:
        response = load_payments()

        if response["status"] != "success":
            print(f"PAYMENT LOAD ERROR: {response['message']}")
            return pd.DataFrame()

        df = response["data"]

        if df is None or df.empty:
            print("PAYMENT DATAFRAME EMPTY")
            return pd.DataFrame()

        print("========== ORIGINAL COLUMNS ==========")
        print(df.columns.tolist())
        print("======================================")

        df.rename(
            columns={
                "gr no.":       "student_id",
                "gr no":        "student_id",
                "student name": "student_name",
                "std/div":      "class_name",
                "std div":      "class_name",
                "mobile no.":   "mobile_number",
                "mobile no":    "mobile_number",
                "total unpaid": "pending_amount",
                "total paid":   "total_paid",
                "discount":     "discount",
            },
            inplace=True,
        )

        print("========== RENAMED COLUMNS ==========")
        print(df.columns.tolist())
        print("=====================================")

        for col in ["student_id", "pending_amount"]:
            if col not in df.columns:
                df[col] = "" if col == "student_id" else 0

        for col, default in {
            "student_name": "", "class_name": "", "mobile_number": "",
            "status": "", "total_paid": 0, "discount": 0,
        }.items():
            if col not in df.columns:
                df[col] = default

        df["student_id"] = (
            df["student_id"].astype(str)
            .str.replace(".0", "", regex=False)
            .str.strip()
        )

        invalid_values = ["", "nan", "none", "null"]
        df = df[~df["student_id"].str.lower().isin(invalid_values)]

        df["pending_amount"] = pd.to_numeric(df["pending_amount"], errors="coerce").fillna(0)

        if "total_unpaid" not in df.columns:
            df["total_unpaid"] = df["pending_amount"]
        df["total_unpaid"] = pd.to_numeric(df["total_unpaid"], errors="coerce").fillna(0)
        df["total_unpaid"] = df[["total_unpaid", "pending_amount"]].max(axis=1)

        for col in ["total_paid", "discount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df["status"] = df["status"].astype(str).str.strip().str.upper()
        df = df.drop_duplicates(subset=["student_id"])
        df["due_date"] = datetime.today().strftime("%Y-%m-%d")

        # ============================================
        # PHASE 2 — historical + trend feature blocks
        # ============================================
        df = _build_historical_features(df)
        df = _build_trend_features(df)

        df = df.reset_index(drop=True)

        print("========== FINAL COLUMNS ==========")
        print(df.columns.tolist())
        print("========== IDS ==========")
        print(df["student_id"].unique().tolist()[:20])
        print("========== SAMPLE ==========")
        print(df.head(3).to_dict(orient="records"))
        print("===================================")

        return df

    except Exception as error:
        print(f"Error in get_payments: {str(error)}")
        return pd.DataFrame()


# =====================================================
# CALCULATE DUE STATUS — unchanged public signature
# =====================================================
def calculate_due_status(df: pd.DataFrame) -> pd.DataFrame:

    try:
        df = df.copy()

        if df.empty:
            df["oldest_unpaid_year"] = None
            df["days_overdue"]       = 0
            df["risk_tier"]          = "GREEN"
            return df

        for col in ["pending_amount", "total_unpaid"]:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df["total_unpaid"] = df[["total_unpaid", "pending_amount"]].max(axis=1)
        df["oldest_unpaid_year"] = df.apply(_oldest_unpaid_year, axis=1)
        df["days_overdue"] = df.apply(_derive_days_overdue, axis=1)
        df["days_overdue"] = df["days_overdue"].fillna(0).astype(int)

        df["risk_tier"] = df.apply(
            lambda row: _assign_risk_tier(
                int(row.get("days_overdue", 0)),
                float(row.get("total_unpaid", 0) or 0),
            ),
            axis=1,
        )
        df["risk_tier"] = df["risk_tier"].fillna("GREEN").astype(str).str.upper()

        print("========== DUE STATUS ==========")
        cols_to_show = [
            c for c in
            ["student_id", "student_name", "pending_amount",
             "total_unpaid", "days_overdue", "risk_tier"]
            if c in df.columns
        ]
        print(df[cols_to_show].head(10).to_dict(orient="records"))
        print("================================")

        return df

    except Exception as error:
        print(f"Error in calculate_due_status: {str(error)}")
        return df
