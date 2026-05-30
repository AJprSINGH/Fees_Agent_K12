import requests
import os

# =========================
# CONFIG
# =========================
BASE_URL = os.getenv("ERP_BASE_URL", "https://erp.triz.co.in")
TOKEN    = os.getenv("ERP_TOKEN", "YOUR_TOKEN")

SYEAR            = os.getenv("ERP_SYEAR", "2026")
SUB_INSTITUTE_ID = os.getenv("ERP_SUB_ID", "1")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# BUG FIX: guard import so server starts even without Supabase
supabase = None
try:
    if SUPABASE_URL and SUPABASE_KEY:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"fee_api_service: Supabase init skipped — {e}")


# =========================
# HEADERS
# =========================
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type":  "application/json",
}


# =========================
# GET FEE DETAILS
# =========================
def get_fee_details(student_id: str):
    try:
        url = (
            f"{BASE_URL}/api/fee-details"
            f"?student_id={student_id}"
            f"&syear={SYEAR}"
        )
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


# =========================
# GET PENDING FEES
# =========================
def get_pending_fees(student_id: str):
    try:
        url = (
            f"{BASE_URL}/api/pending-fees"
            f"?student_id={student_id}"
            f"&syear={SYEAR}"
        )
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}
