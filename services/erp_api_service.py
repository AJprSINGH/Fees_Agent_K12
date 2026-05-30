import requests

BASE_URL = "https://erp.triz.co.in"

def student_lookup(mobile):
    return requests.get(
        f"{BASE_URL}/fees/get-student",
        params={"mobile_number": mobile}
    ).json()

def fee_details(student_id, sub_institute_id, syear, token):
    return requests.post(
        f"{BASE_URL}/studentFeesDetailAPI",
        json={
            "student_id": student_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "token": token
        }
    ).json()

def pending_fees(sub_institute_id, syear):
    return requests.get(
        f"{BASE_URL}/pending_fees",
        params={
            "type": "API",
            "sub_institute_id": sub_institute_id,
            "syear": syear
        }
    ).json()

def total_pending(student_id):
    return requests.get(
        f"{BASE_URL}/ajax_checkFeesBreakoff",
        params={"student_id": student_id}
    ).json()

def fee_receipt(student_id, receipt_id):
    return requests.get(
        f"{BASE_URL}/ajax_PDF_FeesReceipt",
        params={
            "action": "fees_collect_receipt",
            "type": "API",
            "student_id": student_id,
            "receipt_id_html": receipt_id
        }
    ).content