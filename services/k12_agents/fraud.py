from database.supabase_client import supabase


def run(data):
    try:
        # =========================
        # FLEXIBLE INPUT SUPPORT
        # =========================
        transactions = []
        agent_id = None

        # Support dict / object / list
        if isinstance(data, list):
            transactions = data
        elif isinstance(data, dict):
            transactions = data.get("transactions", [])
            agent_id = data.get("agent_id")
        else:
            transactions = getattr(data, "transactions", [])
            agent_id = getattr(data, "agent_id", None)

        # Backward compatibility (single object)
        if not transactions and hasattr(data, "amount"):
            transactions = [{
                "amount": getattr(data, "amount", 0),
                "status": getattr(data, "status", "unknown"),
                "student_id": getattr(data, "student_id", None)
            }]
            agent_id = getattr(data, "agent_id", None)

        # =========================
        # VALIDATION
        # =========================
        if not transactions or len(transactions) == 0:
            return {
                "status": "error",
                "message": "Transaction data is required"
            }

        fraud_results = []
        suspicious = []

        # =========================
        # FRAUD LOGIC (FINAL)
        # =========================
        for tx in transactions:
            try:
                amount = tx.get("amount", 0)
                status = tx.get("status", "unknown")
                student_id = tx.get("student_id")

                fraud = False
                reasons = []

                # Rule 1
                if amount > 100000:
                    fraud = True
                    reasons.append("High transaction")

                # Rule 2
                if status == "failed":
                    fraud = True
                    reasons.append("Failed transaction")

                # Rule 3 (suspicious only)
                if amount > 50000:
                    suspicious.append(tx)

                result_row = {
                    "agent_id": agent_id,
                    "student_id": student_id,
                    "amount": amount,
                    "status": status,
                    "fraud": fraud,
                    "reasons": reasons
                }

                fraud_results.append(result_row)

                # =========================
                # DB INSERT (FIXED TABLE)
                # =========================
                if agent_id:
                    supabase.table("payments").insert({
                        "agent_id": agent_id,
                        "student_id": student_id,
                        "amount": amount,
                        "status": status
                    }).execute()

            except Exception as row_error:
                fraud_results.append({
                    "error": str(row_error),
                    "tx": tx
                })

        # =========================
        # FINAL RESPONSE
        # =========================
        return {
            "status": "success",
            "payload_used": {
                "agent_id": agent_id,
                "total_transactions": len(transactions)
            },
            "data": {
                "total": len(transactions),
                "fraud_count": len([f for f in fraud_results if f.get("fraud")]),
                "fraud_details": fraud_results,
                "suspicious_transactions": suspicious
            },
            "insights": [
                "High value and failed transactions flagged",
                "Suspicious transactions identified above threshold"
            ],
            "recommended_actions": [
                "Review high-value transactions manually",
                "Retry failed payments",
                "Enable alerts for abnormal activity"
            ],
            "confidence_score": 0.9
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }