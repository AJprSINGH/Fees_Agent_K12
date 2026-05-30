# =============================================================
# versioning/model_registry.py
# Phase 6 — Model Versioning
#
# FIXES APPLIED:
#   ✅ Added demote_to_staging()
#   ✅ Added rollback_to_previous()
#   ✅ Added auto_rollback_on_drift()
#   ✅ Added safer offline fallback logging
#   ✅ Added rollback-safe deployment lifecycle
#   ✅ Preserved existing APIs and compatibility
#   ✅ No breaking changes to existing payloads/routes
#
# Tracks every trained model in Supabase table `model_versions`.
# Supports:
#   - registering a new model version after training
#   - promoting a version to Active
#   - demoting bad versions safely
#   - rolling back to previous stable versions
#   - querying active versions
#   - listing version history
#   - automatic rollback on severe drift
#
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import json
import traceback
import uuid

from datetime import datetime, timezone
from typing import Dict, List, Optional


# =============================================================
# VERSION ID GENERATOR
# =============================================================

def _generate_version_id(model_name: str, band: str = "global") -> str:
    """
    Generate unique model version id.

    Example:
        xgboost_secondary_20260513_1530_ab12cd
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    short = str(uuid.uuid4())[:6]

    slug = model_name.lower().replace(" ", "_")[:12]
    band_slug = band.lower()[:12]

    return f"{slug}_{band_slug}_{ts}_{short}"


# =============================================================
# REGISTER A NEW MODEL VERSION
# =============================================================

def register_model_version(
    supabase_client,
    model_name: str,
    band: str,
    f1_score: float,
    pr_auc: float,
    model_path: str,
    pipeline_path: str,
    dataset_range: str = "",
    precision_score: float = 0.0,
    recall_score: float = 0.0,
    holdout_metrics: dict = None,
    notes: str = "",
    trained_by: str = "scheduled_pipeline",
    deployment_status: str = "Staging",
) -> Optional[str]:
    """
    Register a newly trained model version.

    Returns
    -------
    version_id : str | None
    """

    version_id = _generate_version_id(model_name, band)
    trained_on = datetime.now(timezone.utc).isoformat()

    record = {
        "version_id": version_id,
        "model_name": model_name,
        "band": band,
        "trained_on": trained_on,
        "dataset_range": dataset_range,
        "f1_score": round(float(f1_score), 4),
        "pr_auc": round(float(pr_auc), 4),
        "precision_score": round(float(precision_score), 4),
        "recall_score": round(float(recall_score), 4),
        "deployment_status": deployment_status,
        "model_path": model_path,
        "pipeline_path": pipeline_path,
        "trained_by": trained_by,
        "notes": notes,
        "holdout_metrics": json.dumps(
            holdout_metrics or {},
            default=str
        ),
    }

    print(
        f"[MODEL REGISTRY] Registering {version_id} "
        f"(band={band}, model={model_name}, "
        f"F1={f1_score:.4f}, PR-AUC={pr_auc:.4f})"
    )

    # =========================================================
    # OFFLINE MODE
    # =========================================================

    if supabase_client is None:
        print("[MODEL REGISTRY] Supabase unavailable — logging offline.")

        _log_offline({
            "_op": "REGISTER",
            **record
        })

        return version_id

    # =========================================================
    # INSERT INTO SUPABASE
    # =========================================================

    try:
        supabase_client.table("model_versions") \
            .upsert(record, on_conflict="version_id") \
            .execute()

        print(f"[MODEL REGISTRY] Registered → {version_id}")

        return version_id

    except Exception as e:
        print(f"[MODEL REGISTRY] Supabase insert failed: {e}")

        traceback.print_exc()

        _log_offline({
            "_op": "REGISTER_FAILED",
            **record,
            "_error": str(e),
        })

        # Local PKL remains source of truth
        return version_id


# =============================================================
# PROMOTE MODEL TO ACTIVE
# =============================================================

def promote_to_active(
    supabase_client,
    version_id: str,
    band: str,
) -> bool:
    """
    Promote model version to Active.

    Automatically archives previous Active versions
    for the same band.
    """

    if supabase_client is None:
        print(f"[MODEL REGISTRY] Offline — cannot promote {version_id}")

        _log_offline({
            "_op": "PROMOTE",
            "version_id": version_id,
            "band": band,
            "_ts": datetime.now(timezone.utc).isoformat(),
        })

        return False

    try:

        # =====================================================
        # ARCHIVE CURRENT ACTIVE
        # =====================================================

        supabase_client.table("model_versions") \
            .update({
                "deployment_status": "Archived"
            }) \
            .eq("band", band) \
            .eq("deployment_status", "Active") \
            .execute()

        # =====================================================
        # PROMOTE TARGET VERSION
        # =====================================================

        supabase_client.table("model_versions") \
            .update({
                "deployment_status": "Active"
            }) \
            .eq("version_id", version_id) \
            .execute()

        print(
            f"[MODEL REGISTRY] Promoted "
            f"{version_id} → Active (band={band})"
        )

        return True

    except Exception as e:
        print(f"[MODEL REGISTRY] Promote failed: {e}")

        traceback.print_exc()

        return False


# =============================================================
# DEMOTE TO STAGING
# =============================================================

def demote_to_staging(
    supabase_client,
    version_id: str,
    band: str,
    reason: str = "",
) -> bool:
    """
    Demote Active model back to Staging.

    Does NOT promote another version automatically.

    Useful when:
        - model drift detected
        - deployment issue
        - temporary disable needed
    """

    if supabase_client is None:

        print(
            f"[MODEL REGISTRY] Offline — logging demotion "
            f"for {version_id}"
        )

        _log_offline({
            "_op": "DEMOTE",
            "version_id": version_id,
            "band": band,
            "reason": reason,
            "_ts": datetime.now(timezone.utc).isoformat(),
        })

        return False

    try:

        supabase_client.table("model_versions") \
            .update({
                "deployment_status": "Staging",
                "notes": f"[DEMOTED] {reason}",
            }) \
            .eq("version_id", version_id) \
            .execute()

        print(
            f"[MODEL REGISTRY] Demoted "
            f"{version_id} → Staging "
            f"(band={band})"
        )

        return True

    except Exception as e:

        print(f"[MODEL REGISTRY] Demote failed: {e}")

        traceback.print_exc()

        return False


# =============================================================
# ROLLBACK TO PREVIOUS VERSION
# =============================================================

def rollback_to_previous(
    supabase_client,
    band: str,
    reason: str = "drift detected",
) -> Dict:
    """
    Roll back current Active model to previous Archived version.

    Workflow
    --------
    1. Find current Active model
    2. Demote Active → Staging
    3. Find most recent Archived version
    4. Promote Archived → Active
    """

    result = {
        "status": "error",
        "band": band,
        "reason": reason,
        "demoted": None,
        "promoted": None,
    }

    if supabase_client is None:

        print(
            f"[MODEL REGISTRY] Offline — rollback skipped "
            f"(band={band})"
        )

        _log_offline({
            "_op": "ROLLBACK_ATTEMPTED",
            "band": band,
            "reason": reason,
            "_ts": datetime.now(timezone.utc).isoformat(),
        })

        result["message"] = "Supabase unavailable"

        return result

    try:

        # =====================================================
        # FIND CURRENT ACTIVE VERSION
        # =====================================================

        current_resp = (
            supabase_client.table("model_versions")
            .select("*")
            .eq("band", band)
            .eq("deployment_status", "Active")
            .order("trained_on", desc=True)
            .limit(1)
            .execute()
        )

        current_rows = current_resp.data or []

        current_vid = (
            current_rows[0]["version_id"]
            if current_rows else None
        )

        # =====================================================
        # DEMOTE CURRENT ACTIVE
        # =====================================================

        if current_vid:

            demote_to_staging(
                supabase_client=supabase_client,
                version_id=current_vid,
                band=band,
                reason=f"Rollback initiated — {reason}"
            )

            result["demoted"] = current_vid

        else:
            print(
                f"[MODEL REGISTRY] No Active version "
                f"found for band={band}"
            )

        # =====================================================
        # FIND PREVIOUS ARCHIVED VERSION
        # =====================================================

        prev_resp = (
            supabase_client.table("model_versions")
            .select("*")
            .eq("band", band)
            .eq("deployment_status", "Archived")
            .order("trained_on", desc=True)
            .limit(1)
            .execute()
        )

        prev_rows = prev_resp.data or []

        if not prev_rows:

            print(
                f"[MODEL REGISTRY] No Archived version "
                f"available for rollback"
            )

            result["status"] = "no_previous_version"

            result["message"] = (
                "No Archived version exists."
            )

            return result

        previous_vid = prev_rows[0]["version_id"]

        # =====================================================
        # PROMOTE PREVIOUS VERSION
        # =====================================================

        promote_to_active(
            supabase_client=supabase_client,
            version_id=previous_vid,
            band=band,
        )

        result["promoted"] = previous_vid

        # =====================================================
        # ADD AUDIT NOTE
        # =====================================================

        supabase_client.table("model_versions") \
            .update({
                "notes": f"[ROLLBACK] Promoted from Archived — {reason}"
            }) \
            .eq("version_id", previous_vid) \
            .execute()

        print(
            f"[MODEL REGISTRY] Rollback completed "
            f"(band={band}) "
            f"demoted={current_vid} "
            f"promoted={previous_vid}"
        )

        result["status"] = "success"

        return result

    except Exception as e:

        traceback.print_exc()

        result["message"] = str(e)

        return result


# =============================================================
# GET ACTIVE VERSION
# =============================================================

def get_active_version(
    supabase_client,
    band: str = "global",
) -> Optional[Dict]:
    """
    Return currently Active model record.
    """

    if supabase_client is None:
        return None

    try:

        response = (
            supabase_client.table("model_versions")
            .select("*")
            .eq("band", band)
            .eq("deployment_status", "Active")
            .order("trained_on", desc=True)
            .limit(1)
            .execute()
        )

        rows = response.data or []

        return rows[0] if rows else None

    except Exception as e:

        print(f"[MODEL REGISTRY] get_active_version failed: {e}")

        return None


# =============================================================
# LIST VERSION HISTORY
# =============================================================

def list_versions(
    supabase_client,
    band: Optional[str] = None,
    limit: int = 20,
) -> List[Dict]:
    """
    Return model version history.
    """

    if supabase_client is None:
        return []

    try:

        query = supabase_client.table("model_versions").select("*")

        if band:
            query = query.eq("band", band)

        response = (
            query
            .order("trained_on", desc=True)
            .limit(limit)
            .execute()
        )

        return response.data or []

    except Exception as e:

        print(f"[MODEL REGISTRY] list_versions failed: {e}")

        return []


# =============================================================
# REGISTER TRAINING RUN
# =============================================================

def register_training_run(
    supabase_client,
    training_output: Dict,
    dataset_range: str = "",
    notes: str = "",
    auto_promote: bool = False,
) -> Dict:
    """
    Register all trained band models from train_and_save().
    """

    registered = {}
    promoted = {}

    bands_output = training_output.get("bands", {})

    for band, result in bands_output.items():

        if result.get("status") != "success":

            print(
                f"[MODEL REGISTRY] Skipping band={band} "
                f"(status={result.get('status')})"
            )

            continue

        holdout = result.get("holdout_metrics", {})

        f1 = holdout.get(
            "f1_score",
            result.get("cv_f1_scores", {}).get(
                result.get("best_model", ""),
                0.0
            )
        )

        pr_auc = holdout.get("pr_auc", 0.0)
        prec = holdout.get("precision", 0.0)
        rec = holdout.get("recall", 0.0)

        version_id = register_model_version(
            supabase_client=supabase_client,
            model_name=result.get("best_model", "unknown").upper(),
            band=band,
            f1_score=f1,
            pr_auc=pr_auc,
            precision_score=prec,
            recall_score=rec,
            model_path=result.get("model_path", ""),
            pipeline_path=result.get("pipeline_path", ""),
            dataset_range=dataset_range,
            holdout_metrics=holdout,
            notes=notes,
            deployment_status="Staging",
        )

        registered[band] = version_id

        # =====================================================
        # AUTO PROMOTE
        # =====================================================

        if auto_promote and version_id:

            ok = promote_to_active(
                supabase_client,
                version_id,
                band
            )

            promoted[band] = ok

    return {
        "registered": registered,
        "promoted": promoted,
    }


# =============================================================
# AUTO ROLLBACK ON DRIFT
# =============================================================

def auto_rollback_on_drift(
    supabase_client,
    drift_report: Dict,
    band: str = "global",
) -> Dict:
    """
    Automatically rollback Active model when severe drift occurs.
    """

    result = {
        "rollback_triggered": False,
        "demoted_version": None,
        "reason": "no drift",
    }

    if not drift_report:
        return result

    if not drift_report.get("retrain_recommended"):
        return result

    psi = drift_report.get("psi_score", 0.0) or 0.0
    ks_p = drift_report.get("ks_p_value", 1.0) or 1.0

    # =========================================================
    # HARD DRIFT THRESHOLDS
    # =========================================================

    severe_drift = (
        psi >= 0.25 or
        ks_p < 0.01
    )

    if not severe_drift:

        result["reason"] = (
            f"Drift below rollback threshold "
            f"(PSI={psi:.4f}, KS-p={ks_p:.4f})"
        )

        print(f"[MODEL REGISTRY] {result['reason']}")

        return result

    reason = (
        f"Auto rollback triggered "
        f"(PSI={psi:.4f}, KS-p={ks_p:.4f})"
    )

    rollback_result = rollback_to_previous(
        supabase_client=supabase_client,
        band=band,
        reason=reason,
    )

    result["rollback_triggered"] = (
        rollback_result.get("status") == "success"
    )

    result["demoted_version"] = rollback_result.get("demoted")

    result["reason"] = reason

    print(
        f"[MODEL REGISTRY] Auto rollback result → {rollback_result}"
    )

    return result


# =============================================================
# OFFLINE LOGGER
# =============================================================

def _log_offline(record: Dict):
    """
    Offline fallback logger when Supabase is unavailable.
    """

    log_path = os.path.join(
        "versioning",
        "offline_version_log.jsonl"
    )

    os.makedirs("versioning", exist_ok=True)

    try:

        with open(log_path, "a", encoding="utf-8") as f:

            f.write(
                json.dumps(record, default=str) + "\n"
            )

        print(
            f"[MODEL REGISTRY] Offline log saved → {log_path}"
        )

    except Exception as e:

        print(f"[MODEL REGISTRY] Offline log failed: {e}")