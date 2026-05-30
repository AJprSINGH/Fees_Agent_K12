# =============================================================
# monitoring/drift_monitor.py
# Phase 6 — Drift Monitoring
#
# Tracks:
#   1. Feature drift  — Population Stability Index (PSI) per term
#   2. Prediction drift — mean risk-score shift between periods
#   3. Class distribution drift — default-rate shift
#
# All functions are pure: accept DataFrames / arrays, return dicts.
# No I/O side-effects. Logging only via print() for HF Spaces.
#
# Usage (scheduled, called from scheduling/batch_pipeline.py):
#   from monitoring.drift_monitor import run_drift_check
#   report = run_drift_check(baseline_df, current_df, model_version)
#   if report["drift_detected"]:
#       ... trigger alert / retrain recommendation ...
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================
# CONSTANTS
# =============================================================

# PSI thresholds (industry standard)
PSI_OK       = 0.10   # < 0.10  → no significant drift
PSI_WARNING  = 0.20   # 0.10–0.20 → moderate drift, monitor
PSI_CRITICAL = 0.20   # > 0.20  → significant drift, retrain

# Prediction mean shift threshold (absolute delta in risk score)
PRED_SHIFT_WARNING  = 0.05   # 5 percentage points
PRED_SHIFT_CRITICAL = 0.10   # 10 percentage points

# Default-rate relative shift threshold
DEFAULT_RATE_SHIFT_WARNING  = 0.10   # 10 % relative change
DEFAULT_RATE_SHIFT_CRITICAL = 0.20   # 20 % relative change

# Features to monitor for PSI (numerical, meaningful for fees)
MONITORED_FEATURES = [
    "total_unpaid",
    "days_overdue",
    "log_unpaid",
    "severity_score",
    "payment_delay_ratio",
    "grade_encoded",
    "grade_band",
    "fee_tier_encoded",
    "payment_momentum_encoded",
    "years_enrolled",
    "years_with_payment",
    "years_without_payment",
    "consecutive_default_years",
    "partial_payment_ratio",
]


# =============================================================
# INTERNAL HELPERS
# =============================================================

def _psi_score(expected: np.ndarray, actual: np.ndarray,
               buckets: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)

    Parameters
    ----------
    expected : baseline distribution (training or prior period)
    actual   : current distribution (inference period)
    buckets  : number of quantile bins (default 10 = deciles)

    Returns
    -------
    float — PSI value  (0 = identical, > 0.2 = significant drift)
    """
    expected = np.array(expected, dtype=float)
    actual   = np.array(actual, dtype=float)

    # Remove NaN
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Build breakpoints from expected distribution
    breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    # Count in each bin
    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual,   bins=breakpoints)[0]

    # Convert to proportions, add small epsilon to avoid log(0)
    eps = 1e-6
    exp_pct = exp_counts / len(expected) + eps
    act_pct = act_counts / len(actual)   + eps

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return round(float(psi), 6)


def _psi_label(psi: float) -> str:
    if psi < PSI_OK:
        return "OK"
    if psi < PSI_CRITICAL:
        return "WARNING"
    return "CRITICAL"


def _safe_col(df: pd.DataFrame, col: str) -> Optional[np.ndarray]:
    """Return numeric values for col, or None if absent."""
    if col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna().values
    return vals if len(vals) > 0 else None


# =============================================================
# 1. FEATURE DRIFT — PSI PER COLUMN
# =============================================================

def compute_feature_drift(
    baseline_df: pd.DataFrame,
    current_df:  pd.DataFrame,
    features:    Optional[List[str]] = None,
    buckets:     int = 10,
) -> Dict:
    """
    Compute PSI for each monitored feature between baseline and current.

    Parameters
    ----------
    baseline_df : DataFrame from training period (or prior term)
    current_df  : DataFrame from current inference period
    features    : list of column names to check; defaults to MONITORED_FEATURES
    buckets     : PSI quantile bins

    Returns
    -------
    {
        "feature_psi": {col: {"psi": float, "label": str}, ...},
        "worst_psi":   float,
        "worst_feature": str,
        "drift_detected": bool,
        "warning_features": [str, ...],
        "critical_features": [str, ...],
    }
    """
    if features is None:
        features = MONITORED_FEATURES

    feature_psi     = {}
    warning_feats   = []
    critical_feats  = []
    worst_psi       = 0.0
    worst_feature   = ""

    for col in features:
        base_vals = _safe_col(baseline_df, col)
        curr_vals = _safe_col(current_df,  col)

        if base_vals is None or curr_vals is None:
            feature_psi[col] = {"psi": None, "label": "MISSING"}
            continue

        psi   = _psi_score(base_vals, curr_vals, buckets)
        label = _psi_label(psi)
        feature_psi[col] = {"psi": psi, "label": label}

        if psi > worst_psi:
            worst_psi     = psi
            worst_feature = col

        if label == "WARNING":
            warning_feats.append(col)
        elif label == "CRITICAL":
            critical_feats.append(col)

    drift_detected = len(critical_feats) > 0

    return {
        "feature_psi":        feature_psi,
        "worst_psi":          round(worst_psi, 6),
        "worst_feature":      worst_feature,
        "drift_detected":     drift_detected,
        "warning_features":   warning_feats,
        "critical_features":  critical_feats,
        "summary": (
            f"{len(critical_feats)} critical, "
            f"{len(warning_feats)} warning, "
            f"{len(feature_psi) - len(critical_feats) - len(warning_feats)} OK"
        ),
    }


# =============================================================
# 2. PREDICTION DRIFT — MEAN RISK SCORE SHIFT
# =============================================================

def compute_prediction_drift(
    baseline_scores: np.ndarray,
    current_scores:  np.ndarray,
) -> Dict:
    """
    Detect shift in the distribution of predicted risk scores.

    Parameters
    ----------
    baseline_scores : 1-D array of risk scores from prior period
    current_scores  : 1-D array of risk scores from current period

    Returns
    -------
    {
        "baseline_mean":  float,
        "current_mean":   float,
        "mean_shift":     float,   (positive = scores increased)
        "label":          str,     (OK / WARNING / CRITICAL)
        "drift_detected": bool,
        "psi":            float,   (PSI on score distributions)
    }
    """
    baseline_scores = np.array(baseline_scores, dtype=float)
    current_scores  = np.array(current_scores,  dtype=float)

    baseline_scores = baseline_scores[~np.isnan(baseline_scores)]
    current_scores  = current_scores[~np.isnan(current_scores)]

    if len(baseline_scores) == 0 or len(current_scores) == 0:
        return {
            "baseline_mean": None, "current_mean": None,
            "mean_shift": None, "label": "NO_DATA",
            "drift_detected": False, "psi": None,
        }

    base_mean  = float(np.mean(baseline_scores))
    curr_mean  = float(np.mean(current_scores))
    mean_shift = abs(curr_mean - base_mean)

    psi = _psi_score(baseline_scores, current_scores, buckets=10)

    if mean_shift >= PRED_SHIFT_CRITICAL or psi >= PSI_CRITICAL:
        label           = "CRITICAL"
        drift_detected  = True
    elif mean_shift >= PRED_SHIFT_WARNING or psi >= PSI_OK:
        label           = "WARNING"
        drift_detected  = False
    else:
        label           = "OK"
        drift_detected  = False

    return {
        "baseline_mean":  round(base_mean,  4),
        "current_mean":   round(curr_mean,  4),
        "mean_shift":     round(mean_shift, 4),
        "direction":      "increased" if curr_mean > base_mean else "decreased",
        "label":          label,
        "drift_detected": drift_detected,
        "psi":            round(psi, 6),
    }


# =============================================================
# 3. TARGET (DEFAULT-RATE) DRIFT
# =============================================================

def compute_target_drift(
    baseline_labels: np.ndarray,
    current_labels:  np.ndarray,
) -> Dict:
    """
    Detect shift in observed default rate between periods.

    Parameters
    ----------
    baseline_labels : binary ground-truth labels (0/1) from prior period
    current_labels  : binary ground-truth labels (0/1) from current period

    Returns
    -------
    {
        "baseline_default_rate": float,
        "current_default_rate":  float,
        "relative_shift":        float,  (fractional change)
        "label":                 str,
        "drift_detected":        bool,
    }
    """
    baseline_labels = np.array(baseline_labels, dtype=float)
    current_labels  = np.array(current_labels,  dtype=float)

    baseline_labels = baseline_labels[~np.isnan(baseline_labels)]
    current_labels  = current_labels[~np.isnan(current_labels)]

    if len(baseline_labels) == 0 or len(current_labels) == 0:
        return {
            "baseline_default_rate": None, "current_default_rate": None,
            "relative_shift": None, "label": "NO_DATA", "drift_detected": False,
        }

    base_rate = float(np.mean(baseline_labels))
    curr_rate = float(np.mean(current_labels))

    relative_shift = (
        abs(curr_rate - base_rate) / max(base_rate, 1e-6)
    )

    if relative_shift >= DEFAULT_RATE_SHIFT_CRITICAL:
        label          = "CRITICAL"
        drift_detected = True
    elif relative_shift >= DEFAULT_RATE_SHIFT_WARNING:
        label          = "WARNING"
        drift_detected = False
    else:
        label          = "OK"
        drift_detected = False

    return {
        "baseline_default_rate": round(base_rate,      4),
        "current_default_rate":  round(curr_rate,      4),
        "baseline_default_pct":  round(base_rate * 100, 2),
        "current_default_pct":   round(curr_rate * 100, 2),
        "absolute_shift":        round(abs(curr_rate - base_rate), 4),
        "relative_shift":        round(relative_shift, 4),
        "direction":             "increased" if curr_rate > base_rate else "decreased",
        "label":                 label,
        "drift_detected":        drift_detected,
    }


# =============================================================
# 4. ORCHESTRATOR — FULL DRIFT CHECK
# =============================================================

def run_drift_check(
    baseline_df:     pd.DataFrame,
    current_df:      pd.DataFrame,
    model_version:   str = "unknown",
    baseline_scores: Optional[np.ndarray] = None,
    current_scores:  Optional[np.ndarray] = None,
    baseline_labels: Optional[np.ndarray] = None,
    current_labels:  Optional[np.ndarray] = None,
) -> Dict:
    """
    Full drift check across features, predictions, and targets.

    This is the main entry point called by the scheduling pipeline
    at the end of each term / after batch inference.

    Parameters
    ----------
    baseline_df     : preprocessed DataFrame from training period
    current_df      : preprocessed DataFrame from current period
    model_version   : version string for logging
    baseline_scores : optional prior-period risk scores
    current_scores  : optional current-period risk scores
    baseline_labels : optional prior-period ground-truth labels
    current_labels  : optional current-period ground-truth labels

    Returns
    -------
    Full drift report dict including:
        drift_detected     : bool  — True if ANY metric is CRITICAL
        retrain_recommended: bool  — True if drift warrants retraining
        feature_drift      : dict
        prediction_drift   : dict (if scores supplied)
        target_drift       : dict (if labels supplied)
        timestamp          : ISO-8601 string
        model_version      : str
        alert_message      : str  — human-readable summary
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    print("\n" + "=" * 60)
    print(f"[DRIFT MONITOR] Running drift check — model={model_version}")
    print(f"  baseline rows: {len(baseline_df)}  |  current rows: {len(current_df)}")
    print("=" * 60)

    # ── Feature drift ─────────────────────────────────────────
    try:
        feat_drift = compute_feature_drift(baseline_df, current_df)
    except Exception as e:
        feat_drift = {"error": str(e), "drift_detected": False}
        traceback.print_exc()

    # ── Prediction drift ──────────────────────────────────────
    pred_drift = None
    if baseline_scores is not None and current_scores is not None:
        try:
            pred_drift = compute_prediction_drift(baseline_scores, current_scores)
        except Exception as e:
            pred_drift = {"error": str(e), "drift_detected": False}

    # ── Target drift ──────────────────────────────────────────
    tgt_drift = None
    if baseline_labels is not None and current_labels is not None:
        try:
            tgt_drift = compute_target_drift(baseline_labels, current_labels)
        except Exception as e:
            tgt_drift = {"error": str(e), "drift_detected": False}

    # ── Aggregate ─────────────────────────────────────────────
    any_drift = feat_drift.get("drift_detected", False)
    if pred_drift:
        any_drift = any_drift or pred_drift.get("drift_detected", False)
    if tgt_drift:
        any_drift = any_drift or tgt_drift.get("drift_detected", False)

    retrain_recommended = any_drift

    # Build human-readable alert
    parts = []
    if feat_drift.get("critical_features"):
        parts.append(
            f"Feature PSI critical for: "
            f"{', '.join(feat_drift['critical_features'])}"
        )
    if pred_drift and pred_drift.get("drift_detected"):
        shift = pred_drift.get("mean_shift", 0)
        parts.append(f"Prediction mean shift: {shift:.3f} (CRITICAL)")
    if tgt_drift and tgt_drift.get("drift_detected"):
        rel = tgt_drift.get("relative_shift", 0)
        parts.append(f"Default rate shifted {rel * 100:.1f}% relatively (CRITICAL)")

    alert_message = (
        "DRIFT ALERT: " + " | ".join(parts)
        if parts else
        "No critical drift detected."
    )

    if any_drift:
        alert_message += " → Retraining recommended."

    print(f"[DRIFT MONITOR] {alert_message}")

    report = {
        "timestamp":            timestamp,
        "model_version":        model_version,
        "drift_detected":       any_drift,
        "retrain_recommended":  retrain_recommended,
        "alert_message":        alert_message,
        "feature_drift":        feat_drift,
        "prediction_drift":     pred_drift,
        "target_drift":         tgt_drift,
        "baseline_rows":        len(baseline_df),
        "current_rows":         len(current_df),
    }

    return report


# =============================================================
# 5. SUPABASE LOGGER
# =============================================================

def log_drift_report_to_supabase(report: Dict, supabase_client) -> bool:
    """
    Persist a drift report dict to the drift_monitoring_log Supabase table.

    Table schema (create once):
        CREATE TABLE drift_monitoring_log (
            id                  BIGSERIAL PRIMARY KEY,
            timestamp           TIMESTAMPTZ NOT NULL,
            model_version       TEXT,
            drift_detected      BOOLEAN,
            retrain_recommended BOOLEAN,
            alert_message       TEXT,
            worst_psi_feature   TEXT,
            worst_psi           NUMERIC,
            prediction_shift    NUMERIC,
            default_rate_shift  NUMERIC,
            full_report         JSONB
        );

    Returns True on success, False on failure.
    """
    if supabase_client is None:
        print("[DRIFT MONITOR] Supabase not configured — skipping log.")
        return False

    try:
        import json

        feat = report.get("feature_drift", {})
        pred = report.get("prediction_drift") or {}
        tgt  = report.get("target_drift")    or {}

        row = {
            "timestamp":            report["timestamp"],
            "model_version":        report.get("model_version", "unknown"),
            "drift_detected":       report.get("drift_detected", False),
            "retrain_recommended":  report.get("retrain_recommended", False),
            "alert_message":        report.get("alert_message", ""),
            "worst_psi_feature":    feat.get("worst_feature", ""),
            "worst_psi":            feat.get("worst_psi"),
            "prediction_shift":     pred.get("mean_shift"),
            "default_rate_shift":   tgt.get("relative_shift"),
            "full_report":          json.dumps(report, default=str),
        }

        supabase_client.table("drift_monitoring_log").insert(row).execute()
        print(f"[DRIFT MONITOR] Report logged to Supabase — {report['timestamp']}")
        return True

    except Exception as e:
        print(f"[DRIFT MONITOR] Supabase log failed: {e}")
        return False
