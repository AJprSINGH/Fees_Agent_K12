# =============================================================
# evaluation/evaluator.py
# Phase 5 — Evaluation, Explainability, Calibration & Fairness
#
# Covers all four Phase 5 pillars:
#   1. Primary metrics  (PR-AUC, F1, ROC-AUC, confusion matrix)
#   2. SHAP explainability  (TreeExplainer for XGB/LGB/RF,
#                            LinearExplainer for LogisticRegression)
#   3. Probability calibration  (CalibratedClassifierCV, sigmoid)
#   4. Fairness audit  (per-grade / per-division / per-year FPR/FNR)
#   5. Temporal stability  (feature-importance drift across CV folds)
#
# All public functions are pure: they accept fitted objects and
# DataFrames, and return plain dicts — no side-effects, no I/O.
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline    import Pipeline

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── local import for threshold constant ─────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from services.ml_prediction_service import THRESHOLD
except Exception:
    THRESHOLD = 0.4


# =============================================================
# 1. PRIMARY METRICS
# =============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = THRESHOLD,
    label: str = "",
) -> Dict:
    """
    Full primary evaluation suite.

    Returns
    -------
    {
        "label":       str,
        "threshold":   float,
        "samples":     int,
        "f1":          float,
        "precision":   float,
        "recall":      float,
        "roc_auc":     float,
        "pr_auc":      float,
        "brier":       float,
        "confusion_matrix": {"tp", "fp", "tn", "fn",
                             "fpr", "fnr", "tpr"},
    }
    """
    y_pred = (y_proba >= threshold).astype(int)
    n      = len(y_true)

    if n == 0:
        return {"label": label, "samples": 0, "error": "empty split"}

    # --- scalar metrics ----------------------------------------
    f1    = f1_score(y_true, y_pred, zero_division=0)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_proba)

    try:
        roc = roc_auc_score(y_true, y_proba)
    except Exception:
        roc = 0.0

    try:
        pr_auc = average_precision_score(y_true, y_proba)
    except Exception:
        pr_auc = 0.0

    # --- confusion matrix ---------------------------------------
    cm       = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    fpr = round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4)
    fnr = round(fn / (fn + tp) if (fn + tp) > 0 else 0.0, 4)
    tpr = round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4)

    return {
        "label":     label,
        "threshold": threshold,
        "samples":   n,
        "f1":        round(float(f1),    4),
        "precision": round(float(prec),  4),
        "recall":    round(float(rec),   4),
        "roc_auc":   round(float(roc),   4),
        "pr_auc":    round(float(pr_auc), 4),
        "brier":     round(float(brier), 4),
        "confusion_matrix": {
            "tp":  tp,
            "fp":  fp,
            "tn":  tn,
            "fn":  fn,
            "fpr": fpr,
            "fnr": fnr,
            "tpr": tpr,
        },
    }


# =============================================================
# 2. SHAP EXPLAINABILITY
# =============================================================

def _unwrap_model(model: Any) -> Any:
    """
    Unwrap sklearn Pipeline or CalibratedClassifierCV to get
    the base estimator that SHAP understands.
    """
    if isinstance(model, Pipeline):
        return model.named_steps.get("model", model.steps[-1][1])
    if isinstance(model, CalibratedClassifierCV):
        # Take the first calibrated classifier's base estimator
        try:
            return model.calibrated_classifiers_[0].estimator
        except Exception:
            return model
    return model


def _build_explainer(base_model: Any, X_background: pd.DataFrame):
    """
    Build the right SHAP explainer for the given model type.
    Uses TreeExplainer for tree models (XGB/LGB/RF),
    LinearExplainer for LogisticRegression.
    Returns (explainer, explainer_type_str) or (None, "unavailable").
    """
    if not SHAP_AVAILABLE:
        return None, "shap_not_installed"

    model_class = type(base_model).__name__

    try:
        if model_class in ("XGBClassifier", "LGBMClassifier",
                           "RandomForestClassifier",
                           "GradientBoostingClassifier",
                           "DecisionTreeClassifier"):
            explainer = shap.TreeExplainer(
                base_model,
                feature_perturbation="interventional",
                model_output="probability",
            )
            return explainer, "TreeExplainer"

        elif model_class in ("LogisticRegression",):
            # LinearExplainer needs a background dataset
            bg = shap.sample(X_background, min(100, len(X_background)))
            explainer = shap.LinearExplainer(base_model, bg)
            return explainer, "LinearExplainer"

        else:
            # Fallback: KernelExplainer (slow but universal)
            bg = shap.sample(X_background, min(50, len(X_background)))
            explainer = shap.KernelExplainer(
                base_model.predict_proba, bg
            )
            return explainer, "KernelExplainer"

    except Exception as e:
        print(f"[SHAP] explainer build error ({model_class}): {e}")
        return None, f"error: {e}"


def explain_prediction(
    model: Any,
    X_single: pd.DataFrame,
    X_background: pd.DataFrame,
    top_n: int = 5,
) -> Dict:
    """
    Generate a SHAP explanation for ONE student row.

    Parameters
    ----------
    model        : fitted sklearn/xgb/lgb model (or Pipeline)
    X_single     : single-row DataFrame aligned to model features
    X_background : background dataset for explainer initialisation
    top_n        : number of top contributing features to return

    Returns
    -------
    {
        "shap_available":  bool,
        "explainer_type":  str,
        "top_factors": [
            {"feature": "log_unpaid", "impact": 0.23,
             "direction": "increases_risk"},
            ...
        ],
        "explanation_summary": str,
    }
    """
    if not SHAP_AVAILABLE:
        return {
            "shap_available":    False,
            "explainer_type":    "shap_not_installed",
            "top_factors":       [],
            "explanation_summary": "SHAP not installed. Add `shap` to requirements.txt.",
        }

    try:
        base_model = _unwrap_model(model)
        explainer, etype = _build_explainer(base_model, X_background)

        if explainer is None:
            return {
                "shap_available":    False,
                "explainer_type":    etype,
                "top_factors":       [],
                "explanation_summary": f"SHAP explainer unavailable: {etype}",
            }

        shap_values = explainer.shap_values(X_single)

        # TreeExplainer for binary classification returns list[2];
        # we want the positive-class values (index 1)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            sv = np.array(shap_values[1][0])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            sv = shap_values[0]
        else:
            sv = np.array(shap_values).flatten()

        feature_names = list(X_single.columns)
        pairs = sorted(
            zip(feature_names, sv.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        top_factors = []
        for feat, impact in pairs[:top_n]:
            top_factors.append({
                "feature":   feat,
                "impact":    round(float(impact), 4),
                "direction": "increases_risk" if impact > 0 else "decreases_risk",
            })

        # Human-readable summary
        if top_factors:
            top = top_factors[0]
            summary = (
                f"Primary driver: '{top['feature']}' "
                f"({'pushes risk up' if top['impact'] > 0 else 'pushes risk down'}, "
                f"SHAP={top['impact']:+.3f}). "
                f"Top {len(top_factors)} features shown."
            )
        else:
            summary = "No SHAP factors computed."

        return {
            "shap_available":    True,
            "explainer_type":    etype,
            "top_factors":       top_factors,
            "explanation_summary": summary,
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "shap_available":    False,
            "explainer_type":    "error",
            "top_factors":       [],
            "explanation_summary": f"SHAP error: {e}",
        }


def explain_batch(
    model: Any,
    X: pd.DataFrame,
    top_n: int = 5,
) -> List[Dict]:
    """
    SHAP explanations for every row in X.
    Uses X itself as background (fast; suitable for batch reporting).
    Returns list of explanation dicts, one per row.
    """
    explanations = []
    for i in range(len(X)):
        row_df = X.iloc[[i]]
        exp    = explain_prediction(model, row_df, X, top_n=top_n)
        explanations.append(exp)
    return explanations


def compute_feature_importance(
    model: Any,
    feature_names: List[str],
) -> List[Dict]:
    """
    Extract native feature importances (for tree models) or
    absolute coefficients (for LogisticRegression).
    Returns list sorted by importance descending.
    """
    base = _unwrap_model(model)
    cls  = type(base).__name__

    try:
        if hasattr(base, "feature_importances_"):
            imps = base.feature_importances_
        elif hasattr(base, "coef_"):
            imps = np.abs(base.coef_).flatten()
        else:
            return []

        pairs = sorted(
            zip(feature_names, imps.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            {"feature": f, "importance": round(float(v), 6)}
            for f, v in pairs
        ]
    except Exception:
        return []


# =============================================================
# 3. PROBABILITY CALIBRATION
# =============================================================

def calibrate_model(
    model: Any,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    method: str = "sigmoid",
) -> Tuple[Any, Dict]:
    """
    Wrap a fitted model in CalibratedClassifierCV (cv="prefit")
    using Platt scaling (sigmoid) or isotonic regression.

    Parameters
    ----------
    model   : already-fitted sklearn/pipeline estimator
    X_cal   : calibration set features  (held-out, not training data)
    y_cal   : calibration set labels
    method  : "sigmoid" (Platt) or "isotonic"

    Returns
    -------
    (calibrated_model, calibration_stats_dict)
    """
    try:
        cal_model = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv="prefit",
        )
        cal_model.fit(X_cal, y_cal)

        # Brier score before vs after
        proba_raw = model.predict_proba(X_cal)[:, 1]
        proba_cal = cal_model.predict_proba(X_cal)[:, 1]

        brier_before = round(float(brier_score_loss(y_cal, proba_raw)), 4)
        brier_after  = round(float(brier_score_loss(y_cal, proba_cal)), 4)

        # Calibration curve (10 bins)
        try:
            frac_pos_before, mean_pred_before = calibration_curve(
                y_cal, proba_raw, n_bins=10
            )
            frac_pos_after, mean_pred_after   = calibration_curve(
                y_cal, proba_cal, n_bins=10
            )
            cal_curve = {
                "before": {
                    "mean_predicted": [round(float(v), 4) for v in mean_pred_before],
                    "fraction_positive": [round(float(v), 4) for v in frac_pos_before],
                },
                "after": {
                    "mean_predicted": [round(float(v), 4) for v in mean_pred_after],
                    "fraction_positive": [round(float(v), 4) for v in frac_pos_after],
                },
            }
        except Exception:
            cal_curve = {}

        stats = {
            "calibration_method": method,
            "brier_before":       brier_before,
            "brier_after":        brier_after,
            "brier_improvement":  round(brier_before - brier_after, 4),
            "calibration_curve":  cal_curve,
            "calibration_samples": len(y_cal),
        }

        return cal_model, stats

    except Exception as e:
        traceback.print_exc()
        return model, {"error": str(e), "calibration_method": method}


# =============================================================
# 4. FAIRNESS AUDIT
# =============================================================

# Groups to audit; all are integer-encoded after preprocess()
_FAIRNESS_GROUPS = {
    "grade_encoded":    "Grade",
    "division_encoded": "Division",
    "grade_band":       "Grade Band",
}

_GRADE_BAND_LABELS = {0: "Nursery-I", 1: "Primary (II-V)", 2: "Secondary (VI-XII)"}


def fairness_audit(
    y_true:  pd.Series,
    y_proba: np.ndarray,
    df_meta: pd.DataFrame,
    threshold: float = THRESHOLD,
    min_group_size: int = 5,
) -> Dict:
    """
    Compute per-subgroup metrics and flag disparate impact.

    Parameters
    ----------
    y_true        : ground-truth labels (aligned index with df_meta)
    y_proba       : predicted probabilities
    df_meta       : DataFrame with grouping columns (grade_encoded etc.)
    threshold     : decision threshold
    min_group_size: skip groups smaller than this

    Returns
    -------
    {
        "groups": {
            "grade_encoded": [
                {"value": 6, "label": "VI", "samples": 120,
                 "precision": .., "recall": .., "f1": ..,
                 "fpr": .., "fnr": .., "default_rate": ..},
                ...
            ],
            ...
        },
        "disparate_impact": {
            "grade_encoded": {
                "max_fpr_group": .., "min_fpr_group": ..,
                "fpr_ratio": .., "flag": bool
            },
            ...
        },
        "overall": <compute_metrics result>
    }
    """
    y_pred = (y_proba >= threshold).astype(int)

    audit = {
        "overall":          compute_metrics(
            y_true.values, y_proba, threshold, label="overall"
        ),
        "groups":           {},
        "disparate_impact": {},
    }

    for col, col_label in _FAIRNESS_GROUPS.items():
        if col not in df_meta.columns:
            continue

        group_results = []
        groups        = sorted(df_meta[col].dropna().unique())

        for gval in groups:
            mask  = df_meta[col] == gval
            idx   = df_meta.index[mask]
            yt    = y_true.loc[idx]
            yp    = y_proba[mask.values]

            if len(yt) < min_group_size:
                continue

            m = compute_metrics(yt.values, yp, threshold, label=str(gval))

            entry = {
                "value":        int(gval),
                "label":        _GRADE_BAND_LABELS.get(int(gval), str(gval)),
                "samples":      m["samples"],
                "precision":    m["precision"],
                "recall":       m["recall"],
                "f1":           m["f1"],
                "fpr":          m["confusion_matrix"]["fpr"],
                "fnr":          m["confusion_matrix"]["fnr"],
                "default_rate": round(float(yt.mean()), 4),
                "confusion_matrix": m["confusion_matrix"],
            }
            group_results.append(entry)

        if not group_results:
            continue

        audit["groups"][col] = group_results

        # --- Disparate Impact check ----------------------------
        fprs = [g["fpr"] for g in group_results if g["samples"] >= min_group_size]
        if len(fprs) >= 2:
            max_fpr = max(fprs)
            min_fpr = min(fprs)
            ratio   = round(max_fpr / min_fpr if min_fpr > 0 else float("inf"), 3)
            # Flag if max FPR is more than 80% higher than min FPR  (4/5ths rule)
            flag    = (max_fpr - min_fpr) > 0.10 or (min_fpr > 0 and ratio > 1.8)

            max_grp = group_results[fprs.index(max_fpr)]
            min_grp = group_results[fprs.index(min_fpr)]

            audit["disparate_impact"][col] = {
                "group_column":  col,
                "group_label":   col_label,
                "max_fpr":       round(max_fpr, 4),
                "max_fpr_group": max_grp["label"],
                "min_fpr":       round(min_fpr, 4),
                "min_fpr_group": min_grp["label"],
                "fpr_ratio":     ratio,
                "flag":          flag,
                "note": (
                    "⚠ Disparate impact detected — review model bias."
                    if flag else
                    "✓ FPR spread within acceptable range."
                ),
            }

    return audit


# =============================================================
# 5. TEMPORAL STABILITY
# =============================================================

def temporal_stability_report(
    fold_importances: List[Dict],
) -> Dict:
    """
    Measure how much feature importances shift across CV folds.

    Parameters
    ----------
    fold_importances : list of dicts, each produced by
                       compute_feature_importance(), one per fold.

    Returns
    -------
    {
        "features": [
            {"feature": str,
             "mean_importance": float,
             "std_importance":  float,
             "cv_coefficient":  float,   # std / mean  (instability ratio)
             "stable":          bool},
            ...
        ],
        "unstable_features": [list of names with cv > 0.5],
        "overall_stability": "stable" | "moderate" | "unstable",
    }
    """
    if not fold_importances:
        return {"features": [], "unstable_features": [],
                "overall_stability": "unknown"}

    # Collect all feature names
    all_features = sorted({
        entry["feature"]
        for fold in fold_importances
        for entry in fold
    })

    rows = []
    unstable = []

    for feat in all_features:
        vals = []
        for fold in fold_importances:
            lookup = {e["feature"]: e["importance"] for e in fold}
            vals.append(lookup.get(feat, 0.0))

        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        cv   = round(std / mean if mean > 1e-9 else 0.0, 3)
        stable = cv < 0.5

        rows.append({
            "feature":          feat,
            "mean_importance":  round(mean, 6),
            "std_importance":   round(std,  6),
            "cv_coefficient":   cv,
            "stable":           stable,
        })

        if not stable:
            unstable.append(feat)

    rows.sort(key=lambda x: x["mean_importance"], reverse=True)

    pct_unstable = len(unstable) / max(len(all_features), 1)
    if pct_unstable < 0.15:
        stability = "stable"
    elif pct_unstable < 0.40:
        stability = "moderate"
    else:
        stability = "unstable"

    return {
        "features":          rows,
        "unstable_features": unstable,
        "overall_stability": stability,
        "pct_unstable":      round(pct_unstable * 100, 1),
        "note": (
            "Feature importances are consistent across folds — "
            "model is not overfitting to specific academic years."
            if stability == "stable" else
            f"{len(unstable)} feature(s) show high importance variance — "
            "check for COVID-era or structural economic shifts in the data."
        ),
    }


# =============================================================
# CONVENIENCE: FULL PHASE 5 REPORT
# =============================================================

def full_evaluation_report(
    model:        Any,
    X_holdout:    pd.DataFrame,
    y_holdout:    pd.Series,
    df_meta:      pd.DataFrame,
    fold_importances: Optional[List[Dict]] = None,
    band_name:    str = "",
    calibrate:    bool = True,
    explain_sample_size: int = 0,
) -> Dict:
    """
    One-shot Phase 5 report for a single grade-band model.

    Parameters
    ----------
    model             : fitted model (Pipeline / raw estimator)
    X_holdout         : holdout feature matrix
    y_holdout         : holdout labels
    df_meta           : original (unencoded) columns for fairness groups
    fold_importances  : list of per-fold importance dicts (from training CV)
    band_name         : "nursery" / "primary" / "secondary"
    calibrate         : whether to run calibration (needs enough samples)
    explain_sample_size: if > 0, run SHAP on a sample of this many rows

    Returns
    -------
    Full evaluation dict ready to attach to the training results.
    """
    report = {"band": band_name, "phase": "Phase 5"}

    # -- 1. Primary metrics ----------------------------------------
    proba = model.predict_proba(X_holdout)[:, 1]
    report["metrics"] = compute_metrics(
        y_holdout.values, proba, label=band_name
    )

    # -- 2. Fairness audit -----------------------------------------
    df_meta_aligned = df_meta.reset_index(drop=True)
    y_aligned       = y_holdout.reset_index(drop=True)
    report["fairness"] = fairness_audit(y_aligned, proba, df_meta_aligned)

    # -- 3. Calibration --------------------------------------------
    if calibrate and len(y_holdout) >= 20 and len(y_holdout.unique()) > 1:
        cal_model, cal_stats = calibrate_model(model, X_holdout, y_holdout)
        report["calibration"] = cal_stats
        # Re-run metrics with calibrated probabilities
        proba_cal = cal_model.predict_proba(X_holdout)[:, 1]
        report["metrics_calibrated"] = compute_metrics(
            y_holdout.values, proba_cal, label=f"{band_name}_calibrated"
        )
        report["calibrated_model"] = cal_model
    else:
        report["calibration"] = {"note": "Skipped — insufficient holdout samples."}
        report["calibrated_model"] = None

    # -- 4. Temporal stability -------------------------------------
    if fold_importances:
        report["temporal_stability"] = temporal_stability_report(fold_importances)
    else:
        report["temporal_stability"] = {
            "note": "No fold importance data supplied."
        }

    # -- 5. SHAP sample explanation --------------------------------
    if explain_sample_size > 0 and SHAP_AVAILABLE:
        sample_size = min(explain_sample_size, len(X_holdout))
        X_sample    = X_holdout.iloc[:sample_size]
        exps        = explain_batch(model, X_sample, top_n=5)
        report["shap_sample"] = {
            "rows_explained":   sample_size,
            "explanations":     exps,
        }
    else:
        report["shap_sample"] = {
            "note": "SHAP sample not requested or shap not installed."
        }

    return report
