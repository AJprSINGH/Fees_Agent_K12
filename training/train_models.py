# =============================================================
# training/train_models.py
# Phase 4 — Offline Model Training
#
# FIXES APPLIED:
#   FIX 1: XGBClassifier use_label_encoder removed (deprecated in XGBoost >= 1.6)
#   FIX 2: Graceful fallback when XGBoost / LightGBM not installed
#   FIX 3: Absolute BASE_DIR path resolution — works from any cwd
#   FIX 4: Robust CLI entry point with clear success/failure output
#   FIX 5: VotingClassifier only built when all sub-estimators available
#   FIX 6: HP tuning safely skips when dataset too small
#   FIX 7: _apply_smote fallback when imbalanced-learn not installed
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import traceback

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    TimeSeriesSplit,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    f1_score,
    average_precision_score,
)

# ── Resolve BASE_DIR absolutely so the script works from any cwd ──────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR    = os.path.join(BASE_DIR, "models")
PIPELINE_DIR = os.path.join(BASE_DIR, "pipelines")

# ── Optional heavy dependencies — degrade gracefully ─────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[TRAIN] WARNING — xgboost not installed; XGBoost model skipped.")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[TRAIN] WARNING — lightgbm not installed; LightGBM model skipped.")

try:
    from sklearn.ensemble import VotingClassifier
    _VOTING_AVAILABLE = True
except ImportError:
    _VOTING_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTENC, SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[TRAIN] WARNING — imbalanced-learn not installed; SMOTE disabled.")

# =============================================================
# PROJECT IMPORTS
# =============================================================
sys.path.insert(0, BASE_DIR)

from evaluation.evaluator import (
    full_evaluation_report,
    compute_feature_importance,
    temporal_stability_report,
)

from services.ml_prediction_service import (
    filter_eligible_cohort,
    preprocess,
    _apply_smote,
    evaluate_predictions,
    ALL_YEAR_COLS,
    THRESHOLD,
)

# =============================================================
# CONSTANTS
# =============================================================
SHAP_EXPLAIN_SAMPLE = int(os.environ.get("SHAP_EXPLAIN_SAMPLE", "50"))
HP_TUNING_ENABLED   = (os.environ.get("HP_TUNING", "1") != "0")
HP_N_ITER           = int(os.environ.get("HP_N_ITER", "20"))

NURSERY_GRADES   = ["Nursery", "KG", "NR", "I"]
PRIMARY_GRADES   = ["II", "III", "IV", "V"]
SECONDARY_GRADES = ["VI", "VII", "VIII", "IX", "X", "XI", "XII"]

GRADE_MAP = {
    "Nursery": 0, "NR": 0, "KG": 0,
    # JR (Junior KG) and SR (Senior KG) — present in your data
    "JR": 0, "SR": 0,
    "I": 1,   "II": 2,   "III": 3, "IV": 4,  "V": 5,
    "VI": 6,  "VII": 7,  "VIII": 8, "IX": 9,  "X": 10,
    "XI": 11, "XII": 12,
}

BAND_DEFINITIONS = {
    "nursery":   NURSERY_GRADES,
    "primary":   PRIMARY_GRADES,
    "secondary": SECONDARY_GRADES,
}

# =============================================================
# HELPERS
# =============================================================
def _log(title: str, value=None):
    print("\n" + "=" * 60)
    print(title)
    if value is not None:
        print(value)
    print("=" * 60)


def _split_by_grade_band(df: pd.DataFrame, grade_col: str) -> dict:
    gb = df["grade_band"]
    return {
        "nursery":   df[gb == 0].copy(),
        "primary":   df[gb == 1].copy(),
        "secondary": df[gb == 2].copy(),
    }

# =============================================================
# HYPERPARAMETER SEARCH SPACES
# =============================================================
_HP_GRIDS = {}

if XGBOOST_AVAILABLE:
    _HP_GRIDS["xgboost"] = {
        "n_estimators":     [100, 200, 300, 400],
        "max_depth":        [3, 4, 5, 6, 7],
        "learning_rate":    [0.01, 0.03, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }

if LIGHTGBM_AVAILABLE:
    _HP_GRIDS["lightgbm"] = {
        "n_estimators":      [100, 200, 300, 400],
        "learning_rate":     [0.01, 0.03, 0.05, 0.1],
        "num_leaves":        [15, 31, 50, 63],
        "max_depth":         [-1, 4, 6, 8],
        "min_child_samples": [5, 10, 20, 30],
        "subsample":         [0.7, 0.8, 0.9, 1.0],
    }

_HP_GRIDS["random_forest"] = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [None, 5, 8, 12],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":    ["sqrt", "log2", 0.5],
}

# =============================================================
# HYPERPARAMETER TUNING
# =============================================================
def _tune_model(name: str, base_model, X: pd.DataFrame, y: pd.Series, cat_indices: list):
    if not HP_TUNING_ENABLED:
        return base_model

    grid = _HP_GRIDS.get(name)
    if grid is None:
        return base_model

    if len(y) < 30 or int(y.sum()) < 5:
        return base_model

    try:
        _log(f"[TUNE] RandomizedSearchCV for {name}")

        X_s, y_s, _, _ = _apply_smote(X, y, cat_indices)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=grid,
            n_iter=HP_N_ITER,
            scoring="average_precision",
            cv=TimeSeriesSplit(n_splits=4),
            random_state=42,
            n_jobs=-1,
            refit=False,
            verbose=0,
            error_score=0.0,
        )

        search.fit(X_s, y_s)
        best_params = search.best_params_

        _log(f"[TUNE] Best params for {name}", best_params)

        tuned = base_model.__class__(**{**base_model.get_params(), **best_params})
        return tuned

    except Exception as exc:
        print(f"[TUNE] Failed for {name}: {exc}")
        traceback.print_exc()
        return base_model

# =============================================================
# MODEL FACTORY
# =============================================================
def _build_models(imbalance_ratio: float) -> dict:
    models = {}

    # Logistic Regression — always available
    models["logistic"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )),
    ])

    # Random Forest — always available
    models["random_forest"] = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # XGBoost — optional
    if XGBOOST_AVAILABLE:
        # FIX: use_label_encoder removed (deprecated in XGBoost >= 1.6)
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=max(1.0, imbalance_ratio),
            eval_metric="aucpr",
            random_state=42,
            verbosity=0,
        )

    # LightGBM — optional
    if LIGHTGBM_AVAILABLE:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )

    # VotingClassifier — only when enough sub-estimators available
    if _VOTING_AVAILABLE and XGBOOST_AVAILABLE and LIGHTGBM_AVAILABLE:
        models["ensemble"] = VotingClassifier(
            estimators=[
                ("xgb", XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    scale_pos_weight=max(1.0, imbalance_ratio),
                    eval_metric="aucpr",
                    random_state=42,
                    verbosity=0,
                )),
                ("lgbm", LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    class_weight="balanced",
                    random_state=42,
                    verbose=-1,
                )),
                ("rf", RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )),
            ],
            voting="soft",
            n_jobs=-1,
        )

    return models

# =============================================================
# MODEL SELECTION
# =============================================================
def _select_best_model(X, y, models, cat_indices, band_name):
    tscv = TimeSeriesSplit(n_splits=4)
    cv_results = {}
    fold_importances_by_model = {name: [] for name in models}

    _log(f"[TRAIN] CV selection band={band_name} rows={len(X)}")

    tuned_models = {}
    for name, model in models.items():
        if name in ["logistic", "ensemble"]:
            tuned_models[name] = model
        else:
            tuned_models[name] = _tune_model(
                name=name, base_model=model, X=X, y=y, cat_indices=cat_indices,
            )

    for name, model in tuned_models.items():
        fold_f1s = []
        for fold_i, (tr_idx, ts_idx) in enumerate(tscv.split(X)):
            X_tr, X_ts = X.iloc[tr_idx], X.iloc[ts_idx]
            y_tr, y_ts = y.iloc[tr_idx], y.iloc[ts_idx]

            if len(y_tr.unique()) < 2:
                continue

            X_tr_s, y_tr_s, _, _ = _apply_smote(X_tr, y_tr, cat_indices)

            try:
                model.fit(X_tr_s, y_tr_s)
                probs = model.predict_proba(X_ts)[:, 1]
                preds = (probs >= THRESHOLD).astype(int)
                score = f1_score(y_ts, preds, zero_division=0)
                fold_f1s.append(score)

                fi = compute_feature_importance(model, list(X.columns))
                if fi:
                    fold_importances_by_model[name].append(fi)

            except Exception as e:
                print(f"[{name}] Fold {fold_i + 1} ERROR: {e}")

        mean_f1 = round(float(np.mean(fold_f1s)) if fold_f1s else 0.0, 4)
        cv_results[name] = mean_f1
        _log(f"[{name}] Mean CV F1", mean_f1)

    best_name  = max(cv_results, key=cv_results.get)
    best_model = tuned_models[best_name]

    _log(f"[TRAIN] Winner for {band_name}", {"model": best_name, "f1": cv_results[best_name]})

    X_full_s, y_full_s, _, _ = _apply_smote(X, y, cat_indices)
    best_model.fit(X_full_s, y_full_s)

    return (
        best_name,
        best_model,
        cv_results,
        fold_importances_by_model.get(best_name, []),
    )

# =============================================================
# MAIN TRAINING PIPELINE
# =============================================================
def train_and_save(df: pd.DataFrame) -> dict:
    results = {}

    # 1. Cohort filter
    df, cohort_stats = filter_eligible_cohort(df)
    _log("[TRAIN] cohort filter", cohort_stats)

    if len(df) < 10:
        return {"status": "error", "message": "Insufficient data after cohort filter."}

    # 2. Preprocess
    df = preprocess(df)

    target = "is_defaulter"
    if target not in df.columns:
        return {"status": "error", "message": "Target column missing after preprocess."}

    # 3. Sort
    sort_cols = []
    if "_temporal_key" in df.columns:
        sort_cols.append("_temporal_key")
    if "student_id" in df.columns:
        sort_cols.append("student_id")
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

    # 4. Split bands
    band_dfs = _split_by_grade_band(df, grade_col="grade_encoded")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PIPELINE_DIR, exist_ok=True)

    cat_feature_names = [
        "grade_encoded", "division_encoded", "grade_band",
        "payment_momentum_encoded", "fee_tier_encoded", "has_bus_fee",
    ] + ["paid_" + yr.replace("-", "_") for yr in ALL_YEAR_COLS]

    exclude_cols = {target, "payment_ratio_bin", "_temporal_key", "student_id"}

    # 5. Train per band
    for band_name, band_df in band_dfs.items():
        _log(f"[TRAIN] band={band_name}", {"rows": len(band_df)})

        if len(band_df) < 10:
            results[band_name] = {
                "status": "skipped",
                "reason": f"Only {len(band_df)} rows",
            }
            continue

        class_dist = band_df[target].value_counts().to_dict()

        if len(class_dist) < 2:
            results[band_name] = {
                "status": "skipped",
                "reason": "Single class",
                "class_distribution": class_dist,
            }
            continue

        n_neg = class_dist.get(0, 1)
        n_pos = class_dist.get(1, 1)
        imbalance_ratio = n_neg / max(n_pos, 1)

        feature_cols = [
            c for c in band_df.columns
            if c not in exclude_cols
            and pd.api.types.is_numeric_dtype(band_df[c])
        ]

        X = band_df[feature_cols].copy()
        y = band_df[target].copy()

        cat_indices = [i for i, c in enumerate(X.columns) if c in cat_feature_names]
        models = _build_models(imbalance_ratio)

        try:
            best_name, best_model, cv_scores, fold_importances = _select_best_model(
                X=X, y=y, models=models,
                cat_indices=cat_indices, band_name=band_name,
            )
        except Exception as e:
            traceback.print_exc()
            results[band_name] = {"status": "error", "message": str(e)}
            continue

        # Year boundary holdout
        holdout_metrics = {}
        phase5_report   = {}

        if "_temporal_key" in band_df.columns:
            max_key = band_df["_temporal_key"].max()
            min_key = band_df["_temporal_key"].min()

            if max_key > min_key:
                holdout_mask = band_df["_temporal_key"] == max_key
                X_holdout    = X.loc[band_df.index[holdout_mask]]
                y_holdout    = y.loc[band_df.index[holdout_mask]]
                df_meta      = band_df.loc[band_df.index[holdout_mask]].reset_index(drop=True)
                _log("[TRAIN] Year boundary holdout", {"key": int(max_key), "holdout_rows": int(holdout_mask.sum())})
            else:
                split_idx = int(len(X) * 0.80)
                X_holdout = X.iloc[split_idx:]
                y_holdout = y.iloc[split_idx:]
                df_meta   = band_df.iloc[split_idx:].reset_index(drop=True)
        else:
            split_idx = int(len(X) * 0.80)
            X_holdout = X.iloc[split_idx:]
            y_holdout = y.iloc[split_idx:]
            df_meta   = band_df.iloc[split_idx:].reset_index(drop=True)

        if len(y_holdout) > 0 and len(y_holdout.unique()) > 1:
            phase5_report = full_evaluation_report(
                model=best_model,
                X_holdout=X_holdout,
                y_holdout=y_holdout,
                df_meta=df_meta,
                fold_importances=fold_importances,
                band_name=band_name,
                calibrate=True,
                explain_sample_size=SHAP_EXPLAIN_SAMPLE,
            )
            holdout_metrics = phase5_report.get("metrics", {})

        # Persist
        model_path = os.path.join(MODEL_DIR,    f"{band_name}_model.pkl")
        pipe_path  = os.path.join(PIPELINE_DIR, f"{band_name}_preprocessor.pkl")

        preprocessor_meta = {
            "band":            band_name,
            "feature_cols":    list(X.columns),
            "cat_indices":     cat_indices,
            "threshold":       THRESHOLD,
            "best_model":      best_name,
            "hp_tuning":       HP_TUNING_ENABLED,
            "hp_n_iter":       HP_N_ITER,
        }

        # ==================================================
        # SAVE MODEL + PIPELINE
        # ==================================================
        os.makedirs(MODEL_DIR,    exist_ok=True)
        os.makedirs(PIPELINE_DIR, exist_ok=True)

        joblib.dump(best_model,        model_path)
        joblib.dump(preprocessor_meta, pipe_path)

        # ==================================================
        # VALIDATE — fail loudly if save was silently broken
        # ==================================================
        if not os.path.exists(model_path):
            raise RuntimeError(f"[TRAIN] Model file not written: {model_path}")
        if os.path.getsize(model_path) == 0:
            raise RuntimeError(f"[TRAIN] Model file is 0 bytes (corrupt save): {model_path}")
        if not os.path.exists(pipe_path):
            raise RuntimeError(f"[TRAIN] Pipeline file not written: {pipe_path}")
        if os.path.getsize(pipe_path) == 0:
            raise RuntimeError(f"[TRAIN] Pipeline file is 0 bytes (corrupt save): {pipe_path}")

        _log("[TRAIN] Saved + Validated", {
            "model":    f"{model_path} ({os.path.getsize(model_path):,} bytes)",
            "pipeline": f"{pipe_path} ({os.path.getsize(pipe_path):,} bytes)",
        })

        results[band_name] = {
            "status":             "success",
            "best_model":         best_name,
            "cv_f1_scores":       cv_scores,
            "holdout_metrics":    holdout_metrics,
            "phase5_report":      phase5_report,
            "rows_trained":       len(band_df),
            "class_distribution": class_dist,
            "imbalance_ratio":    round(imbalance_ratio, 2),
            "model_path":         model_path,
            "pipeline_path":      pipe_path,
            "feature_count":      len(feature_cols),
            "hp_tuning_enabled":  HP_TUNING_ENABLED,
        }

    return {"status": "success", "cohort_stats": cohort_stats, "bands": results}

# =============================================================
# MULTICLASS TRAINING
# =============================================================
def train_multiclass(df: pd.DataFrame) -> dict:
    results = {}

    df, cohort_stats = filter_eligible_cohort(df)
    if len(df) < 10:
        return {"status": "error", "message": "Insufficient data."}

    df = preprocess(df)
    target = "payment_ratio_bin"
    if target not in df.columns:
        return {"status": "error", "message": "payment_ratio_bin missing"}

    sort_cols = []
    if "_temporal_key" in df.columns:
        sort_cols.append("_temporal_key")
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

    exclude_cols = {"is_defaulter", "payment_ratio_bin", "_temporal_key", "student_id"}
    band_dfs = _split_by_grade_band(df, grade_col="grade_encoded")

    os.makedirs(MODEL_DIR, exist_ok=True)

    for band_name, band_df in band_dfs.items():
        if len(band_df) < 15:
            results[band_name] = {"status": "skipped", "reason": "too few rows"}
            continue

        class_dist = band_df[target].value_counts().to_dict()
        if len(class_dist) < 2:
            results[band_name] = {"status": "skipped", "reason": "single class", "class_distribution": class_dist}
            continue

        feature_cols = [
            c for c in band_df.columns
            if c not in exclude_cols
            and pd.api.types.is_numeric_dtype(band_df[c])
        ]
        X = band_df[feature_cols].copy()
        y = band_df[target].copy()

        if LIGHTGBM_AVAILABLE:
            mc_model = LGBMClassifier(
                n_estimators=200, learning_rate=0.05,
                class_weight="balanced", random_state=42, verbose=-1,
                objective="multiclass", num_class=3,
            )
        else:
            mc_model = RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )

        tscv = TimeSeriesSplit(n_splits=4)
        fold_accs = []

        for tr_idx, ts_idx in tscv.split(X):
            X_tr, X_ts = X.iloc[tr_idx], X.iloc[ts_idx]
            y_tr, y_ts = y.iloc[tr_idx], y.iloc[ts_idx]
            if len(y_tr.unique()) < 2:
                continue
            try:
                mc_model.fit(X_tr, y_tr)
                preds = mc_model.predict(X_ts)
                fold_accs.append(float((preds == y_ts).mean()))
            except Exception as e:
                print(f"[MULTICLASS] {band_name} {e}")

        mc_model.fit(X, y)

        mc_path = os.path.join(MODEL_DIR, f"{band_name}_multiclass_model.pkl")
        joblib.dump(mc_model, mc_path)

        results[band_name] = {
            "status":           "success",
            "model_path":       mc_path,
            "class_distribution": class_dist,
            "mean_cv_accuracy": round(float(np.mean(fold_accs)) if fold_accs else 0.0, 4),
            "target":           target,
            "class_labels":     {0: "on_time", 1: "partial_payer", 2: "full_defaulter"},
        }
        _log("[MULTICLASS] Saved", mc_path)

    return {"status": "success", "cohort_stats": cohort_stats, "bands": results}

# =============================================================
# TRAIN + REGISTER
# =============================================================
def train_save_and_register(
    df: pd.DataFrame,
    dataset_range: str = "",
    notes: str = "",
    auto_promote: bool = False,
) -> dict:
    from versioning.model_registry import register_training_run

    try:
        from database.supabase_client import supabase as sb
    except Exception:
        sb = None

    training_output = train_and_save(df)
    if training_output.get("status") != "success":
        return training_output

    registry_result = register_training_run(
        supabase_client=sb,
        training_output=training_output,
        dataset_range=dataset_range,
        notes=notes,
        auto_promote=auto_promote,
    )

    _log("[TRAIN] Registry updated", registry_result)
    training_output["registry"] = registry_result
    return training_output

# =============================================================
# CLI ENTRY POINT
# =============================================================
if __name__ == "__main__":

    DATA_PATH = os.environ.get(
        "TRAINING_DATA_PATH",
        os.path.join(BASE_DIR, "data", "Fees Defaulter Report - Overall.xlsx"),
    )

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Training data not found: {DATA_PATH}")
        print("[ERROR] Set TRAINING_DATA_PATH env var or place the Excel file at:")
        print(f"        {DATA_PATH}")
        sys.exit(1)

    _log("[TRAIN] Loading data", DATA_PATH)
    df_raw = pd.read_excel(DATA_PATH)
    _log("[TRAIN] Raw shape", df_raw.shape)

    output = train_and_save(df_raw)

    _log("[TRAIN] COMPLETE", {k: v for k, v in output.items() if k != "bands"})

    all_ok = True
    for band, res in output.get("bands", {}).items():
        status = res.get("status", "unknown")
        _log(f"[TRAIN] Band={band}", {k: v for k, v in res.items() if k not in ("holdout_metrics", "phase5_report")})
        if status != "success":
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("SUCCESS — All models trained and saved.")
        print(f"  models/    → {MODEL_DIR}")
        print(f"  pipelines/ → {PIPELINE_DIR}")
    else:
        print("PARTIAL — Some bands were skipped or errored. Check logs above.")
    print("Run: uvicorn app:app --reload")
    print("=" * 60 + "\n")
