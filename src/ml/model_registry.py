"""
SkillTrack-ML: Model Registry & Quality Gate Management
Handles model version metadata, evaluation tracking, promotion rules,
and retraining threshold checks.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"
MIN_NEW_LABELED_SAMPLES_FOR_RETRAINING = 50


def get_db_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def get_active_production_model(db_path: Path = DATABASE_PATH) -> Optional[Dict[str, Any]]:
    """
    Retrieves the currently active production model from SQLite.
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT model_id, model_version, model_type, algorithm_cls, algorithm_reg,
               dataset_version, trained_at, validation_metrics, test_metrics,
               feature_version, artifact_path_cls, artifact_path_reg, status, promotion_reason
        FROM model_registry
        WHERE status = 'production'
        ORDER BY model_id DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    cols = [
        "model_id", "model_version", "model_type", "algorithm_cls", "algorithm_reg",
        "dataset_version", "trained_at", "validation_metrics", "test_metrics",
        "feature_version", "artifact_path_cls", "artifact_path_reg", "status", "promotion_reason"
    ]
    res = dict(zip(cols, row))
    try:
        res["validation_metrics"] = json.loads(res["validation_metrics"])
    except Exception:
        pass
    try:
        res["test_metrics"] = json.loads(res["test_metrics"]) if res["test_metrics"] else {}
    except Exception:
        pass

    return res


def check_retraining_eligibility(
    db_path: Path = DATABASE_PATH,
    min_new_samples: int = MIN_NEW_LABELED_SAMPLES_FOR_RETRAINING
) -> Tuple[bool, int, str]:
    """
    Checks whether enough new observations have accumulated to justify controlled retraining.
    Retraining must never be triggered on every user click.
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    # Get timestamp of last production model training
    cur.execute("SELECT trained_at FROM model_registry WHERE status = 'production' ORDER BY model_id DESC LIMIT 1")
    row = cur.fetchone()
    last_trained_at = row[0] if row else "2000-01-01T00:00:00"

    # Count observations added since last training
    cur.execute("SELECT count(*) FROM ml_observations WHERE created_at > ?", (last_trained_at,))
    new_obs_count = cur.fetchone()[0]
    conn.close()

    is_eligible = new_obs_count >= min_new_samples
    reason = (
        f"Eligible: {new_obs_count} new observations accumulated (threshold = {min_new_samples})"
        if is_eligible
        else f"Ineligible: {new_obs_count}/{min_new_samples} new observations needed before controlled retraining"
    )
    return is_eligible, new_obs_count, reason


def evaluate_quality_gate(
    val_macro_f1: float,
    val_ni_recall: float,
    val_rmse: float,
    val_r2: float,
    current_prod_metrics: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """
    Applies quality gate criteria to determine if a candidate model qualifies for promotion:
    - Classification Macro F1 >= 0.65
    - Needs Improvement Recall >= 0.70 (critical early-intervention safety rule)
    - Regression RMSE <= 10.0
    - Regression R2 >= 0.50
    """
    reasons = []
    passed = True

    if val_macro_f1 < 0.65:
        passed = False
        reasons.append(f"Macro F1 ({val_macro_f1:.4f}) is below minimum threshold (0.65)")

    if val_ni_recall < 0.70:
        passed = False
        reasons.append(f"Needs Improvement Recall ({val_ni_recall:.4f}) is below safety threshold (0.70)")

    if val_rmse > 10.0:
        passed = False
        reasons.append(f"RMSE ({val_rmse:.4f}) exceeds maximum threshold (10.0)")

    if val_r2 < 0.50:
        passed = False
        reasons.append(f"R^2 ({val_r2:.4f}) is below minimum acceptable fit (0.50)")

    if passed:
        return True, "Passed all quality gates: Macro F1 >= 0.65, NI Recall >= 0.70, RMSE <= 10.0, R2 >= 0.50"
    else:
        return False, "Quality gate rejected: " + "; ".join(reasons)


def register_trained_model(
    model_version: str,
    algorithm_cls: str,
    algorithm_reg: str,
    dataset_version: str,
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    artifact_path_cls: str,
    artifact_path_reg: str,
    feature_version: str = "43-features-v1",
    db_path: Path = DATABASE_PATH,
    auto_promote: bool = True
) -> Dict[str, Any]:
    """
    Registers the newly trained model in SQLite, evaluates quality gates,
    and updates promotion status accordingly.
    """
    current_prod = get_active_production_model(db_path)
    val_macro_f1 = val_metrics.get("classification", {}).get("macro_f1", 0.0)
    val_ni_recall = val_metrics.get("classification", {}).get("needs_improvement_recall", 0.0)
    val_rmse = val_metrics.get("regression", {}).get("rmse", 999.0)
    val_r2 = val_metrics.get("regression", {}).get("r2", 0.0)

    passed_gate, gate_reason = evaluate_quality_gate(
        val_macro_f1=val_macro_f1,
        val_ni_recall=val_ni_recall,
        val_rmse=val_rmse,
        val_r2=val_r2,
        current_prod_metrics=current_prod
    )

    status = "production" if (passed_gate and auto_promote) else ("candidate" if passed_gate else "rejected")
    now_str = datetime.now().isoformat(timespec="seconds")

    conn = get_db_connection(db_path)
    cur = conn.cursor()

    if status == "production":
        # Archive previous production model
        cur.execute("UPDATE model_registry SET status = 'archived' WHERE status = 'production'")

    cur.execute("""
        INSERT INTO model_registry (
            model_version, model_type, algorithm_cls, algorithm_reg,
            dataset_version, trained_at, validation_metrics, test_metrics,
            feature_version, artifact_path_cls, artifact_path_reg, status, promotion_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        model_version,
        "dual_pipeline",
        algorithm_cls,
        algorithm_reg,
        dataset_version,
        now_str,
        json.dumps(val_metrics),
        json.dumps(test_metrics),
        feature_version,
        artifact_path_cls,
        artifact_path_reg,
        status,
        gate_reason
    ))
    conn.commit()
    conn.close()

    return {
        "model_version": model_version,
        "status": status,
        "passed_gate": passed_gate,
        "gate_reason": gate_reason
    }


def get_all_registered_models(db_path: Path = DATABASE_PATH) -> List[Dict[str, Any]]:
    """
    Retrieves all registered models in the registry ordered by registration time.
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT model_id, model_version, model_type, algorithm_cls, algorithm_reg,
               dataset_version, trained_at, validation_metrics, test_metrics,
               feature_version, artifact_path_cls, artifact_path_reg, status, promotion_reason
        FROM model_registry
        ORDER BY model_id DESC
    """)
    rows = cur.fetchall()
    conn.close()

    cols = [
        "model_id", "model_version", "model_type", "algorithm_cls", "algorithm_reg",
        "dataset_version", "trained_at", "validation_metrics", "test_metrics",
        "feature_version", "artifact_path_cls", "artifact_path_reg", "status", "promotion_reason"
    ]
    results = []
    for r in rows:
        d = dict(zip(cols, r))
        try:
            d["validation_metrics"] = json.loads(d["validation_metrics"]) if d["validation_metrics"] else {}
        except Exception:
            pass
        try:
            d["test_metrics"] = json.loads(d["test_metrics"]) if d["test_metrics"] else {}
        except Exception:
            pass
        results.append(d)
    return results
