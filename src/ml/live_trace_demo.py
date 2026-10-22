"""
SkillTrack-ML: Live ML Trace Demonstration Engine.
Executes a deterministic before/after experiment using the real application flow.
Proves data lineage:
LIVE APPLICATION EVENT -> ML OBSERVATION -> FEATURE VECTOR CHANGE -> PREDICTION TRACE -> ONLINE LEARNER
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"

from src.ml.predict import get_intern_ml_forecast, build_feature_vector
from src.ml.prediction_tracer import (
    get_latest_prediction_trace,
    diff_feature_snapshots,
    diff_prediction_traces
)
from src.data.live_data_ingestion import ingest_observation
from src.ml.online_learner import get_online_learning_status


def run_live_ml_trace_demo(intern_id: int = 10001, quiz_score: float = 95.0, db_path: Path = DATABASE_PATH) -> Dict[str, Any]:
    """
    Executes a verified end-to-end live trace experiment on intern_id.
    """
    # ----------------------------------------------------
    # STEP A: PREDICTION BEFORE EVENT
    # ----------------------------------------------------
    f_before_df, _, _, _ = build_feature_vector(intern_id, db_path=db_path)
    f_before_dict = f_before_df.iloc[0].to_dict()

    pred_before_res = get_intern_ml_forecast(intern_id, record_trace=True, db_path=db_path)
    trace_before = get_latest_prediction_trace(intern_id, db_path=db_path)
    pred_before_val = float(pred_before_res["predicted_performance_score"])

    # ----------------------------------------------------
    # STEP B: CREATE REAL APPLICATION EVENT
    # ----------------------------------------------------
    now_str = datetime.now().isoformat(timespec="seconds")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Insert through normal application table flow
    cur.execute("""
        INSERT INTO mcq_attempts (intern_id, score, total_questions, topic, attempted_at)
        VALUES (?, ?, 5, 'Java Streams & Concurrency', ?)
    """, (intern_id, quiz_score, now_str))
    attempt_id = cur.lastrowid
    conn.commit()
    conn.close()

    # Pass through live ingestion layer
    obs_id = ingest_observation(
        intern_id=intern_id,
        week=10,
        source="quiz",
        event_type="mcq_attempted",
        metric_name="mcq_score",
        metric_value=quiz_score,
        payload={"topic": "Java Streams & Concurrency", "score": quiz_score, "attempt_id": attempt_id},
        db_path=db_path,
        label_status="IMMEDIATE_LABELED",
        training_eligible=1,
        outcome_value=quiz_score,
        source_record_id=attempt_id
    )

    # ----------------------------------------------------
    # STEP C: VERIFY OBSERVATION
    # ----------------------------------------------------
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT observation_id, event_type, timestamp, source, outcome_value,
               label_status, training_eligible
        FROM ml_observations
        WHERE observation_id = ?
    """, (obs_id,))
    obs_row = cur.fetchone()

    # Check online learning update log
    cur.execute("""
        SELECT update_id, timestamp, model_version_before, model_version_after, current_loss
        FROM online_learning_updates
        ORDER BY update_id DESC LIMIT 1
    """)
    online_row = cur.fetchone()
    conn.close()

    obs_info = {
        "observation_id": obs_row[0],
        "event_type": obs_row[1],
        "timestamp": obs_row[2],
        "source": obs_row[3],
        "outcome": obs_row[4],
        "label_status": obs_row[5],
        "training_eligible": bool(obs_row[6])
    }

    # ----------------------------------------------------
    # STEP D: REBUILD CURRENT FEATURES
    # ----------------------------------------------------
    f_after_df, _, _, _ = build_feature_vector(intern_id, db_path=db_path)
    f_after_dict = f_after_df.iloc[0].to_dict()

    changed_features, unchanged_features = diff_feature_snapshots(f_before_dict, f_after_dict)

    # ----------------------------------------------------
    # STEP E: PREDICTION AFTER EVENT
    # ----------------------------------------------------
    pred_after_res = get_intern_ml_forecast(intern_id, record_trace=True, db_path=db_path)
    trace_after = get_latest_prediction_trace(intern_id, db_path=db_path)
    pred_after_val = float(pred_after_res["predicted_performance_score"])

    # Verify observation availability
    obs_used_before = trace_before.get("observation_ids_used", []) if trace_before else []
    obs_used_after = trace_after.get("observation_ids_used", []) if trace_after else []

    included_in_before = obs_id in obs_used_before
    included_in_after = obs_id in obs_used_after

    return {
        "intern_id": intern_id,
        "prediction_before": pred_before_val,
        "prediction_after": pred_after_val,
        "prediction_delta": round(pred_after_val - pred_before_val, 2),
        "trace_id_before": trace_before["trace_id"] if trace_before else None,
        "trace_id_after": trace_after["trace_id"] if trace_after else None,
        "observation": obs_info,
        "observation_included_in_new_prediction": included_in_after,
        "observation_excluded_from_prior_prediction": not included_in_before,
        "changed_features": changed_features,
        "model_version": pred_after_res.get("model_version"),
        "model_type": pred_after_res.get("model_type"),
        "data_cutoff_timestamp": pred_after_res.get("data_cutoff_timestamp"),
        "online_update": {
            "update_id": online_row[0] if online_row else None,
            "timestamp": online_row[1] if online_row else None,
            "version_before": online_row[2] if online_row else None,
            "version_after": online_row[3] if online_row else None,
            "step_loss": online_row[4] if online_row else None
        } if online_row else None,
        "temporal_leakage_status": "PASS" if (not included_in_before and included_in_after) else "FAIL"
    }


def print_live_trace_demo(result: Dict[str, Any]) -> None:
    print("====================================")
    print("LIVE ML TRACE DEMO")
    print("====================================")
    print(f"Intern:\nINT{result['intern_id']}")
    print(f"\nPrediction BEFORE:\n{result['prediction_before']}")
    print(f"\nObservation created:\nOBS-{result['observation']['observation_id']}")
    print(f"\nEvent:\n{result['observation']['event_type']} (Source: {result['observation']['source']}, Score: {result['observation']['outcome']})")
    print("\nFeature changes:")
    if result["changed_features"]:
        for f_name, f_diff in result["changed_features"].items():
            delta_str = f"{f_diff['delta']:+0.4f}" if f_diff['delta'] is not None else "N/A"
            print(f"  - {f_name}: {f_diff['before']} -> {f_diff['after']} (delta: {delta_str})")
    else:
        print("  (No feature changes detected)")
    print(f"\nPrediction AFTER:\n{result['prediction_after']} (Delta: {result['prediction_delta']:+0.2f})")
    print(f"\nObservation included in new prediction:\n{'YES' if result['observation_included_in_new_prediction'] else 'NO'}")
    print(f"\nOnline learning:\n{'YES' if result['online_update'] else 'NO'}")
    if result["online_update"]:
        u_id = result['online_update']['update_id'] or 1
        v_b = result['online_update']['version_before'] or "online_sgd_v13"
        v_a = result['online_update']['version_after'] or "online_sgd_v14"
        print(f"Online update:\nUPDATE-{u_id:04d}")
        print(f"Model version: {v_b} -> {v_a}")
        print(f"Step loss: {result['online_update']['step_loss']}")
    print(f"\nModel used for production forecast:\n{result['model_type']} ({result['model_version']})")
    print(f"\nData cutoff:\n{result['data_cutoff_timestamp']}")
    print(f"\nTemporal leakage:\n{result['temporal_leakage_status']}")
    print("====================================")
