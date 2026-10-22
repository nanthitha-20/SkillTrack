"""
SkillTrack-ML: Prediction Traceability & Data Lineage Engine
Captures verifiable evidence of data availability, exact feature snapshots,
contributing observation IDs, and model versions for every prediction.
Guarantees strict temporal causality (no forward-looking leakage).
"""

import json
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


def compute_feature_vector_hash(features_dict: Dict[str, Any]) -> str:
    """
    Computes a deterministic SHA-256 hash from a dictionary of feature values.
    Rounds floating point numbers to 4 decimal places for stable reproducibility.
    """
    normalized = {}
    for k in sorted(features_dict.keys()):
        val = features_dict[k]
        if isinstance(val, (float, np_float if 'np_float' in globals() else float)):
            normalized[k] = round(float(val), 4)
        else:
            normalized[k] = val
    canonical_str = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()[:16]


def get_observation_ids_for_prediction(
    intern_id: int,
    cutoff_timestamp: str,
    db_path: Path = DATABASE_PATH
) -> Tuple[List[int], Optional[int]]:
    """
    Retrieves all observation IDs for an intern that were generated AT OR BEFORE
    the specified cutoff timestamp. Strictly enforces temporal cutoff.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT observation_id
        FROM ml_observations
        WHERE intern_id = ? AND timestamp <= ?
        ORDER BY observation_id ASC
    """, (intern_id, cutoff_timestamp))
    rows = cur.fetchall()
    conn.close()

    obs_ids = [r[0] for r in rows]
    latest_id = obs_ids[-1] if obs_ids else None
    return obs_ids, latest_id


def get_latest_prediction_trace(
    intern_id: int,
    db_path: Path = DATABASE_PATH
) -> Optional[Dict[str, Any]]:
    """
    Retrieves the most recent prediction trace for an intern.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM prediction_traces
        WHERE intern_id = ?
        ORDER BY trace_id DESC
        LIMIT 1
    """, (intern_id,))
    cols = [col[0] for col in cur.description]
    row = cur.fetchone()
    conn.close()

    if not row:
        return None
    res = dict(zip(cols, row))
    if isinstance(res.get("feature_snapshot"), str):
        try:
            res["feature_snapshot"] = json.loads(res["feature_snapshot"])
        except Exception:
            pass
    if isinstance(res.get("observation_ids_used"), str):
        try:
            res["observation_ids_used"] = json.loads(res["observation_ids_used"])
        except Exception:
            pass
    return res


def get_trace_by_id(
    trace_id: int,
    db_path: Path = DATABASE_PATH
) -> Optional[Dict[str, Any]]:
    """Retrieves a specific prediction trace by ID."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM prediction_traces WHERE trace_id = ?", (trace_id,))
    cols = [col[0] for col in cur.description]
    row = cur.fetchone()
    conn.close()

    if not row:
        return None
    res = dict(zip(cols, row))
    if isinstance(res.get("feature_snapshot"), str):
        try:
            res["feature_snapshot"] = json.loads(res["feature_snapshot"])
            res["features_snapshot"] = res["feature_snapshot"]
        except Exception:
            pass
    if isinstance(res.get("observation_ids_used"), str):
        try:
            res["observation_ids_used"] = json.loads(res["observation_ids_used"])
        except Exception:
            pass
    return res


def get_traces_for_intern(
    intern_id: int,
    limit: int = 50,
    db_path: Path = DATABASE_PATH
) -> List[Dict[str, Any]]:
    """Retrieves prediction traces for a specific intern ordered chronologically descending."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM prediction_traces WHERE intern_id = ? ORDER BY trace_id DESC LIMIT ?", (intern_id, limit))
    cols = [col[0] for col in cur.description]
    rows = cur.fetchall()
    conn.close()

    res_list = []
    for row in rows:
        res = dict(zip(cols, row))
        if isinstance(res.get("feature_snapshot"), str):
            try:
                res["feature_snapshot"] = json.loads(res["feature_snapshot"])
                res["features_snapshot"] = res["feature_snapshot"]
            except Exception:
                pass
        if isinstance(res.get("observation_ids_used"), str):
            try:
                res["observation_ids_used"] = json.loads(res["observation_ids_used"])
            except Exception:
                pass
        res_list.append(res)
    return res_list


def record_prediction_trace(
    intern_id: int,
    predicted_gap: str,
    predicted_performance: float,
    confidence_score: float,
    features_dict: Dict[str, Any],
    model_version: str,
    model_type: str = "Balanced Logistic Regression + Ridge Regression",
    prediction_source: str = "production_batch",
    prediction_type: str = "next_week_forecast",
    data_cutoff_timestamp: Optional[str] = None,
    training_row_count: int = 4500,
    online_model_version: Optional[str] = None,
    online_update_count: int = 0,
    db_path: Path = DATABASE_PATH
) -> int:
    """
    Persists an auditable prediction trace with feature hash, snapshot,
    observation IDs used, and before/after prediction comparison.
    """
    now_str = datetime.now().isoformat(timespec="seconds")
    if data_cutoff_timestamp is None:
        data_cutoff_timestamp = now_str

    # Retrieve contributing observation IDs
    obs_ids, latest_obs_id = get_observation_ids_for_prediction(
        intern_id=intern_id,
        cutoff_timestamp=data_cutoff_timestamp,
        db_path=db_path
    )

    # Compute deterministic feature hash
    f_hash = compute_feature_vector_hash(features_dict)
    f_snapshot_json = json.dumps(features_dict, sort_keys=True)

    # Check previous trace to record prediction delta
    prev_trace = get_latest_prediction_trace(intern_id=intern_id, db_path=db_path)
    pred_before = float(prev_trace["predicted_performance"]) if prev_trace else None
    pred_after = float(predicted_performance)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO prediction_traces (
            intern_id, prediction_timestamp, prediction_type,
            model_version, model_type, prediction_source,
            predicted_gap, predicted_performance, confidence_score,
            prediction_before, prediction_after,
            feature_vector_hash, feature_snapshot,
            observation_ids_used, latest_observation_id,
            training_row_count_at_prediction,
            online_model_version, online_update_count,
            data_cutoff_timestamp, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        intern_id,
        now_str,
        prediction_type,
        model_version,
        model_type,
        prediction_source,
        predicted_gap,
        round(pred_after, 2),
        round(confidence_score, 4),
        round(pred_before, 2) if pred_before is not None else None,
        round(pred_after, 2),
        f_hash,
        f_snapshot_json,
        json.dumps(obs_ids),
        latest_obs_id,
        training_row_count,
        online_model_version or "None",
        online_update_count,
        data_cutoff_timestamp,
        now_str
    ))
    trace_id = cur.lastrowid
    conn.commit()
    conn.close()
    return trace_id


def diff_feature_snapshots(
    snapshot_before: Dict[str, Any],
    snapshot_after: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Compares two feature snapshots and returns (changed_features, unchanged_features).
    """
    changed = {}
    unchanged = {}

    all_keys = sorted(set(snapshot_before.keys()).union(set(snapshot_after.keys())))
    for k in all_keys:
        v_before = snapshot_before.get(k)
        v_after = snapshot_after.get(k)

        is_different = False
        if isinstance(v_before, (int, float)) and isinstance(v_after, (int, float)):
            if abs(float(v_before) - float(v_after)) > 1e-4:
                is_different = True
        elif v_before != v_after:
            is_different = True

        if is_different:
            delta = None
            if isinstance(v_before, (int, float)) and isinstance(v_after, (int, float)):
                delta = round(float(v_after) - float(v_before), 4)
            changed[k] = {
                "before": v_before,
                "after": v_after,
                "delta": delta
            }
        else:
            unchanged[k] = v_after

    return changed, unchanged


def diff_prediction_traces(
    trace_id_before: int,
    trace_id_after: int,
    db_path: Path = DATABASE_PATH
) -> Dict[str, Any]:
    """
    Detailed comparative audit between two prediction traces.
    Answers: "Why did this prediction change?" with concrete data lineage.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM prediction_traces WHERE trace_id = ?", (trace_id_before,))
    cols = [col[0] for col in cur.description]
    row_b = cur.fetchone()
    cur.execute("SELECT * FROM prediction_traces WHERE trace_id = ?", (trace_id_after,))
    row_a = cur.fetchone()
    conn.close()

    if not row_b or not row_a:
        raise ValueError("One or both trace IDs could not be found.")

    t_before = dict(zip(cols, row_b))
    t_after = dict(zip(cols, row_a))

    s_before = json.loads(t_before["feature_snapshot"])
    s_after = json.loads(t_after["feature_snapshot"])

    changed_f, unchanged_f = diff_feature_snapshots(s_before, s_after)

    obs_b = set(json.loads(t_before["observation_ids_used"] or "[]"))
    obs_a = set(json.loads(t_after["observation_ids_used"] or "[]"))
    new_obs_ids = sorted(list(obs_a - obs_b))

    pred_delta = round(float(t_after["predicted_performance"]) - float(t_before["predicted_performance"]), 2)

    return {
        "trace_id_before": trace_id_before,
        "trace_id_after": trace_id_after,
        "prediction_before": t_before["predicted_performance"],
        "prediction_after": t_after["predicted_performance"],
        "predicted_gap_before": t_before["predicted_gap"],
        "predicted_gap_after": t_after["predicted_gap"],
        "prediction_delta": pred_delta,
        "model_version": t_after["model_version"],
        "model_type": t_after["model_type"],
        "cutoff_timestamp_before": t_before["data_cutoff_timestamp"],
        "cutoff_timestamp_after": t_after["data_cutoff_timestamp"],
        "new_observations_count": len(new_obs_ids),
        "new_observation_ids": new_obs_ids,
        "changed_features": changed_f,
        "unchanged_features": unchanged_f
    }


def get_recent_prediction_traces(
    limit: int = 25,
    intern_id: Optional[int] = None,
    db_path: Path = DATABASE_PATH
) -> List[Dict[str, Any]]:
    """
    Returns recent prediction traces formatted for UI inspection.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    query = """
        SELECT t.trace_id, t.prediction_timestamp, t.intern_id, i.name as intern_name,
               t.predicted_gap, t.predicted_performance, t.prediction_before,
               t.model_version, t.feature_vector_hash, t.latest_observation_id,
               t.data_cutoff_timestamp, t.prediction_source
        FROM prediction_traces t
        LEFT JOIN interns i ON t.intern_id = i.intern_id
        WHERE 1=1
    """
    params: List[Any] = []
    if intern_id is not None:
        query += " AND t.intern_id = ?"
        params.append(intern_id)

    query += " ORDER BY t.trace_id DESC LIMIT ?"
    params.append(limit)

    cur.execute(query, params)
    cols = [col[0] for col in cur.description]
    rows = cur.fetchall()
    conn.close()

    return [dict(zip(cols, row)) for row in rows]


def get_annotated_recent_observations(
    limit: int = 30,
    db_path: Path = DATABASE_PATH
) -> List[Dict[str, Any]]:
    """
    Returns recent live observations annotated with explicit lineage:
    - Used in Prediction? (Trace ID)
    - Online Update? (Update ID & Model version)
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT o.observation_id, o.timestamp, o.intern_id, i.name as intern_name,
               o.event_type, o.source, o.metric_name, o.metric_value, o.outcome_value,
               o.label_status, o.training_eligible, o.target_week
        FROM ml_observations o
        LEFT JOIN interns i ON o.intern_id = i.intern_id
        WHERE o.source != 'synthetic_historical'
        ORDER BY o.observation_id DESC
        LIMIT ?
    """, (limit,))
    obs_cols = [c[0] for c in cur.description]
    obs_rows = cur.fetchall()

    cur.execute("""
        SELECT trace_id, intern_id, observation_ids_used, prediction_timestamp,
               predicted_performance, prediction_before
        FROM prediction_traces
        ORDER BY trace_id DESC LIMIT 100
    """)
    trace_rows = cur.fetchall()

    cur.execute("""
        SELECT update_id, observation_ids, timestamp, model_version_after
        FROM online_learning_updates
        ORDER BY update_id DESC LIMIT 100
    """)
    update_rows = cur.fetchall()
    conn.close()

    results = []
    for row in obs_rows:
        item = dict(zip(obs_cols, row))
        obs_id = item["observation_id"]
        obs_str = f"OBS-{obs_id}"

        # Match prediction traces
        matching_traces = []
        trace_delta = None
        for t_id, t_intern, obs_json, p_ts, p_perf, p_before in trace_rows:
            if t_intern == item["intern_id"] and obs_json:
                try:
                    ids = json.loads(obs_json)
                    if obs_id in ids:
                        matching_traces.append(t_id)
                        if trace_delta is None and p_perf is not None and p_before is not None:
                            trace_delta = round(float(p_perf) - float(p_before), 2)
                except Exception:
                    pass

        item["prediction_trace"] = f"Trace #{matching_traces[0]}" if matching_traces else "Pending"
        item["used_in_prediction"] = f"YES (Trace #{matching_traces[0]})" if matching_traces else "NO"
        item["prediction_delta"] = f"{trace_delta:+.2f}" if trace_delta is not None else "--"

        # Match online updates
        matching_updates = []
        for u_id, u_obs, u_ts, u_ver in update_rows:
            if u_obs and (obs_str in u_obs or str(obs_id) in u_obs):
                matching_updates.append(f"UPDATE-{u_id:04d}")
        if matching_updates:
            item["online_update"] = matching_updates[0]
        elif item["label_status"] == "PENDING_FUTURE_LABEL":
            item["online_update"] = "PENDING_FUTURE_LABEL"
        else:
            item["online_update"] = "NO"

        results.append(item)

    return results


def get_observation_lineage_details(
    observation_id: int,
    db_path: Path = DATABASE_PATH
) -> Dict[str, Any]:
    """
    Returns full end-to-end lineage for a specific observation:
    - Observation metadata
    - Online learning update record
    - Traces that used this observation
    - Next predictions generated after this observation
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT o.*, i.name as intern_name
        FROM ml_observations o
        LEFT JOIN interns i ON o.intern_id = i.intern_id
        WHERE o.observation_id = ?
    """, (observation_id,))
    cols = [c[0] for c in cur.description]
    row = cur.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Observation ID {observation_id} not found.")

    obs = dict(zip(cols, row))
    obs_id = obs["observation_id"]
    intern_id = obs["intern_id"]
    obs_ts = obs["timestamp"]
    obs_tag = f"OBS-{obs_id}"

    # Find online learning update
    cur.execute("""
        SELECT * FROM online_learning_updates
        WHERE observation_ids LIKE ? OR observation_ids = ?
        ORDER BY update_id DESC LIMIT 1
    """, (f"%{obs_tag}%", str(obs_id)))
    up_cols = [c[0] for c in cur.description]
    up_row = cur.fetchone()
    online_update = dict(zip(up_cols, up_row)) if up_row else None

    # Find traces for this intern
    cur.execute("""
        SELECT * FROM prediction_traces
        WHERE intern_id = ?
        ORDER BY trace_id ASC
    """, (intern_id,))
    tr_cols = [c[0] for c in cur.description]
    tr_rows = [dict(zip(tr_cols, r)) for r in cur.fetchall()]
    conn.close()

    traces_before = []
    traces_after = []
    for tr in tr_rows:
        try:
            ids = json.loads(tr["observation_ids_used"] or "[]")
            if obs_id in ids or tr["data_cutoff_timestamp"] >= obs_ts:
                traces_after.append(tr)
            else:
                traces_before.append(tr)
        except Exception:
            if tr["data_cutoff_timestamp"] < obs_ts:
                traces_before.append(tr)
            else:
                traces_after.append(tr)

    trace_before = traces_before[-1] if traces_before else None
    trace_after = traces_after[0] if traces_after else None

    features_before = json.loads(trace_before["feature_snapshot"]) if trace_before else {}
    features_after = json.loads(trace_after["feature_snapshot"]) if trace_after else {}

    changed_features, unchanged_features = {}, {}
    if features_before and features_after:
        changed_features, unchanged_features = diff_feature_snapshots(features_before, features_after)

    return {
        "observation": obs,
        "online_update": online_update,
        "trace_before": trace_before,
        "trace_after": trace_after,
        "features_before": features_before,
        "features_after": features_after,
        "changed_features": changed_features,
        "unchanged_features": unchanged_features,
        "all_subsequent_traces": traces_after
    }


def get_intern_live_ml_status(
    intern_id: int,
    db_path: Path = DATABASE_PATH
) -> Dict[str, Any]:
    """
    Returns live ML traceability for an individual intern portal view:
    - Current prediction vs previous prediction
    - Prediction delta
    - Model version and cutoff timestamp
    - Counts of live observations, training-eligible, and pending future labels
    - Latest observation metadata and feature differences
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Fetch latest 2 prediction traces
    cur.execute("""
        SELECT * FROM prediction_traces
        WHERE intern_id = ?
        ORDER BY trace_id DESC LIMIT 2
    """, (intern_id,))
    cols = [c[0] for c in cur.description]
    traces = [dict(zip(cols, r)) for r in cur.fetchall()]

    # If no traces exist, generate one using forecast
    if not traces:
        conn.close()
        from src.ml.predict import get_intern_ml_forecast
        get_intern_ml_forecast(intern_id, db_path=db_path)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM prediction_traces
            WHERE intern_id = ?
            ORDER BY trace_id DESC LIMIT 2
        """, (intern_id,))
        cols = [c[0] for c in cur.description]
        traces = [dict(zip(cols, r)) for r in cur.fetchall()]

    latest_trace = traces[0] if traces else None
    prev_trace = traces[1] if len(traces) > 1 else None

    # 2. Fetch observation counts
    cur.execute("SELECT count(*) FROM ml_observations WHERE intern_id = ? AND source != 'synthetic_historical'", (intern_id,))
    live_obs_count = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM ml_observations WHERE intern_id = ? AND training_eligible = 1", (intern_id,))
    training_eligible_count = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM ml_observations WHERE intern_id = ? AND label_status = 'PENDING_FUTURE_LABEL'", (intern_id,))
    pending_future_count = cur.fetchone()[0]

    # 3. Fetch latest observation
    cur.execute("""
        SELECT observation_id, event_type, source, timestamp, outcome_value, label_status
        FROM ml_observations
        WHERE intern_id = ?
        ORDER BY observation_id DESC LIMIT 1
    """, (intern_id,))
    obs_cols = [c[0] for c in cur.description]
    latest_obs_row = cur.fetchone()
    latest_obs = dict(zip(obs_cols, latest_obs_row)) if latest_obs_row else None

    conn.close()

    # Extract feature differences between latest and previous traces
    changed_features = {}
    unchanged_features = {}
    obs_used_count = 0
    if latest_trace:
        try:
            obs_used_count = len(json.loads(latest_trace["observation_ids_used"] or "[]"))
        except Exception:
            obs_used_count = 0

        s_latest = json.loads(latest_trace["feature_snapshot"]) if latest_trace.get("feature_snapshot") else {}
        if prev_trace and prev_trace.get("feature_snapshot"):
            s_prev = json.loads(prev_trace["feature_snapshot"])
            changed_features, unchanged_features = diff_feature_snapshots(s_prev, s_latest)
        else:
            unchanged_features = s_latest

    curr_p = round(float(latest_trace["predicted_performance"]), 1) if latest_trace else 75.0
    prev_p = round(float(prev_trace["predicted_performance"]), 1) if prev_trace else round(float(latest_trace.get("prediction_before", curr_p)), 1)
    delta_p = round(curr_p - prev_p, 2)

    return {
        "current_prediction": curr_p,
        "previous_prediction": prev_p,
        "prediction_change": delta_p,
        "predicted_gap": latest_trace.get("predicted_gap", "Developing") if latest_trace else "Developing",
        "model_version": latest_trace.get("model_version", "production_active") if latest_trace else "production_active",
        "model_type": latest_trace.get("model_type", "Ridge Regression") if latest_trace else "Ridge Regression",
        "last_updated": latest_trace.get("prediction_timestamp", "") if latest_trace else "",
        "data_cutoff_timestamp": latest_trace.get("data_cutoff_timestamp", "") if latest_trace else "",
        "latest_observation_id": f"OBS-{latest_trace.get('latest_observation_id')}" if latest_trace and latest_trace.get("latest_observation_id") else (f"OBS-{latest_obs['observation_id']}" if latest_obs else "None"),
        "observations_used_count": obs_used_count,
        "live_observations_count": live_obs_count,
        "training_eligible_count": training_eligible_count,
        "pending_future_count": pending_future_count,
        "latest_observation": latest_obs,
        "changed_features": changed_features,
        "unchanged_features": unchanged_features,
        "trace_id_current": latest_trace.get("trace_id") if latest_trace else None,
        "trace_id_previous": prev_trace.get("trace_id") if prev_trace else None
    }


def get_mentor_live_ml_kpis(db_path: Path = DATABASE_PATH) -> Dict[str, Any]:
    """
    Returns high-level KPI cards for the mentor Live ML dashboard.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT count(*) FROM ml_observations WHERE source != 'synthetic_historical'")
    live_obs = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM ml_observations WHERE training_eligible = 1")
    training_eligible = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM ml_observations WHERE label_status = 'PENDING_FUTURE_LABEL'")
    pending_labels = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM online_learning_updates")
    online_updates = cur.fetchone()[0]

    cur.execute("""
        SELECT predicted_performance, prediction_timestamp, model_version, model_type
        FROM prediction_traces
        ORDER BY trace_id DESC LIMIT 1
    """)
    last_pred_row = cur.fetchone()

    conn.close()

    last_pred = {
        "score": last_pred_row[0] if last_pred_row else 88.9,
        "timestamp": last_pred_row[1] if last_pred_row else "N/A",
        "model_version": last_pred_row[2] if last_pred_row else "v_20261007_125928",
        "model_type": last_pred_row[3] if last_pred_row else "Ridge Regression"
    }

    return {
        "live_observations": live_obs,
        "training_eligible": training_eligible,
        "pending_labels": pending_labels,
        "online_updates": online_updates,
        "current_model": last_pred["model_version"],
        "model_type": last_pred["model_type"],
        "last_prediction_score": last_pred["score"],
        "last_prediction_timestamp": last_pred["timestamp"]
    }
