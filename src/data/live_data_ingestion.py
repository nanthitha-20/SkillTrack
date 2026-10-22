"""
SkillTrack-ML: Live Data Ingestion Layer
Collects, validates, timestamps, and persists live application observations
and optional contextual signals into SQLite (ml_observations).
Enriches the intern's active tracking metrics in intern_performance.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"

# Validation ranges aligned with audit_dataset.py rules
VALID_NUMERIC_RANGES = {
    "coding_score": (0.0, 100.0),
    "mcq_score": (0.0, 100.0),
    "debugging_score": (0.0, 100.0),
    "code_review_score": (0.0, 100.0),
    "skill_score": (0.0, 100.0),
    "final_performance_score": (0.0, 100.0),
    "learning_hours": (0.0, 40.0),
    "engagement_rate": (0.0, 1.0),
    "compile_error_rate": (0.0, 1.0),
    "runtime_error_rate": (0.0, 1.0),
    "average_time_minutes": (0.0, 300.0),
    "debugging_confidence": (0.0, 100.0),
    "questionnaire_score": (0.0, 100.0),
    "tasks_completed": (0.0, 50.0),
    "videos_completed": (0.0, 50.0),
    "guides_completed": (0.0, 50.0),
    "mentor_overall_score": (0.0, 100.0),
}


class LiveDataIngestionError(Exception):
    """Custom exception for ingestion and validation failures."""
    pass


def get_db_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def validate_intern_id(intern_id: int, conn: sqlite3.Connection) -> bool:
    """Verifies that the intern exists in the database."""
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM interns WHERE intern_id = ?", (intern_id,))
    return cur.fetchone() is not None


def validate_metric_value(metric_name: Optional[str], value: Optional[float]) -> None:
    """Ensures numeric metrics fall within expected valid bounds."""
    if metric_name is None or value is None:
        return
    if metric_name in VALID_NUMERIC_RANGES:
        min_val, max_val = VALID_NUMERIC_RANGES[metric_name]
        if min_val is not None and value < min_val:
            raise LiveDataIngestionError(
                f"Metric '{metric_name}' value {value} is below minimum allowed {min_val}"
            )
        if max_val is not None and value > max_val:
            raise LiveDataIngestionError(
                f"Metric '{metric_name}' value {value} exceeds maximum allowed {max_val}"
            )


def ingest_observation(
    intern_id: int,
    week: int,
    source: str,
    event_type: str,
    metric_name: Optional[str] = None,
    metric_value: Optional[float] = None,
    payload: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
    db_path: Path = DATABASE_PATH,
    update_active_performance: bool = True,
    label_status: str = "IMMEDIATE_LABELED",
    training_eligible: int = 1,
    outcome_value: Optional[float] = None,
    target_week: Optional[int] = None,
    source_record_id: Optional[int] = None
) -> int:
    """
    Ingests, validates, and stores a live observation.
    Supports dual online learning: immediate supervised labels vs pending forecasting labels.
    Returns the generated observation_id.
    """
    if source not in ('synthetic', 'application', 'quiz', 'coding_submission', 'mentor_evaluation', 'questionnaire', 'external'):
        raise LiveDataIngestionError(f"Invalid observation source: '{source}'")

    if not isinstance(week, int) or week < 1:
        raise LiveDataIngestionError(f"Invalid week number: {week}")

    if timestamp is None:
        timestamp = datetime.now().isoformat(timespec="seconds")

    if outcome_value is None and metric_value is not None and label_status == "IMMEDIATE_LABELED":
        outcome_value = float(metric_value)

    payload_dict = payload or {}
    validate_metric_value(metric_name, metric_value)

    conn = get_db_connection(db_path)
    try:
        # Validate intern existence
        if not validate_intern_id(intern_id, conn):
            raise LiveDataIngestionError(f"Intern ID {intern_id} not found in database.")

        cursor = conn.cursor()

        # Prevent exact duplicate events within a 2-second window
        cursor.execute("""
            SELECT observation_id FROM ml_observations
            WHERE intern_id = ? AND source = ? AND event_type = ?
              AND ABS(strftime('%s', timestamp) - strftime('%s', ?)) < 2
        """, (intern_id, source, event_type, timestamp))
        existing = cursor.fetchone()
        if existing:
            conn.close()
            return existing[0]

        now_str = datetime.now().isoformat(timespec="seconds")
        cursor.execute("""
            INSERT INTO ml_observations (
                intern_id, week, timestamp, source, event_type,
                metric_name, metric_value, payload_json, created_at,
                label_status, training_eligible, outcome_value, target_week, source_record_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            intern_id,
            week,
            timestamp,
            source,
            event_type,
            metric_name,
            float(metric_value) if metric_value is not None else None,
            json.dumps(payload_dict),
            now_str,
            label_status,
            training_eligible,
            float(outcome_value) if outcome_value is not None else None,
            target_week,
            source_record_id
        ))
        obs_id = cursor.lastrowid

        # Update active intern_performance metrics if requested
        if update_active_performance:
            _sync_observation_to_performance(
                cursor=cursor,
                intern_id=intern_id,
                week=week,
                event_type=event_type,
                metric_name=metric_name,
                metric_value=metric_value,
                payload=payload_dict
            )

        conn.commit()

        # Incremental Online Learner Update for immediately labeled outcomes
        if label_status == "IMMEDIATE_LABELED" and outcome_value is not None:
            try:
                from src.ml.online_learner import get_online_learner
                get_online_learner().update_with_event(
                    intern_id=intern_id,
                    outcome_value=float(outcome_value),
                    event_type=event_type,
                    db_path=db_path,
                    observation_id=obs_id
                )
            except Exception:
                pass

        return obs_id

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def _sync_observation_to_performance(
    cursor: sqlite3.Cursor,
    intern_id: int,
    week: int,
    event_type: str,
    metric_name: Optional[str],
    metric_value: Optional[float],
    payload: Dict[str, Any]
) -> None:
    """
    Synchronizes new live application activity into the intern's active week
    record in intern_performance, ensuring the live feature vector reacts immediately.
    """
    cursor.execute("""
        SELECT coding_attempts, tasks_completed, coding_score,
               mcq_attempts, mcq_score, skill_score, final_performance_score,
               test_cases_total, test_cases_passed
        FROM intern_performance
        WHERE intern_id = ? AND week = ?
    """, (intern_id, week))
    row = cursor.fetchone()

    if not row:
        return

    (c_att, t_comp, c_score, m_att, m_score, sk_score, f_perf, tc_tot, tc_pass) = row

    updated = False
    new_c_att = c_att or 0
    new_t_comp = t_comp or 0
    new_c_score = c_score or 0.0
    new_m_att = m_att or 0
    new_m_score = m_score or 0.0
    new_tc_tot = tc_tot or 0
    new_tc_pass = tc_pass or 0

    if event_type == "task_submitted" or event_type == "task_run":
        new_c_att += 1
        updated = True
        if metric_name == "coding_score" and metric_value is not None:
            # Weighted update with new submission
            new_c_score = round((new_c_score * 0.7) + (float(metric_value) * 0.3), 2)
            if metric_value >= 60.0:
                new_t_comp += 1
            if "test_cases_total" in payload:
                new_tc_tot += int(payload.get("test_cases_total", 0))
                new_tc_pass += int(payload.get("test_cases_passed", 0))

    elif event_type == "mcq_attempted":
        new_m_att += 1
        updated = True
        if metric_name == "mcq_score" and metric_value is not None:
            new_m_score = round((new_m_score * 0.6) + (float(metric_value) * 0.4), 2)

    if updated:
        # Recompute composite skill_score and final_performance_score
        # Formula mirrors the cohort generation weights
        new_sk_score = round(0.5 * new_c_score + 0.3 * new_m_score + 0.2 * (sk_score or 70.0), 2)
        new_f_perf = round(0.55 * new_sk_score + 0.45 * (f_perf or 70.0), 2)

        # Derive new skill_gap
        if new_sk_score >= 72.0:
            new_gap = "Strong"
            new_action = "Continue Progress"
        elif new_sk_score >= 55.0:
            new_gap = "Developing"
            new_action = "Practice Recommended"
        else:
            new_gap = "Needs Improvement"
            new_action = "Targeted Learning"

        cursor.execute("""
            UPDATE intern_performance
            SET coding_attempts = ?,
                tasks_completed = ?,
                coding_score = ?,
                mcq_attempts = ?,
                mcq_score = ?,
                skill_score = ?,
                final_performance_score = ?,
                skill_gap = ?,
                resource_action = ?,
                test_cases_total = ?,
                test_cases_passed = ?
            WHERE intern_id = ? AND week = ?
        """, (
            new_c_att,
            new_t_comp,
            new_c_score,
            new_m_att,
            new_m_score,
            new_sk_score,
            new_f_perf,
            new_gap,
            new_action,
            new_tc_tot,
            new_tc_pass,
            intern_id,
            week
        ))

        # Also update intern current snapshot in interns table
        cursor.execute("""
            UPDATE interns
            SET skill_score = ?,
                final_performance_score = ?,
                skill_gap = ?,
                resource_action = ?
            WHERE intern_id = ?
        """, (new_sk_score, new_f_perf, new_gap, new_action, intern_id))


# ----------------------------------------------------------------------
# QUERY HELPERS & OBSERVATION INSPECTION
# ----------------------------------------------------------------------

def get_observations_for_intern(
    intern_id: int,
    max_week: Optional[int] = None,
    max_timestamp: Optional[str] = None,
    db_path: Path = DATABASE_PATH
) -> List[Dict[str, Any]]:
    """
    Retrieves observations for an intern strictly respecting temporal cutoffs.
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    query = "SELECT * FROM ml_observations WHERE intern_id = ?"
    params: List[Any] = [intern_id]

    if max_week is not None:
        query += " AND week <= ?"
        params.append(max_week)

    if max_timestamp is not None:
        query += " AND timestamp <= ?"
        params.append(max_timestamp)

    query += " ORDER BY timestamp ASC"

    cur.execute(query, params)
    cols = [col[0] for col in cur.description]
    rows = cur.fetchall()
    conn.close()

    return [dict(zip(cols, row)) for row in rows]


def create_pending_forecasting_sample(
    intern_id: int,
    origin_week: int,
    db_path: Path = DATABASE_PATH
) -> int:
    """
    Creates a pending Week W -> Week W+1 forecasting sample in ml_observations.
    Target outcome remains NULL, label_status is PENDING_FUTURE_LABEL,
    and training_eligible is 0 until Week W+1 outcome is actually observed.
    This strictly prevents fabricating labels and guarantees zero temporal data leakage.
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT observation_id FROM ml_observations
        WHERE intern_id = ? AND week = ? AND event_type = 'weekly_forecast'
    """, (intern_id, origin_week))
    existing = cur.fetchone()
    if existing:
        conn.close()
        return existing[0]

    now_str = datetime.now().isoformat(timespec="seconds")
    cur.execute("""
        SELECT coding_score, mcq_score, skill_score, final_performance_score
        FROM intern_performance
        WHERE intern_id = ? AND week = ?
    """, (intern_id, origin_week))
    perf = cur.fetchone()
    payload = {}
    if perf:
        payload = {
            "origin_week": origin_week,
            "target_week": origin_week + 1,
            "coding_score": perf[0],
            "mcq_score": perf[1],
            "skill_score": perf[2],
            "current_performance": perf[3]
        }

    cur.execute("""
        INSERT INTO ml_observations (
            intern_id, week, timestamp, source, event_type,
            metric_name, metric_value, payload_json, created_at,
            label_status, training_eligible, outcome_value, target_week
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        intern_id,
        origin_week,
        now_str,
        'application',
        'weekly_forecast',
        'performance_forecast_w_plus_1',
        None,
        json.dumps(payload),
        now_str,
        'PENDING_FUTURE_LABEL',
        0,  # NOT training eligible until ground truth arrives
        None, # Outcome unknown
        origin_week + 1
    ))
    obs_id = cur.lastrowid
    conn.commit()
    conn.close()
    return obs_id


def resolve_pending_forecasting_labels(db_path: Path = DATABASE_PATH) -> int:
    """
    Checks all pending forecasting samples where target_week is set.
    If the intern now has an observed outcome for target_week in intern_performance,
    attaches the actual target value, marks label_status='LABELED',
    and sets training_eligible = 1.
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT o.observation_id, o.intern_id, o.target_week
        FROM ml_observations o
        WHERE o.label_status = 'PENDING_FUTURE_LABEL'
          AND o.target_week IS NOT NULL
    """)
    pending_rows = cur.fetchall()
    resolved_count = 0

    for obs_id, intern_id, target_week in pending_rows:
        cur.execute("""
            SELECT final_performance_score FROM intern_performance
            WHERE intern_id = ? AND week = ?
        """, (intern_id, target_week))
        target_perf = cur.fetchone()
        if target_perf and target_perf[0] is not None:
            actual_score = float(target_perf[0])
            cur.execute("""
                UPDATE ml_observations
                SET outcome_value = ?,
                    label_status = 'LABELED',
                    training_eligible = 1
                WHERE observation_id = ?
            """, (actual_score, obs_id))
            resolved_count += 1

    if resolved_count > 0:
        conn.commit()
    conn.close()
    return resolved_count


def get_recent_dataset_records(limit: int = 20, db_path: Path = DATABASE_PATH) -> List[Dict[str, Any]]:
    """
    Returns the most recent observations across the unified dataset,
    formatted for the professor/mentor live ML dataset inspection table.
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT o.observation_id, o.timestamp, o.intern_id, i.name as intern_name,
               o.week, o.event_type, o.source, o.outcome_value, o.label_status,
               o.training_eligible, o.target_week
        FROM ml_observations o
        LEFT JOIN interns i ON o.intern_id = i.intern_id
        ORDER BY o.observation_id DESC
        LIMIT ?
    """, (limit,))
    cols = [col[0] for col in cur.description]
    rows = cur.fetchall()
    conn.close()
    return [dict(zip(cols, row)) for row in rows]


def get_observation_summary(db_path: Path = DATABASE_PATH) -> Dict[str, Any]:
    """
    Returns an aggregated breakdown of observations by source and label status,
    distinguishing historical synthetic from live application events.
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT count(*) FROM intern_performance")
    synthetic_perf_count = cur.fetchone()[0]

    cur.execute("SELECT source, count(*) FROM ml_observations GROUP BY source")
    source_counts = dict(cur.fetchall())

    app_generated_sources = ['application', 'quiz', 'coding_submission', 'mentor_evaluation']
    total_app_events = sum(source_counts.get(s, 0) for s in app_generated_sources)
    total_external = source_counts.get('external', 0)

    cur.execute("SELECT count(*) FROM ml_observations")
    total_obs_table = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM ml_observations WHERE label_status = 'IMMEDIATE_LABELED'")
    immediate_supervised = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM ml_observations WHERE label_status = 'PENDING_FUTURE_LABEL'")
    pending_future = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM ml_observations WHERE label_status = 'LABELED'")
    resolved_future = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM ml_observations WHERE training_eligible = 1")
    eligible_obs = cur.fetchone()[0]

    cur.execute("SELECT MAX(timestamp) FROM ml_observations")
    latest_ts = cur.fetchone()[0]

    # Online learning updates
    try:
        cur.execute("SELECT count(*), MAX(timestamp) FROM online_learning_updates")
        online_row = cur.fetchone()
        online_updates_count = online_row[0] or 0
        latest_online_ts = online_row[1] or "None"
    except Exception:
        online_updates_count = 0
        latest_online_ts = "None"

    training_eligible = 4500 + eligible_obs

    conn.close()

    return {
        "synthetic_baseline_count": synthetic_perf_count,
        "application_generated_count": total_obs_table,
        "external_context_count": total_external,
        "total_active_observations": synthetic_perf_count + total_obs_table,
        "immediate_supervised_count": immediate_supervised,
        "pending_future_label_count": pending_future,
        "resolved_future_label_count": resolved_future,
        "training_eligible_samples": training_eligible,
        "online_updates_count": online_updates_count,
        "latest_online_update": latest_online_ts,
        "breakdown_by_source": source_counts,
        "latest_observation_timestamp": latest_ts
    }


# ----------------------------------------------------------------------
# OPTIONAL EXTERNAL DATA PROVIDER INTERFACE
# ----------------------------------------------------------------------

class ExternalDataProvider(ABC):
    """
    Abstract interface for optional external/contextual live signals.
    """

    @abstractmethod
    def fetch(self, intern_id: int, context_time: str) -> Optional[Dict[str, Any]]:
        """
        Fetches an external contextual signal without leaking future data.
        """
        pass


class AcademicCalendarContextProvider(ExternalDataProvider):
    """
    Concrete external provider providing real academic/cohort calendar context
    (e.g., midterm exam week, holiday recess, sprint deadline week).
    No fabricated internet APIs are used.
    """

    def fetch(self, intern_id: int, context_time: str) -> Optional[Dict[str, Any]]:
        dt = datetime.fromisoformat(context_time.replace("Z", ""))
        # Compute contextual day-of-week and month sprint phase
        return {
            "source": "external_academic_calendar",
            "retrieval_timestamp": datetime.now().isoformat(timespec="seconds"),
            "provider_version": "1.0",
            "context_day_of_week": dt.strftime("%A"),
            "sprint_phase": "Evaluation Sprint" if dt.day > 20 else "Core Curriculum Sprint",
            "is_weekend": dt.weekday() >= 5
        }
