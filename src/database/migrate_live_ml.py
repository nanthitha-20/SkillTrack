"""
SkillTrack-ML: Live ML Database Migration Script
Safely extends SQLite schema with ml_observations, ml_dataset_versions,
and model_registry without altering existing tables.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


def run_migration(db_path: Path = DATABASE_PATH):
    print("=" * 60)
    print("SKILLTRACK-ML: LIVE ML SCHEMA MIGRATION")
    print("=" * 60)
    print(f"Database target: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Create ml_observations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_observations (
            observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            intern_id INTEGER NOT NULL,
            week INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL CHECK(source IN ('synthetic', 'application', 'quiz', 'coding_submission', 'mentor_evaluation', 'questionnaire', 'external')),
            event_type TEXT NOT NULL,
            metric_name TEXT,
            metric_value REAL,
            payload_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (intern_id) REFERENCES interns(intern_id)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_intern_week ON ml_observations(intern_id, week);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_timestamp ON ml_observations(timestamp);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_source ON ml_observations(source);")
    print("[OK] Created table: ml_observations (with indices)")

    # 2. Create ml_dataset_versions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_dataset_versions (
            dataset_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_tag TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL,
            source_composition TEXT NOT NULL,
            total_row_count INTEGER NOT NULL,
            synthetic_row_count INTEGER NOT NULL,
            application_row_count INTEGER NOT NULL,
            external_row_count INTEGER DEFAULT 0,
            feature_count INTEGER NOT NULL,
            training_eligible_rows INTEGER NOT NULL,
            artifact_path TEXT NOT NULL,
            description TEXT
        )
    """)
    print("[OK] Created table: ml_dataset_versions")

    # 3. Create model_registry table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT UNIQUE NOT NULL,
            model_type TEXT NOT NULL,
            algorithm_cls TEXT NOT NULL,
            algorithm_reg TEXT NOT NULL,
            dataset_version TEXT NOT NULL,
            trained_at TEXT NOT NULL,
            validation_metrics TEXT NOT NULL,
            test_metrics TEXT,
            feature_version TEXT NOT NULL,
            artifact_path_cls TEXT NOT NULL,
            artifact_path_reg TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('candidate', 'production', 'archived', 'rejected')),
            promotion_reason TEXT
        )
    """)
    print("[OK] Created table: model_registry")

    # 4. Check if baseline dataset version is registered
    cursor.execute("SELECT count(*) FROM ml_dataset_versions WHERE version_tag = 'v1.0-synthetic-baseline'")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO ml_dataset_versions (
                version_tag, created_at, source_composition,
                total_row_count, synthetic_row_count, application_row_count, external_row_count,
                feature_count, training_eligible_rows, artifact_path, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "v1.0-synthetic-baseline",
            "2026-09-01T00:00:00",
            json.dumps({"synthetic": 5000, "application": 0, "external": 0}),
            5000,
            5000,
            0,
            0,
            43,
            4500,
            "data/processed/intern_ml_dataset.csv",
            "Initial synthetic benchmark cohort (500 interns x 10 weeks, 4500 transition samples)"
        ))
        print("[OK] Registered initial dataset version: v1.0-synthetic-baseline")

    # 5. Check if initial production model is registered
    cursor.execute("SELECT count(*) FROM model_registry WHERE model_version = 'v1.0-prod'")
    if cursor.fetchone()[0] == 0:
        val_metrics = {
            "classification": {
                "model": "Balanced Logistic Regression",
                "accuracy": 0.8089,
                "macro_f1": 0.7277,
                "needs_improvement_recall": 0.8780,
                "needs_improvement_precision": 0.5070
            },
            "regression": {
                "model": "Ridge Regression (alpha=1.0)",
                "mae": 6.0552,
                "rmse": 7.5577,
                "r2": 0.6320
            }
        }
        test_metrics = {
            "classification": {
                "accuracy": 0.8030,
                "macro_f1": 0.7188,
                "needs_improvement_recall": 0.8077
            },
            "regression": {
                "mae": 6.0712,
                "rmse": 7.6347,
                "r2": 0.6525
            }
        }
        cursor.execute("""
            INSERT INTO model_registry (
                model_version, model_type, algorithm_cls, algorithm_reg,
                dataset_version, trained_at, validation_metrics, test_metrics,
                feature_version, artifact_path_cls, artifact_path_reg, status, promotion_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "v1.0-prod",
            "dual_pipeline",
            "Balanced Logistic Regression",
            "Ridge Regression",
            "v1.0-synthetic-baseline",
            "2026-09-01T00:00:00",
            json.dumps(val_metrics),
            json.dumps(test_metrics),
            "43-features-v1",
            "models/skill_gap_model.pkl",
            "models/performance_model.pkl",
            "production",
            "Initial production benchmark model trained on baseline cohort"
        ))
        print("[OK] Registered initial model version: v1.0-prod")

    # 6. Backfill existing live user events from submissions, mcq_attempts, mentor_evaluations
    # Submissions
    cursor.execute("SELECT submission_id, intern_id, task_id, score, status, submitted_at, language FROM submissions")
    submissions = cursor.fetchall()
    for sub in submissions:
        sub_id, intern_id, task_id, score, status, submitted_at, lang = sub
        # check if already in ml_observations
        cursor.execute(
            "SELECT count(*) FROM ml_observations WHERE source = 'coding_submission' AND payload_json LIKE ?",
            (f'%"submission_id": {sub_id}%',)
        )
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO ml_observations (
                    intern_id, week, timestamp, source, event_type,
                    metric_name, metric_value, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                intern_id,
                10,
                submitted_at or datetime.now().isoformat(timespec="seconds"),
                "coding_submission",
                "task_submitted",
                "coding_score",
                score if score is not None else 85.0,
                json.dumps({"submission_id": sub_id, "task_id": task_id, "language": lang, "status": status}),
                datetime.now().isoformat(timespec="seconds")
            ))
    print(f"[OK] Backfilled {len(submissions)} existing coding submissions into ml_observations")

    # MCQ attempts
    cursor.execute("SELECT attempt_id, intern_id, score, total_questions, topic, attempted_at FROM mcq_attempts")
    mcqs = cursor.fetchall()
    for mcq in mcqs:
        att_id, intern_id, score, total_q, topic, att_at = mcq
        cursor.execute(
            "SELECT count(*) FROM ml_observations WHERE source = 'quiz' AND payload_json LIKE ?",
            (f'%"attempt_id": {att_id}%',)
        )
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO ml_observations (
                    intern_id, week, timestamp, source, event_type,
                    metric_name, metric_value, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                intern_id,
                10,
                att_at or datetime.now().isoformat(timespec="seconds"),
                "quiz",
                "mcq_attempted",
                "mcq_score",
                float(score),
                json.dumps({"attempt_id": att_id, "total_questions": total_q, "topic": topic}),
                datetime.now().isoformat(timespec="seconds")
            ))
    print(f"[OK] Backfilled {len(mcqs)} existing quiz attempts into ml_observations")

    # Mentor evaluations
    cursor.execute("SELECT evaluation_id, intern_id, mentor_id, overall_score, decision, evaluated_at FROM mentor_evaluations")
    evals = cursor.fetchall()
    for ev in evals:
        ev_id, intern_id, mentor_id, ov_score, decision, ev_at = ev
        cursor.execute(
            "SELECT count(*) FROM ml_observations WHERE source = 'mentor_evaluation' AND payload_json LIKE ?",
            (f'%"evaluation_id": {ev_id}%',)
        )
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO ml_observations (
                    intern_id, week, timestamp, source, event_type,
                    metric_name, metric_value, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                intern_id,
                10,
                ev_at or datetime.now().isoformat(timespec="seconds"),
                "mentor_evaluation",
                "mentor_evaluated",
                "mentor_overall_score",
                float(ov_score) if ov_score is not None else 85.0,
                json.dumps({"evaluation_id": ev_id, "mentor_id": mentor_id, "decision": decision}),
                datetime.now().isoformat(timespec="seconds")
            ))
    print(f"[OK] Backfilled {len(evals)} existing mentor evaluations into ml_observations")

    conn.commit()
    conn.close()
    print("=" * 60)
    print("MIGRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_migration()
