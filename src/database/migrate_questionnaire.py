"""
SkillTrack-ML: Questionnaire Database Schema Migration
Creates questionnaire_responses table and indexes in SQLite,
and updates ml_observations source CHECK constraint to include 'questionnaire'.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


def run_questionnaire_migration(db_path: Path = DATABASE_PATH) -> bool:
    print(f"Applying questionnaire migration on: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS questionnaire_responses (
            response_id INTEGER PRIMARY KEY AUTOINCREMENT,
            intern_id INTEGER NOT NULL,
            week INTEGER NOT NULL,
            learning_hours REAL NOT NULL,
            tasks_completed INTEGER NOT NULL,
            videos_completed INTEGER NOT NULL,
            guides_completed INTEGER NOT NULL,
            average_time_minutes REAL NOT NULL,
            debugging_confidence REAL NOT NULL,
            code_review_score REAL NOT NULL,
            technical_focus_skill TEXT NOT NULL,
            challenge_encountered TEXT,
            submitted_at TEXT NOT NULL,
            FOREIGN KEY (intern_id) REFERENCES interns(intern_id),
            UNIQUE(intern_id, week)
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_questionnaire_intern_week ON questionnaire_responses(intern_id, week);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_questionnaire_submitted_at ON questionnaire_responses(submitted_at);")

    # Update ml_observations CHECK constraint to include 'questionnaire'
    cur.execute("SELECT sql FROM sqlite_master WHERE name='ml_observations'")
    row = cur.fetchone()
    if row and "'questionnaire'" not in row[0]:
        print("Updating ml_observations CHECK constraint to include 'questionnaire'...")
        cur.execute("PRAGMA foreign_keys = OFF;")
        cur.execute("""
            CREATE TABLE ml_observations_new (
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
                label_status TEXT DEFAULT 'IMMEDIATE_LABELED',
                training_eligible INTEGER DEFAULT 1,
                outcome_value REAL,
                target_week INTEGER,
                source_record_id INTEGER,
                FOREIGN KEY (intern_id) REFERENCES interns(intern_id)
            )
        """)
        cur.execute("""
            INSERT INTO ml_observations_new (
                observation_id, intern_id, week, timestamp, source, event_type,
                metric_name, metric_value, payload_json, created_at,
                label_status, training_eligible, outcome_value, target_week, source_record_id
            )
            SELECT observation_id, intern_id, week, timestamp, source, event_type,
                   metric_name, metric_value, payload_json, created_at,
                   label_status, training_eligible, outcome_value, target_week, source_record_id
            FROM ml_observations
        """)
        cur.execute("DROP TABLE ml_observations;")
        cur.execute("ALTER TABLE ml_observations_new RENAME TO ml_observations;")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_intern_week ON ml_observations(intern_id, week);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_timestamp ON ml_observations(timestamp);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_source ON ml_observations(source);")
        cur.execute("PRAGMA foreign_keys = ON;")
        print("ml_observations schema updated successfully.")

    conn.commit()
    conn.close()
    print("Questionnaire migration applied successfully.")
    return True


if __name__ == "__main__":
    run_questionnaire_migration()
