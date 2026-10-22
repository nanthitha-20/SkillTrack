"""
SkillTrack-ML: Database Migration for Prediction Tracing and Online Lineage.
Creates prediction_traces table and extends online_learning_updates schema.
"""

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


def run_migration(db_path: Path = DATABASE_PATH) -> None:
    print(f"Running prediction trace migration on {db_path}...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Create prediction_traces table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_traces (
            trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
            intern_id INTEGER NOT NULL,
            prediction_timestamp TEXT NOT NULL,
            prediction_type TEXT NOT NULL DEFAULT 'next_week_forecast',
            model_version TEXT NOT NULL,
            model_type TEXT NOT NULL,
            prediction_source TEXT NOT NULL DEFAULT 'production_batch',
            predicted_gap TEXT,
            predicted_performance REAL,
            confidence_score REAL,
            prediction_before REAL,
            prediction_after REAL,
            feature_vector_hash TEXT NOT NULL,
            feature_snapshot TEXT NOT NULL,
            observation_ids_used TEXT,
            latest_observation_id INTEGER,
            training_row_count_at_prediction INTEGER DEFAULT 4500,
            online_model_version TEXT,
            online_update_count INTEGER DEFAULT 0,
            data_cutoff_timestamp TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (intern_id) REFERENCES interns(intern_id)
        );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_traces_intern_ts ON prediction_traces (intern_id, prediction_timestamp);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_traces_latest_obs ON prediction_traces (latest_observation_id);")

    # 2. Extend online_learning_updates if missing columns
    cur.execute("PRAGMA table_info(online_learning_updates);")
    existing_cols = [r[1] for r in cur.fetchall()]

    columns_to_add = [
        ("model_version_before", "TEXT"),
        ("model_version_after", "TEXT"),
        ("observation_ids", "TEXT"),
        ("training_row_count", "INTEGER DEFAULT 1"),
        ("loss", "REAL DEFAULT 0.0"),
        ("features_used", "TEXT"),
        ("target_used", "REAL")
    ]

    for col_name, col_type in columns_to_add:
        if col_name not in existing_cols:
            print(f"Adding column '{col_name}' ({col_type}) to online_learning_updates...")
            cur.execute(f"ALTER TABLE online_learning_updates ADD COLUMN {col_name} {col_type};")

    conn.commit()
    conn.close()
    print("[OK] Prediction traces migration applied successfully.")


if __name__ == "__main__":
    run_migration()
