"""
SkillTrack-ML: Unified ML Dataset Builder
Combines historical synthetic baseline data with live application-generated observations.
Performs feature engineering with strict temporal cutoff (W -> W+1 shifted targets),
ensures data provenance tracking, and registers dataset versions in SQLite.
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_CSV_PATH = BASE_DIR / "data" / "skilltrack_frontend_interns_500x10_29cols.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DATA_PATH = PROCESSED_DIR / "intern_ml_dataset.csv"
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


def build_unified_dataset(
    db_path: Path = DATABASE_PATH,
    raw_csv_path: Path = RAW_CSV_PATH,
    output_path: Path = OUTPUT_DATA_PATH,
    version_tag: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Constructs the unified ML dataset merging synthetic historical baseline
    with current application-generated records from SQLite.
    Saves to output_path and registers the new dataset version in SQLite.
    """
    print("=" * 70)
    print("SKILLTRACK-ML: BUILDING UNIFIED ML DATASET")
    print("=" * 70)

    # 1. Connect to SQLite to read current state of intern_performance
    conn = sqlite3.connect(db_path)

    # Read intern_performance joined with interns to include static profile features (role, experience_level)
    query = """
        SELECT p.*, i.role, i.experience_level
        FROM intern_performance p
        JOIN interns i ON p.intern_id = i.intern_id
        ORDER BY p.intern_id, p.week
    """
    db_perf_df = pd.read_sql_query(query, conn)

    # Check provenance counts from ml_observations
    obs_summary_query = "SELECT source, count(*) FROM ml_observations GROUP BY source"
    obs_counts = dict(conn.execute(obs_summary_query).fetchall())
    total_app_obs = sum(obs_counts.get(s, 0) for s in ['application', 'quiz', 'coding_submission', 'mentor_evaluation'])

    print(f"Loaded {len(db_perf_df)} weekly performance records from database.")
    print(f"Active application-generated observations in event store: {total_app_obs}")

    # 2. Tag Provenance
    # Synthetic baseline covers interns 10001 to 10500 across weeks 1-10
    # Any rows with user observations recorded are tracked as hybrid/application-updated
    db_perf_df["data_provenance"] = "synthetic_baseline"

    # Identify interns with live application events
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT intern_id FROM ml_observations WHERE source != 'synthetic'")
    active_intern_ids = {r[0] for r in cur.fetchall()}
    if active_intern_ids:
        # Mark active intern Week 10 as hybrid/application-updated
        mask_live = (db_perf_df["intern_id"].isin(active_intern_ids)) & (db_perf_df["week"] == 10)
        db_perf_df.loc[mask_live, "data_provenance"] = "application_updated"

    # 3. Sort chronologically by intern and week
    df = db_perf_df.sort_values(by=["intern_id", "week"]).reset_index(drop=True).copy()

    # 4. Feature Engineering (Strictly computed from week <= w)
    print("\nEngineering historical and cumulative features...")

    # Ratio feature: Test case pass rate
    df["test_case_pass_rate"] = np.where(
        df["test_cases_total"] > 0,
        df["test_cases_passed"] / df["test_cases_total"],
        0.0
    )

    # Cumulative effort/activity metrics up to current week
    cumulative_cols = [
        "learning_hours",
        "tasks_completed",
        "videos_completed",
        "guides_completed",
        "coding_attempts",
        "execution_count"
    ]
    for col in cumulative_cols:
        df[f"cumulative_{col}"] = df.groupby("intern_id")[col].transform(lambda x: x.cumsum())

    # Historical expanding averages up to current week
    avg_cols = [
        "coding_score",
        "mcq_score",
        "debugging_score",
        "code_review_score",
        "skill_score",
        "learning_hours",
        "engagement_rate",
        "compile_error_rate",
        "runtime_error_rate",
        "average_time_minutes"
    ]
    for col in avg_cols:
        df[f"historical_{col}_avg"] = df.groupby("intern_id")[col].transform(
            lambda x: x.expanding().mean()
        )

    # 5. Create Future Prediction Targets (Shifted: Target at week w is outcome at week w+1)
    print("Aligning future prediction targets (Week W -> Week W+1 outcome)...")
    df["target_skill_gap"] = df.groupby("intern_id")["skill_gap"].shift(-1)
    df["target_resource_action"] = df.groupby("intern_id")["resource_action"].shift(-1)
    df["target_performance_score"] = df.groupby("intern_id")["final_performance_score"].shift(-1)
    df["target_skill_score"] = df.groupby("intern_id")["skill_score"].shift(-1)

    # 6. Filter to supervised training records (where Week W+1 target exists)
    # Week 10 is currently the origin week awaiting Week 11 target completion
    labeled_df = df.dropna(subset=["target_skill_gap"]).reset_index(drop=True).copy()

    total_rows = len(df)
    supervised_rows = len(labeled_df)
    feature_cols = [
        c for c in labeled_df.columns
        if not c.startswith("target_") and c not in [
            "skill_gap", "resource_action", "performance_id", "data_provenance"
        ]
    ]

    # 7. Save dataset artifact
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved unified ML dataset to: {output_path}")

    # 8. Register dataset version in SQLite
    if version_tag is None:
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_tag = f"v_unified_{ts_tag}"

    composition = {
        "synthetic_baseline": int((df["data_provenance"] == "synthetic_baseline").sum()),
        "application_updated": int((df["data_provenance"] == "application_updated").sum()),
        "external_context": obs_counts.get("external", 0)
    }

    cur.execute("""
        INSERT INTO ml_dataset_versions (
            version_tag, created_at, source_composition,
            total_row_count, synthetic_row_count, application_row_count, external_row_count,
            feature_count, training_eligible_rows, artifact_path, description
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        version_tag,
        datetime.now().isoformat(timespec="seconds"),
        json.dumps(composition),
        total_rows,
        composition["synthetic_baseline"],
        composition["application_updated"],
        composition["external_context"],
        len(feature_cols),
        supervised_rows,
        str(output_path.relative_to(BASE_DIR)).replace("\\", "/"),
        f"Unified dataset build with {composition['application_updated']} live application observations integrated"
    ))
    conn.commit()
    conn.close()

    print(f"[OK] Registered dataset version in SQLite: {version_tag}")
    print(f"Total Rows:               {total_rows}")
    print(f"Supervised Rows:          {supervised_rows}")
    print(f"Feature Columns:          {len(feature_cols)}")
    print(f"Composition:              {composition}")
    print("=" * 70)

    return labeled_df, version_tag


if __name__ == "__main__":
    build_unified_dataset()
