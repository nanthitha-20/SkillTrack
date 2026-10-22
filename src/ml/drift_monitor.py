"""
SkillTrack-ML: Data Drift & Quality Monitoring
Monitors distribution shifts between the historical synthetic baseline
and incoming application-generated observations.
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "intern_ml_dataset.csv"
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


def compute_feature_drift(
    baseline_series: pd.Series,
    current_series: pd.Series,
    threshold_pct: float = 15.0
) -> Dict[str, Any]:
    """
    Computes relative mean shift and std shift for a continuous feature.
    """
    b_mean = float(baseline_series.mean())
    c_mean = float(current_series.mean())
    b_std = float(baseline_series.std()) if baseline_series.std() > 0 else 1.0
    c_std = float(current_series.std()) if current_series.std() > 0 else 1.0

    mean_diff_pct = abs(c_mean - b_mean) / (abs(b_mean) + 1e-6) * 100.0
    z_score = abs(c_mean - b_mean) / (b_std / np.sqrt(len(current_series) + 1e-6))

    has_drift = mean_diff_pct > threshold_pct

    return {
        "baseline_mean": round(b_mean, 3),
        "current_mean": round(c_mean, 3),
        "baseline_std": round(b_std, 3),
        "current_std": round(c_std, 3),
        "mean_change_percent": round(mean_diff_pct, 2),
        "z_score": round(float(z_score), 3),
        "drift_detected": bool(has_drift)
    }


def assess_system_data_drift(
    db_path: Path = DATABASE_PATH,
    dataset_path: Path = PROCESSED_DATA_PATH
) -> Dict[str, Any]:
    """
    Runs a comprehensive drift analysis across continuous features and categorical targets.
    Returns status ('LOW', 'MODERATE', 'HIGH') and detailed diagnostics.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Provenance summary from SQLite
    cur.execute("SELECT count(*) FROM intern_performance")
    total_perf = cur.fetchone()[0]

    cur.execute("SELECT source, count(*) FROM ml_observations GROUP BY source")
    obs_sources = dict(cur.fetchall())
    app_count = sum(obs_sources.get(s, 0) for s in ['application', 'quiz', 'coding_submission', 'mentor_evaluation'])

    # Read current dataset
    if not dataset_path.exists():
        conn.close()
        return {
            "drift_status": "UNKNOWN",
            "message": "Processed dataset not found.",
            "synthetic_count": total_perf,
            "application_count": app_count
        }

    df = pd.read_csv(dataset_path)
    conn.close()

    # Features to monitor
    monitored_features = [
        "final_performance_score",
        "skill_score",
        "coding_score",
        "mcq_score",
        "engagement_rate",
        "compile_error_rate",
        "learning_hours"
    ]

    # Baseline is the historical training set (first 80% chronologically or interns)
    baseline_df = df[df["week"] <= 8]
    current_df = df[df["week"] > 8]

    feature_metrics = {}
    drift_flags = 0

    for feat in monitored_features:
        if feat in df.columns:
            m = compute_feature_drift(baseline_df[feat], current_df[feat])
            feature_metrics[feat] = m
            if m["drift_detected"]:
                drift_flags += 1

    # Target class distribution shift
    b_class_dist = baseline_df["target_skill_gap"].value_counts(normalize=True).to_dict()
    c_class_dist = current_df["target_skill_gap"].value_counts(normalize=True).to_dict()

    max_class_shift = 0.0
    for cls_name, b_prop in b_class_dist.items():
        c_prop = c_class_dist.get(cls_name, 0.0)
        shift = abs(c_prop - b_prop) * 100.0
        if shift > max_class_shift:
            max_class_shift = shift

    # Determine overall status
    if drift_flags >= 3 or max_class_shift > 20.0:
        overall_status = "HIGH"
    elif drift_flags >= 1 or max_class_shift > 10.0:
        overall_status = "MODERATE"
    else:
        overall_status = "LOW"

    return {
        "assessment_timestamp": datetime.now().isoformat(timespec="seconds"),
        "drift_status": overall_status,
        "monitored_feature_count": len(feature_metrics),
        "drifting_features_count": drift_flags,
        "max_class_distribution_shift_pct": round(max_class_shift, 2),
        "baseline_sample_size": len(baseline_df),
        "current_sample_size": len(current_df),
        "synthetic_baseline_count": total_perf,
        "application_generated_count": app_count,
        "feature_details": feature_metrics,
        "class_distribution_comparison": {
            "baseline": {k: round(v * 100, 1) for k, v in b_class_dist.items()},
            "current": {k: round(v * 100, 1) for k, v in c_class_dist.items()}
        }
    }


if __name__ == "__main__":
    res = assess_system_data_drift()
    print("=" * 60)
    print("SKILLTRACK-ML: DATA DRIFT AUDIT SUMMARY")
    print("=" * 60)
    print(f"Overall Drift Status:       {res['drift_status']}")
    print(f"Drifting Features:          {res['drifting_features_count']} / {res['monitored_feature_count']}")
    print(f"Max Target Class Shift:     {res['max_class_distribution_shift_pct']}%")
    print(f"Synthetic Baseline Records: {res['synthetic_baseline_count']}")
    print(f"Live Application Records:   {res['application_generated_count']}")
    print("=" * 60)
