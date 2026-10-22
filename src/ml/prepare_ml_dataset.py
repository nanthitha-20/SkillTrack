import os
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BASE_DIR / "data" / "skilltrack_frontend_interns_500x10_29cols.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DATA_PATH = OUTPUT_DIR / "intern_ml_dataset.csv"


def prepare_ml_dataset(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = OUTPUT_DATA_PATH
) -> pd.DataFrame:
    """
    Loads raw synthetic frontend intern dataset, constructs historical features,
    aligns future prediction targets without target leakage, and saves the prepared dataset.
    """
    print("=" * 60)
    print("SKILLTRACK-ML: PREPARING ML DATASET")
    print("=" * 60)

    # 1. Load and validate input dataset
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found at: {input_path}")

    print(f"Loading input dataset from: {input_path}")
    raw_df = pd.read_csv(input_path)
    input_rows, input_cols = raw_df.shape
    print(f"Loaded {input_rows} rows and {input_cols} columns from raw dataset.")

    # 2. Sort chronologically by intern and week
    df = raw_df.sort_values(by=["intern_id", "week"]).reset_index(drop=True).copy()

    # 3. Engineer Historical & Cumulative Features (strictly computed from week <= w)
    print("\nEngineering historical features (leakage-free)...")

    # Ratio features
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

    # 4. Create Future Prediction Targets (Shifted: Target at week w is outcome at week w+1)
    print("Aligning future prediction targets (week w -> week w+1 outcome)...")
    df["target_skill_gap"] = df.groupby("intern_id")["skill_gap"].shift(-1)
    df["target_resource_action"] = df.groupby("intern_id")["resource_action"].shift(-1)
    df["target_performance_score"] = df.groupby("intern_id")["final_performance_score"].shift(-1)
    df["target_skill_score"] = df.groupby("intern_id")["skill_score"].shift(-1)

    # 5. Filter out final week records where future target is undefined (week 10 has no week 11)
    # This leaves 4,500 labeled transition records for training/evaluation
    processed_df = df.dropna(subset=["target_skill_gap"]).reset_index(drop=True).copy()
    output_rows, output_cols = processed_df.shape

    # 6. Save prepared dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"\nSaved prepared ML dataset to: {output_path}")

    # 7. Summary & Validation Reporting
    feature_cols = [
        col for col in processed_df.columns
        if not col.startswith("target_") and col not in ["skill_gap", "resource_action"]
    ]
    target_cols = [col for col in processed_df.columns if col.startswith("target_")]

    print("\n" + "=" * 60)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Input Rows:               {input_rows}")
    print(f"Output Rows (Supervised): {output_rows}")
    print(f"Input Columns:            {input_cols}")
    print(f"Output Columns:           {output_cols}")
    print(f"Number of Interns:        {processed_df['intern_id'].nunique()}")
    print(f"Transitions per Intern:   {output_rows // processed_df['intern_id'].nunique()}")
    print(f"Missing Values:           {processed_df.isnull().sum().sum()}")
    print(f"Duplicate Rows:           {processed_df.duplicated().sum()}")
    print(f"\nTotal Features ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"  - {f}")
    print(f"\nTotal Targets ({len(target_cols)}):")
    for t in target_cols:
        print(f"  - {t}")
    print("\nTarget Distribution (`target_skill_gap`):")
    print(processed_df["target_skill_gap"].value_counts().to_string())
    print("\nTarget Distribution (`target_resource_action`):")
    print(processed_df["target_resource_action"].value_counts().to_string())
    print("=" * 60)

    return processed_df


if __name__ == "__main__":
    prepare_ml_dataset()
