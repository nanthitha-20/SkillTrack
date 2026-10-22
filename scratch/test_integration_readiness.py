import joblib
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

print("="*70)
print("1. INSPECT SAVED MODELS")
print("="*70)
cls_model = joblib.load("models/skill_gap_model.pkl")
reg_model = joblib.load("models/performance_model.pkl")

print(f"Classification Model Pipeline: {cls_model}")
print(f"Steps in Classification Pipeline: {list(cls_model.named_steps.keys())}")
cls_prep = cls_model.named_steps["prep"]
print(f"Transformers in Preprocessor: {[t[0] for t in cls_prep.transformers]}")
cls_features = cls_prep.transformers[0][2] + cls_prep.transformers[1][2]
print(f"Total Features Expected by Classifier ({len(cls_features)}):")
print(cls_features)

print(f"\nRegression Model Pipeline: {reg_model}")
reg_prep = reg_model.named_steps["prep"]
reg_features = reg_prep.transformers[0][2] + reg_prep.transformers[1][2]
print(f"Total Features Expected by Regressor ({len(reg_features)}):")
print(reg_features)

print("\n" + "="*70)
print("2. INSPECT PROCESSED DATASET COLUMNS")
print("="*70)
df_proc = pd.read_csv("data/processed/intern_ml_dataset.csv")
cols = list(df_proc.columns)
print(f"Total columns in intern_ml_dataset.csv: {len(cols)}")
for i, c in enumerate(cols, 1):
    print(f"{i:2d}. {c}")

print("\n" + "="*70)
print("4 & 5. TEST ONE REAL DATABASE INTERN")
print("="*70)
# Connect to SQLite DB in read-only mode
conn = sqlite3.connect("file:data/database/skilltrack.db?mode=ro", uri=True)

# Select intern 10001
intern_id = 10001
test_week = 5 # Predict week 6 outcome using data through week 5

# Fetch intern profile
intern_profile = pd.read_sql_query("SELECT intern_id, role, experience_level FROM interns WHERE intern_id = ?", conn, params=(intern_id,)).iloc[0]

# Fetch intern performance records strictly through test_week (week <= test_week)
perf_history = pd.read_sql_query(
    "SELECT * FROM intern_performance WHERE intern_id = ? AND week <= ? ORDER BY week ASC",
    conn,
    params=(intern_id, test_week)
)
conn.close()

print(f"Live Intern Profile: {intern_profile.to_dict()}")
print(f"Performance records fetched (weeks 1 to {test_week}): {len(perf_history)} rows")

# Construct feature vector for week = test_week
current_week_row = perf_history.iloc[-1].copy()

# Compute ratios
test_case_pass_rate = (current_week_row["test_cases_passed"] / current_week_row["test_cases_total"]) if current_week_row["test_cases_total"] > 0 else 0.0

# Compute cumulative features up to test_week
cumulative_learning_hours = perf_history["learning_hours"].sum()
cumulative_tasks_completed = perf_history["tasks_completed"].sum()
cumulative_videos_completed = perf_history["videos_completed"].sum()
cumulative_guides_completed = perf_history["guides_completed"].sum()
cumulative_coding_attempts = perf_history["coding_attempts"].sum()
cumulative_execution_count = perf_history["execution_count"].sum()

# Compute expanding averages up to test_week
historical_coding_score_avg = perf_history["coding_score"].mean()
historical_mcq_score_avg = perf_history["mcq_score"].mean()
historical_debugging_score_avg = perf_history["debugging_score"].mean()
historical_code_review_score_avg = perf_history["code_review_score"].mean()
historical_skill_score_avg = perf_history["skill_score"].mean()
historical_learning_hours_avg = perf_history["learning_hours"].mean()
historical_engagement_rate_avg = perf_history["engagement_rate"].mean()
historical_compile_error_rate_avg = perf_history["compile_error_rate"].mean()
historical_runtime_error_rate_avg = perf_history["runtime_error_rate"].mean()
historical_average_time_minutes_avg = perf_history["average_time_minutes"].mean()

# Assemble live input dictionary
live_features = {
    "role": intern_profile["role"],
    "experience_level": intern_profile["experience_level"],
    "week": int(current_week_row["week"]),
    "days_since_start": int(current_week_row["days_since_start"]),
    "skill": str(current_week_row["skill"]),
    "learning_hours": float(current_week_row["learning_hours"]),
    "videos_completed": int(current_week_row["videos_completed"]),
    "guides_completed": int(current_week_row["guides_completed"]),
    "tasks_completed": int(current_week_row["tasks_completed"]),
    "engagement_rate": float(current_week_row["engagement_rate"]),
    "coding_attempts": int(current_week_row["coding_attempts"]),
    "execution_count": int(current_week_row["execution_count"]),
    "average_time_minutes": float(current_week_row["average_time_minutes"]),
    "compile_error_rate": float(current_week_row["compile_error_rate"]),
    "runtime_error_rate": float(current_week_row["runtime_error_rate"]),
    "test_cases_total": int(current_week_row["test_cases_total"]),
    "test_cases_passed": int(current_week_row["test_cases_passed"]),
    "coding_score": float(current_week_row["coding_score"]),
    "mcq_attempts": int(current_week_row["mcq_attempts"]),
    "mcq_score": float(current_week_row["mcq_score"]),
    "debugging_score": float(current_week_row["debugging_score"]),
    "code_review_score": float(current_week_row["code_review_score"]),
    "previous_skill_score": float(current_week_row["previous_skill_score"]),
    "skill_score": float(current_week_row["skill_score"]),
    "score_improvement": float(current_week_row["score_improvement"]),
    "final_performance_score": float(current_week_row["final_performance_score"]),
    "test_case_pass_rate": float(test_case_pass_rate),
    "cumulative_learning_hours": float(cumulative_learning_hours),
    "cumulative_tasks_completed": int(cumulative_tasks_completed),
    "cumulative_videos_completed": int(cumulative_videos_completed),
    "cumulative_guides_completed": int(cumulative_guides_completed),
    "cumulative_coding_attempts": int(cumulative_coding_attempts),
    "cumulative_execution_count": int(cumulative_execution_count),
    "historical_coding_score_avg": float(historical_coding_score_avg),
    "historical_mcq_score_avg": float(historical_mcq_score_avg),
    "historical_debugging_score_avg": float(historical_debugging_score_avg),
    "historical_code_review_score_avg": float(historical_code_review_score_avg),
    "historical_skill_score_avg": float(historical_skill_score_avg),
    "historical_learning_hours_avg": float(historical_learning_hours_avg),
    "historical_engagement_rate_avg": float(historical_engagement_rate_avg),
    "historical_compile_error_rate_avg": float(historical_compile_error_rate_avg),
    "historical_runtime_error_rate_avg": float(historical_runtime_error_rate_avg),
    "historical_average_time_minutes_avg": float(historical_average_time_minutes_avg)
}

live_input_df = pd.DataFrame([live_features])
print(f"Constructed feature DataFrame shape: {live_input_df.shape}")
print(f"Features match expected classifier features: {list(live_input_df.columns) == cls_features}")

# Run predictions
pred_skill_gap = cls_model.predict(live_input_df)[0]
pred_skill_gap_proba = cls_model.predict_proba(live_input_df)[0]
classes = list(cls_model.classes_)

pred_performance = reg_model.predict(live_input_df)[0]

print(f"\n--- PREDICTION RESULTS FOR INTERN {intern_id} (Input: through Week {test_week} -> Predicting Week {test_week+1}) ---")
print(f"Predicted Next-Week Skill Gap: {pred_skill_gap}")
print(f"Class Probabilities: {dict(zip(classes, [round(p, 4) for p in pred_skill_gap_proba]))}")
print(f"Predicted Next-Week Performance Score: {pred_performance:.2f}")

# Compare with actual ground truth week 6 outcome from DB
conn2 = sqlite3.connect("file:data/database/skilltrack.db?mode=ro", uri=True)
actual_w6 = pd.read_sql_query("SELECT week, skill_gap, final_performance_score FROM intern_performance WHERE intern_id = ? AND week = ?", conn2, params=(intern_id, test_week+1)).iloc[0]
conn2.close()
print(f"Actual Ground Truth in DB for Week {test_week+1}: Skill Gap = '{actual_w6['skill_gap']}', Performance = {actual_w6['final_performance_score']:.2f}")
