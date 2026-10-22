import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ast
import sqlite3
import pandas as pd
from src.ml.predict import get_intern_ml_forecast

DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "database" / "skilltrack.db"

def get_connection():
    return sqlite3.connect(DATABASE_PATH)

def authenticate_user(username, password):
    connection = get_connection()
    user = connection.execute(
        "SELECT user_id, username, role, intern_id, mentor_id FROM users WHERE username = ? AND password = ?",
        (username, password)
    ).fetchone()
    connection.close()
    return user

def get_cohort_summary_df():
    connection = get_connection()
    df = pd.read_sql_query("SELECT * FROM interns ORDER BY intern_id ASC", connection)
    connection.close()
    return df

def get_all_mentor_evaluations():
    connection = get_connection()
    df = pd.read_sql_query(
        """
        SELECT e.*, i.name as intern_name
        FROM mentor_evaluations e
        LEFT JOIN interns i ON e.intern_id = i.intern_id
        ORDER BY e.evaluation_id DESC
        """,
        connection
    )
    connection.close()
    return df

def get_cohort_performance_df():
    connection = get_connection()
    df = pd.read_sql_query(
        """
        SELECT week, skill, AVG(final_performance_score) as avg_final_score
        FROM intern_performance
        GROUP BY week, skill
        ORDER BY week ASC
        """,
        connection
    )
    connection.close()
    return df

def save_evaluation(intern_id, mentor_id, technical_score, communication_score, discipline_score, overall_score, decision, remarks):
    connection = get_connection()
    connection.execute(
        """
        INSERT INTO mentor_evaluations
        (intern_id, mentor_id, technical_score, communication_score, discipline_score, overall_score, decision, remarks, evaluated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (intern_id, mentor_id, technical_score, communication_score, discipline_score, overall_score, decision, remarks)
    )
    connection.commit()
    connection.close()


print("=" * 70)
print("MENTOR PORTAL REDESIGN: COMPLETE VERIFICATION SUITE")
print("=" * 70)

# 1. AST SYNTAX CHECK
print("\n[1] Python Syntax Check:")
with open("app/app.py", "r", encoding="utf-8") as f:
    ast.parse(f.read())
print("  [OK] app/app.py AST parsed successfully with 0 syntax errors.")

# 2. AUTHENTICATION
print("\n[2] Mentor Authentication:")
user = authenticate_user("mentor1", "mentor123")
print(f"  [OK] Authenticated mentor user: {user}")
assert user is not None and user[2] == "mentor" and user[4] == 1

# 3. COHORT METRICS & DATASETS
print("\n[3] Cohort Analytics Data:")
cohort_df = get_cohort_summary_df()
print(f"  [OK] Total Cohort Interns: {len(cohort_df)}")
assert len(cohort_df) == 500

avg_progress = cohort_df["progress"].mean()
avg_perf = cohort_df["final_performance_score"].mean()
needs_attention = cohort_df[cohort_df["skill_gap"].isin(["Needs Improvement", "Developing"])]
print(f"  [OK] Avg Progress: {avg_progress:.1f}% | Avg Performance: {avg_perf:.1f} | Needs Attention: {len(needs_attention)}")

cohort_perf = get_cohort_performance_df()
print(f"  [OK] Cohort Weekly Performance Aggregates: {len(cohort_perf)} records")
assert not cohort_perf.empty

# 4. DYNAMIC ML INFERENCE FOR MULTIPLE INTERNS
print("\n[4] Dynamic Next-Week Skill Outlook (Testing multiple interns):")
for test_id in [10001, 10042, 10105]:
    fc = get_intern_ml_forecast(intern_id=test_id, prediction_week=10)
    print(f"  [OK] Intern {test_id}: Predicted Gap={fc['predicted_skill_gap']} | Conf={fc['confidence_percent']}% | Perf={fc['predicted_performance_score']}")
    assert fc["status"] == "success"

# 5. EVALUATION LOGGING & QUERY
print("\n[5] Mentor Evaluation Submission & History:")
initial_evals = len(get_all_mentor_evaluations())
save_evaluation(
    intern_id=10001,
    mentor_id=1,
    technical_score=85,
    communication_score=90,
    discipline_score=95,
    overall_score=90.0,
    decision="FULL_TIME",
    remarks="Outstanding grasp of frontend principles and proactive learner."
)
all_evals = get_all_mentor_evaluations()
print(f"  [OK] Mentor evaluations before: {initial_evals} -> after: {len(all_evals)}")
assert len(all_evals) == initial_evals + 1

# 6. DATABASE INTEGRITY
print("\n[6] Database Foreign Key Verification:")
conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
c = conn.cursor()
c.execute("PRAGMA foreign_key_check;")
fk = c.fetchall()
print(f"  [OK] PRAGMA foreign_key_check: {fk}")
assert len(fk) == 0
conn.close()

print("\n" + "=" * 70)
print("MENTOR PORTAL REDESIGN VERIFICATION COMPLETE: ALL PASS!")
print("=" * 70)
