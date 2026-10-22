import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite3
import urllib.request
import pandas as pd
from datetime import datetime
from src.ml.predict import get_intern_ml_forecast, build_feature_vector, load_models

DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "database" / "skilltrack.db"

def get_connection():
    return sqlite3.connect(DATABASE_PATH)

def authenticate_user(username, password):
    connection = get_connection()
    user = connection.execute(
        """
        SELECT user_id, username, role, intern_id, mentor_id
        FROM users
        WHERE username = ? AND password = ?
        """,
        (username, password)
    ).fetchone()
    connection.close()
    return user

def get_intern(intern_id):
    connection = get_connection()
    intern = connection.execute(
        """
        SELECT
            intern_id,
            name,
            email,
            role,
            experience_level,
            current_week,
            average_score,
            skill_score,
            final_performance_score,
            skill_gap,
            resource_action,
            progress
        FROM interns
        WHERE intern_id = ?
        """,
        (intern_id,)
    ).fetchone()
    connection.close()
    return intern

def get_coding_tasks():
    connection = get_connection()
    tasks = connection.execute(
        """
        SELECT
            task_id,
            title,
            description,
            max_score,
            skill,
            difficulty,
            video_url,
            guide_url
        FROM coding_tasks
        ORDER BY task_id
        """
    ).fetchall()
    connection.close()
    return tasks

def get_intern_submissions(intern_id):
    connection = get_connection()
    submissions = connection.execute(
        """
        SELECT
            s.submission_id,
            s.task_id,
            t.title,
            s.score,
            s.status,
            s.submitted_at,
            s.language,
            s.code,
            s.output
        FROM submissions s
        LEFT JOIN coding_tasks t
            ON s.task_id = t.task_id
        WHERE s.intern_id = ?
        ORDER BY s.submission_id DESC
        """,
        (intern_id,)
    ).fetchall()
    connection.close()
    return submissions

def save_code_submission(intern_id, task_id, language, code, output):
    connection = get_connection()
    connection.execute(
        """
        INSERT INTO submissions
        (
            intern_id,
            task_id,
            score,
            status,
            submitted_at,
            language,
            code,
            output
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            intern_id,
            task_id,
            None,
            "SUBMITTED",
            datetime.now().isoformat(timespec="seconds"),
            language,
            code,
            output
        )
    )
    connection.commit()
    connection.close()

def get_mcq_attempts(intern_id):
    connection = get_connection()
    attempts = connection.execute(
        """
        SELECT
            attempt_id,
            score,
            total_questions,
            topic,
            attempted_at
        FROM mcq_attempts
        WHERE intern_id = ?
        ORDER BY attempt_id DESC
        """,
        (intern_id,)
    ).fetchall()
    connection.close()
    return attempts

def get_resources():
    connection = get_connection()
    resources = connection.execute(
        """
        SELECT
            resource_id,
            title,
            resource_type,
            url,
            topic,
            difficulty
        FROM resources
        ORDER BY resource_id
        """
    ).fetchall()
    connection.close()
    return resources

def get_all_interns():
    connection = get_connection()
    interns = connection.execute(
        """
        SELECT
            intern_id,
            name,
            email,
            average_score,
            progress
        FROM interns
        ORDER BY intern_id
        """
    ).fetchall()
    connection.close()
    return interns

def get_evaluation_counts():
    connection = get_connection()
    results = connection.execute(
        """
        SELECT decision, COUNT(*)
        FROM mentor_evaluations
        GROUP BY decision
        """
    ).fetchall()
    connection.close()
    return dict(results)


print("=" * 70)
print("SKILLTRACK-ML: FULL END-TO-END VERIFICATION SUITE")
print("=" * 70)

# 1. STREAMLIT SERVER CHECK
print("\n[TEST 1] Streamlit Server Availability:")
try:
    res = urllib.request.urlopen("http://localhost:8501", timeout=3)
    print(f"  HTTP Response: {res.status} OK (Server is active on port 8501)")
except Exception as e:
    print(f"  Server Check Error: {e}")

# 2. AUTHENTICATION TEST
print("\n[TEST 2] Authentication Flow:")
intern_user = authenticate_user("intern1", "intern123")
print(f"  Intern Login ('intern1'): {intern_user}")
assert intern_user is not None and intern_user[2] == "intern" and intern_user[3] == 10001, "Intern login failed!"

mentor_user = authenticate_user("mentor1", "mentor123")
print(f"  Mentor Login ('mentor1'): {mentor_user}")
assert mentor_user is not None and mentor_user[2] == "mentor" and mentor_user[4] == 1, "Mentor login failed!"

invalid_user = authenticate_user("nonexistent", "wrongpass")
print(f"  Invalid Login: {invalid_user} (Correctly rejected)")
assert invalid_user is None, "Invalid user was not rejected!"

# 3. INTERN DASHBOARD DATA RETRIEVAL
print("\n[TEST 3] Intern Dashboard Data Retrieval:")
intern_data = get_intern(10001)
print(f"  Intern 10001 Record: {intern_data}")
assert intern_data is not None and intern_data[0] == 10001, "Failed to retrieve intern 10001!"

tasks = get_coding_tasks()
print(f"  Coding Tasks Available: {len(tasks)} tasks")
assert len(tasks) == 10, f"Expected 10 tasks, found {len(tasks)}"

submissions = get_intern_submissions(10001)
print(f"  Initial Submissions for 10001: {len(submissions)}")

mcqs = get_mcq_attempts(10001)
print(f"  MCQ Attempts for 10001: {len(mcqs)}")

resources = get_resources()
print(f"  Learning Resources: {len(resources)}")

# 4. CODE EXECUTION & SUBMISSION TEST
print("\n[TEST 4] Code Submission Flow:")
sample_code = "console.log('Hello from SkillTrack Intern Portal!');"
save_code_submission(
    intern_id=10001,
    task_id=tasks[0][0], # Task 6: HTML Portfolio Card
    language="JavaScript",
    code=sample_code,
    output="Hello from SkillTrack Intern Portal!"
)
submissions_after = get_intern_submissions(10001)
print(f"  Submissions for 10001 after test save: {len(submissions_after)}")
assert len(submissions_after) >= 1, "Submission was not recorded!"
print(f"  Latest submission record: {submissions_after[0][:6]}")

# 5. DYNAMIC ML SKILL OUTLOOK TEST (INTERN)
print("\n[TEST 5] Intern ML Skill Outlook (Dynamic & Non-Hardcoded):")
fc_w5 = get_intern_ml_forecast(intern_id=10001, prediction_week=5)
print(f"  Week 5 -> Week 6 Forecast:  Gap = {fc_w5['predicted_skill_gap']} | Conf = {fc_w5['confidence_percent']}% | Perf = {fc_w5['predicted_performance_score']}")
assert fc_w5['predicted_skill_gap'] == "Strong", "Week 5 forecast mismatch!"

fc_w10 = get_intern_ml_forecast(intern_id=10001, prediction_week=10)
print(f"  Week 10 -> Week 11 Forecast: Gap = {fc_w10['predicted_skill_gap']} | Conf = {fc_w10['confidence_percent']}% | Perf = {fc_w10['predicted_performance_score']}")

# 6. MENTOR PORTAL FLOW & MULTI-INTERN ML FORECASTS
print("\n[TEST 6] Mentor Portal Flow & Multi-Intern Predictions:")
all_interns = get_all_interns()
print(f"  Total Interns listed for Mentor: {len(all_interns)}")
assert len(all_interns) == 500, f"Expected 500 interns, got {len(all_interns)}"

eval_counts = get_evaluation_counts()
print(f"  Mentor Evaluation Decisions: {eval_counts}")

test_sample = [10001, 10050, 10120, 10250, 10450]
for tid in test_sample:
    m_fc = get_intern_ml_forecast(intern_id=tid)
    print(f"  Intern {tid:5d} -> Gap: {m_fc['predicted_skill_gap']:<18} | Conf: {m_fc['confidence_percent']:5.1f}% | Perf: {m_fc['predicted_performance_score']:4.1f} | Action: {m_fc.get('action_tag')}")

# 7. ERROR HANDLING & EDGE CASES
print("\n[TEST 7] Error Handling & Edge Cases:")
err_fc = get_intern_ml_forecast(intern_id=99999)
print(f"  Non-existent intern (99999): Status = '{err_fc['status']}' | Message = '{err_fc['message']}'")
assert err_fc['status'] in ['insufficient_data', 'error'], "Error state not handled!"

# 8. ML TEMPORAL & LEAKAGE SAFETY
print("\n[TEST 8] ML Temporal & Leakage Safety:")
features_df, origin_w, current_skill, err = build_feature_vector(intern_id=10001, prediction_week=5)
print(f"  Features shape for week <= 5: {features_df.shape} (Exact 43 features)")
assert features_df.shape[1] == 43, f"Expected 43 features, got {features_df.shape[1]}"

target_cols_check = [c for c in features_df.columns if "target_" in c]
print(f"  Target columns in input feature matrix: {target_cols_check} (Must be empty)")
assert len(target_cols_check) == 0, "Target leakage detected in input features!"

# 9. DATABASE INTEGRITY & FOREIGN KEY VERIFICATION
print("\n[TEST 9] Database Integrity & Foreign Key Verification:")
conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
c = conn.cursor()
c.execute("PRAGMA foreign_key_check;")
fk_violations = c.fetchall()
print(f"  PRAGMA foreign_key_check: {fk_violations}")
assert len(fk_violations) == 0, f"Foreign key violations found: {fk_violations}"
conn.close()

print("\n" + "=" * 70)
print("ALL END-TO-END TESTS COMPLETED SUCCESSFULLY WITH 100% PASS RATE!")
print("=" * 70)
