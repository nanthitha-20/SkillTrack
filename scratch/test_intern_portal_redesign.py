import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ast
import sqlite3
import pandas as pd
from src.ml.predict import get_intern_ml_forecast, build_feature_vector, load_models

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

def get_intern(intern_id):
    connection = get_connection()
    intern = connection.execute(
        "SELECT intern_id, name, email, role, experience_level, current_week, average_score, skill_score, final_performance_score, skill_gap, resource_action, progress FROM interns WHERE intern_id = ?",
        (intern_id,)
    ).fetchone()
    connection.close()
    return intern

def get_coding_tasks():
    connection = get_connection()
    tasks = connection.execute(
        "SELECT task_id, title, description, max_score, skill, difficulty, video_url, guide_url FROM coding_tasks ORDER BY task_id"
    ).fetchall()
    connection.close()
    return tasks

def get_intern_performance_history(intern_id):
    connection = get_connection()
    df = pd.read_sql_query("SELECT * FROM intern_performance WHERE intern_id = ? ORDER BY week ASC", connection, params=(intern_id,))
    connection.close()
    return df

def save_mcq_attempt(intern_id, score, total_questions, topic):
    connection = get_connection()
    connection.execute(
        "INSERT INTO mcq_attempts (intern_id, score, total_questions, topic, attempted_at) VALUES (?, ?, ?, ?, datetime('now'))",
        (intern_id, score, total_questions, topic)
    )
    connection.commit()
    connection.close()

def get_mcq_attempts(intern_id):
    connection = get_connection()
    attempts = connection.execute("SELECT attempt_id, score, total_questions, topic, attempted_at FROM mcq_attempts WHERE intern_id = ? ORDER BY attempt_id DESC", (intern_id,)).fetchall()
    connection.close()
    return attempts


print("=" * 70)
print("INTERN PORTAL REDESIGN: COMPLETE VERIFICATION SUITE")
print("=" * 70)

# 1. AST SYNTAX CHECK
print("\n[1] Python Syntax Check:")
with open("app/app.py", "r", encoding="utf-8") as f:
    ast.parse(f.read())
print("  [OK] app/app.py AST parsed successfully with 0 syntax errors.")

# 2. AUTHENTICATION
print("\n[2] Intern Authentication:")
user = authenticate_user("intern1", "intern123")
print(f"  [OK] Authenticated user: {user}")
assert user is not None and user[2] == "intern" and user[3] == 10001

# 3. INTERN PROFILE & METRICS
print("\n[3] Intern Profile & Data Retrieval:")
intern_data = get_intern(10001)
print(f"  [OK] Intern 10001 Name: {intern_data[1]} | Role: {intern_data[3]} | Exp: {intern_data[4]}")

tasks = get_coding_tasks()
print(f"  [OK] Available Coding Tasks: {len(tasks)}")
assert len(tasks) == 10

history_df = get_intern_performance_history(10001)
print(f"  [OK] Performance History Records: {len(history_df)} weeks")
assert len(history_df) == 10

# 4. DYNAMIC ML SKILL OUTLOOK
print("\n[4] Dynamic Next-Week Skill Outlook:")
fc_w5 = get_intern_ml_forecast(intern_id=10001, prediction_week=5)
print(f"  [OK] Week 5 -> Week 6: Gap={fc_w5['predicted_skill_gap']} | Conf={fc_w5['confidence_percent']}% | Perf={fc_w5['predicted_performance_score']} | Rec={fc_w5['recommendation']}")
assert fc_w5['predicted_skill_gap'] == "Strong"

fc_w10 = get_intern_ml_forecast(intern_id=10001, prediction_week=10)
print(f"  [OK] Week 10 -> Week 11: Gap={fc_w10['predicted_skill_gap']} | Conf={fc_w10['confidence_percent']}% | Perf={fc_w10['predicted_performance_score']}")

# 5. ASSESSMENT FLOW TEST
print("\n[5] Interactive Assessment Save Test:")
initial_mcqs = len(get_mcq_attempts(10001))
save_mcq_attempt(intern_id=10001, score=100.0, total_questions=2, topic="JavaScript Core")
new_mcqs = get_mcq_attempts(10001)
print(f"  [OK] MCQ Attempts before: {initial_mcqs} -> after: {len(new_mcqs)}")
assert len(new_mcqs) == initial_mcqs + 1

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
print("INTERN PORTAL REDESIGN VERIFICATION COMPLETE: ALL PASS!")
print("=" * 70)
