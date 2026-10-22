import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"
CSV_PATH = BASE_DIR / "data" / "skilltrack_frontend_interns_500x10_29cols.csv"


def seed_database():
    print(f"Reading synthetic dataset from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Temporarily disable foreign keys during schema rebuild
    cursor.execute("PRAGMA foreign_keys = OFF;")

    # 1. Ensure/Update schema
    # Create intern_performance table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS intern_performance (
            performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            intern_id INTEGER NOT NULL,
            week INTEGER NOT NULL,
            days_since_start INTEGER,
            skill TEXT,
            learning_hours REAL,
            videos_completed INTEGER,
            guides_completed INTEGER,
            tasks_completed INTEGER,
            engagement_rate REAL,
            coding_attempts INTEGER,
            execution_count INTEGER,
            average_time_minutes REAL,
            compile_error_rate REAL,
            runtime_error_rate REAL,
            test_cases_total INTEGER,
            test_cases_passed INTEGER,
            coding_score REAL,
            mcq_attempts INTEGER,
            mcq_score REAL,
            debugging_score REAL,
            code_review_score REAL,
            previous_skill_score REAL,
            skill_score REAL,
            score_improvement REAL,
            final_performance_score REAL,
            skill_gap TEXT,
            resource_action TEXT,
            FOREIGN KEY (intern_id) REFERENCES interns(intern_id),
            UNIQUE(intern_id, week)
        )
    """)

    # Check if interns table needs schema migration from OULAD to Frontend Interns
    cursor.execute("PRAGMA table_info(interns)")
    intern_cols = [c[1] for c in cursor.fetchall()]

    if "studied_credits" in intern_cols or "role" not in intern_cols:
        print("Migrating interns table schema to frontend intern structure...")
        # Create new_interns table
        cursor.execute("DROP TABLE IF EXISTS new_interns")
        cursor.execute("""
            CREATE TABLE new_interns (
                intern_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                role TEXT,
                experience_level TEXT,
                current_week INTEGER DEFAULT 10,
                average_score REAL,
                skill_score REAL,
                final_performance_score REAL,
                skill_gap TEXT,
                resource_action TEXT,
                progress REAL DEFAULT 0
            )
        """)
        cursor.execute("DROP TABLE interns")
        cursor.execute("ALTER TABLE new_interns RENAME TO interns")

    # 2. Clean orphan submissions referencing non-existent coding tasks
    cursor.execute("""
        DELETE FROM submissions
        WHERE task_id NOT IN (SELECT task_id FROM coding_tasks)
    """)
    print("Cleaned orphan submissions referencing non-existent tasks.")

    # 3. Seed 500 Intern Profiles from the dataset (Week 10 summary or averages)
    print("Seeding intern profiles...")
    intern_groups = df.groupby('intern_id')

    intern_records = []
    for intern_id, group in intern_groups:
        latest = group.sort_values('week').iloc[-1]
        role = latest['role']
        experience_level = latest['experience_level']
        avg_score = round(float(group['final_performance_score'].mean()), 2)
        skill_score = round(float(latest['skill_score']), 2)
        final_perf = round(float(latest['final_performance_score']), 2)
        skill_gap = latest['skill_gap']
        resource_action = latest['resource_action']
        name = f"Intern {intern_id}"
        email = f"intern{intern_id}@skilltrack.local"
        progress = round(float(latest['week'] * 10.0), 1)  # 10 weeks -> 100%

        intern_records.append((
            int(intern_id),
            name,
            email,
            role,
            experience_level,
            int(latest['week']),
            avg_score,
            skill_score,
            final_perf,
            skill_gap,
            resource_action,
            progress
        ))

    cursor.executemany("""
        INSERT OR REPLACE INTO interns
        (intern_id, name, email, role, experience_level, current_week, average_score, skill_score, final_performance_score, skill_gap, resource_action, progress)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, intern_records)
    print(f"Seeded {len(intern_records)} intern profiles.")

    # 4. Seed 5,000 Weekly Performance Records
    print("Seeding weekly intern performance records...")
    perf_records = []
    for _, row in df.iterrows():
        perf_records.append((
            int(row['intern_id']),
            int(row['week']),
            int(row['days_since_start']),
            str(row['skill']),
            float(row['learning_hours']),
            int(row['videos_completed']),
            int(row['guides_completed']),
            int(row['tasks_completed']),
            float(row['engagement_rate']),
            int(row['coding_attempts']),
            int(row['execution_count']),
            float(row['average_time_minutes']),
            float(row['compile_error_rate']),
            float(row['runtime_error_rate']),
            int(row['test_cases_total']),
            int(row['test_cases_passed']),
            float(row['coding_score']),
            int(row['mcq_attempts']),
            float(row['mcq_score']),
            float(row['debugging_score']),
            float(row['code_review_score']),
            float(row['previous_skill_score']),
            float(row['skill_score']),
            float(row['score_improvement']),
            float(row['final_performance_score']),
            str(row['skill_gap']),
            str(row['resource_action'])
        ))

    cursor.executemany("""
        INSERT OR REPLACE INTO intern_performance
        (intern_id, week, days_since_start, skill, learning_hours, videos_completed, guides_completed, tasks_completed,
         engagement_rate, coding_attempts, execution_count, average_time_minutes, compile_error_rate, runtime_error_rate,
         test_cases_total, test_cases_passed, coding_score, mcq_attempts, mcq_score, debugging_score, code_review_score,
         previous_skill_score, skill_score, score_improvement, final_performance_score, skill_gap, resource_action)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, perf_records)
    print(f"Seeded {len(perf_records)} weekly performance records.")

    # 5. Ensure Mentors Table has Demo Mentor
    cursor.execute("""
        INSERT OR IGNORE INTO mentors (mentor_id, name, email)
        VALUES (1, 'Demo Mentor', 'mentor@skilltrack.local')
    """)

    # 6. Ensure Demo Users
    cursor.execute("""
        INSERT OR REPLACE INTO users (user_id, username, password, role, intern_id, mentor_id)
        VALUES (1, 'intern1', 'intern123', 'intern', 10001, NULL)
    """)
    cursor.execute("""
        INSERT OR REPLACE INTO users (user_id, username, password, role, intern_id, mentor_id)
        VALUES (2, 'mentor1', 'mentor123', 'mentor', NULL, 1)
    """)
    print("Demo users updated: intern1 -> intern_id 10001, mentor1 -> mentor_id 1.")

    # 7. Check Foreign Keys
    cursor.execute("PRAGMA foreign_key_check")
    fk_violations = cursor.fetchall()
    print("Foreign Key Violations:", fk_violations)

    conn.commit()
    conn.close()
    print("Database migration and seeding completed successfully!")


if __name__ == "__main__":
    seed_database()
