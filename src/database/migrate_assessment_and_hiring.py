"""
SkillTrack-ML: Safe Database Migration for Assessments and Hiring Decisions
Creates:
1. `intern_assessments`: Comprehensive assessment submissions (Technical, Quiz, Professional, Mentor).
2. `mentor_hiring_decisions`: Traceable, explainable hiring decision records.
Does NOT modify or drop any existing tables.
"""

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


def run_assessment_and_hiring_migration(db_path: Path = DATABASE_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Assessments table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS intern_assessments (
            assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            intern_id INTEGER NOT NULL,
            week INTEGER NOT NULL,
            assessment_type TEXT NOT NULL,
            assessment_name TEXT NOT NULL,
            max_marks REAL NOT NULL DEFAULT 100.0,
            marks_obtained REAL NOT NULL,
            percentage_score REAL NOT NULL,
            attempt_number INTEGER NOT NULL DEFAULT 1,
            
            -- Technical specifics
            tasks_completed INTEGER DEFAULT 0,
            test_cases_total INTEGER DEFAULT 0,
            test_cases_passed INTEGER DEFAULT 0,
            code_quality_score REAL,
            completion_time_minutes REAL,
            
            -- Quiz specifics
            questions_attempted INTEGER DEFAULT 0,
            correct_answers INTEGER DEFAULT 0,
            incorrect_answers INTEGER DEFAULT 0,
            topic_category TEXT,
            
            -- Professional / Behavioral
            task_completion_rate REAL,
            attendance_rate REAL,
            communication_rating REAL,
            
            -- Mentor remarks & evaluation
            mentor_id INTEGER,
            mentor_score REAL,
            mentor_remarks TEXT,
            strengths TEXT,
            areas_for_improvement TEXT,
            is_final INTEGER DEFAULT 0,
            
            -- Metadata & Ingestion lineage
            observation_id INTEGER,
            created_at TEXT NOT NULL,
            
            FOREIGN KEY (intern_id) REFERENCES interns(intern_id),
            FOREIGN KEY (mentor_id) REFERENCES mentors(mentor_id)
        )
    """)

    # 2. Mentor Hiring Decisions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mentor_hiring_decisions (
            decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
            intern_id INTEGER NOT NULL,
            mentor_id INTEGER NOT NULL,
            decision_type TEXT NOT NULL, -- 'HIRE', 'EXTEND_INTERNSHIP', 'DO_NOT_RECOMMEND', 'INSUFFICIENT_EVIDENCE'
            is_final INTEGER NOT NULL DEFAULT 0, -- 1=Final, 0=Provisional Decision-Support
            composite_readiness_score REAL NOT NULL,
            confidence_level TEXT NOT NULL, -- 'High', 'Medium', 'Provisional'
            decision_rationale TEXT NOT NULL,
            strengths TEXT,
            growth_areas TEXT,
            
            -- Audit & Provenance Evidence
            actual_assessment_avg REAL,
            predicted_performance_score REAL,
            predicted_gap_category TEXT,
            mentor_evaluation_score REAL,
            evidence_summary_json TEXT,
            created_at TEXT NOT NULL,
            
            FOREIGN KEY (intern_id) REFERENCES interns(intern_id),
            FOREIGN KEY (mentor_id) REFERENCES mentors(mentor_id)
        )
    """)

    conn.commit()
    conn.close()
    print("Assessment and Hiring Decisions schema migration completed successfully.")


if __name__ == "__main__":
    run_assessment_and_hiring_migration()
