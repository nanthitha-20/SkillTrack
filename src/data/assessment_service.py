"""
SkillTrack-ML: Assessment Service
Handles validation, storage, feature transformation, and ML observation ingestion
for actual intern assessments (Technical coding, Quizzes, Professional performance, Mentor evaluations).
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from src.data.live_data_ingestion import (
    ingest_observation,
    validate_intern_id,
    LiveDataIngestionError,
    DATABASE_PATH
)

VALID_ASSESSMENT_TYPES = [
    "Technical Assessment",
    "MCQ Quiz",
    "Professional & Attendance",
    "Comprehensive Review"
]


class AssessmentValidationError(Exception):
    pass


def validate_and_save_assessment(
    intern_id: int,
    week: int,
    assessment_type: str,
    assessment_name: str,
    marks_obtained: float,
    max_marks: float = 100.0,
    attempt_number: int = 1,
    technical_details: Optional[Dict[str, Any]] = None,
    quiz_details: Optional[Dict[str, Any]] = None,
    professional_details: Optional[Dict[str, Any]] = None,
    mentor_details: Optional[Dict[str, Any]] = None,
    db_path: Path = DATABASE_PATH
) -> Dict[str, Any]:
    """
    Validates, persists, transforms, and ingests an intern assessment into SQLite and the ML observation layer.
    Returns:
        Dict containing assessment_id, observation_id, transformed_features, and ingestion status.
    """
    if assessment_type not in VALID_ASSESSMENT_TYPES:
        raise AssessmentValidationError(f"Invalid assessment type: '{assessment_type}'. Must be one of {VALID_ASSESSMENT_TYPES}")

    if not (1 <= week <= 20):
        raise AssessmentValidationError(f"Invalid training week: {week}. Must be between 1 and 20.")

    if not assessment_name or not assessment_name.strip():
        raise AssessmentValidationError("Assessment name cannot be empty.")

    if max_marks <= 0:
        raise AssessmentValidationError("Maximum marks must be greater than 0.")

    if marks_obtained < 0 or marks_obtained > max_marks:
        raise AssessmentValidationError(f"Marks obtained ({marks_obtained}) must be between 0 and maximum marks ({max_marks}).")

    percentage_score = round((marks_obtained / max_marks) * 100.0, 2)

    tech = technical_details or {}
    quiz = quiz_details or {}
    prof = professional_details or {}
    mentor = mentor_details or {}

    tasks_completed = int(tech.get("tasks_completed", 0))
    test_cases_total = int(tech.get("test_cases_total", 0))
    test_cases_passed = int(tech.get("test_cases_passed", 0))
    code_quality_score = tech.get("code_quality_score")
    completion_time = tech.get("completion_time_minutes")

    if test_cases_total < 0 or test_cases_passed < 0:
        raise AssessmentValidationError("Test case counts cannot be negative.")
    if test_cases_passed > test_cases_total:
        raise AssessmentValidationError("Passed test cases cannot exceed total test cases.")

    questions_attempted = int(quiz.get("questions_attempted", 0))
    correct_answers = int(quiz.get("correct_answers", 0))
    incorrect_answers = int(quiz.get("incorrect_answers", 0))
    topic_category = quiz.get("topic_category", "General Technical")

    task_completion_rate = prof.get("task_completion_rate")
    attendance_rate = prof.get("attendance_rate")
    communication_rating = prof.get("communication_rating")

    mentor_id = mentor.get("mentor_id", 1)
    mentor_score = mentor.get("mentor_score")
    mentor_remarks = mentor.get("mentor_remarks")
    strengths = mentor.get("strengths")
    areas_for_improvement = mentor.get("areas_for_improvement")
    is_final = int(mentor.get("is_final", 0))

    now_str = datetime.now().isoformat(timespec="seconds")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if not validate_intern_id(intern_id, conn):
        conn.close()
        raise AssessmentValidationError(f"Intern ID {intern_id} not found in database.")

    # 1. Insert into intern_assessments
    cur.execute("""
        INSERT INTO intern_assessments (
            intern_id, week, assessment_type, assessment_name, max_marks,
            marks_obtained, percentage_score, attempt_number,
            tasks_completed, test_cases_total, test_cases_passed,
            code_quality_score, completion_time_minutes,
            questions_attempted, correct_answers, incorrect_answers, topic_category,
            task_completion_rate, attendance_rate, communication_rating,
            mentor_id, mentor_score, mentor_remarks, strengths, areas_for_improvement,
            is_final, created_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, (
        intern_id, week, assessment_type, assessment_name.strip(), max_marks,
        marks_obtained, percentage_score, attempt_number,
        tasks_completed, test_cases_total, test_cases_passed,
        code_quality_score, completion_time,
        questions_attempted, correct_answers, incorrect_answers, topic_category,
        task_completion_rate, attendance_rate, communication_rating,
        mentor_id, mentor_score, mentor_remarks, strengths, areas_for_improvement,
        is_final, now_str
    ))
    assessment_id = cur.lastrowid
    conn.commit()
    conn.close()

    # 2. Ingest observation into ML layer
    obs_source = "application"
    event_type = "assessment_recorded"
    metric_name = "final_performance_score"

    if assessment_type == "Technical Assessment":
        obs_source = "coding_submission"
        event_type = "task_submitted"
        metric_name = "coding_score"
    elif assessment_type == "MCQ Quiz":
        obs_source = "quiz"
        event_type = "mcq_attempted"
        metric_name = "mcq_score"
    elif assessment_type == "Comprehensive Review" and mentor_score is not None:
        obs_source = "mentor_evaluation"
        event_type = "mentor_evaluated"
        metric_name = "mentor_overall_score"

    obs_payload = {
        "assessment_id": assessment_id,
        "assessment_name": assessment_name,
        "assessment_type": assessment_type,
        "percentage_score": percentage_score,
        "test_cases_total": test_cases_total,
        "test_cases_passed": test_cases_passed,
        "topic": topic_category
    }

    obs_id = ingest_observation(
        intern_id=intern_id,
        week=week,
        source=obs_source,
        event_type=event_type,
        metric_name=metric_name,
        metric_value=percentage_score,
        payload=obs_payload,
        timestamp=now_str,
        label_status="IMMEDIATE_LABELED",
        training_eligible=1,
        outcome_value=percentage_score,
        source_record_id=assessment_id,
        db_path=db_path
    )

    # Link observation_id back to intern_assessments
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "UPDATE intern_assessments SET observation_id = ? WHERE assessment_id = ?",
        (obs_id, assessment_id)
    )
    conn.commit()
    conn.close()

    return {
        "assessment_id": assessment_id,
        "observation_id": obs_id,
        "intern_id": intern_id,
        "week": week,
        "percentage_score": percentage_score,
        "assessment_type": assessment_type,
        "label_status": "IMMEDIATE_LABELED",
        "training_eligible": 1,
        "timestamp": now_str
    }


def get_intern_assessments(
    intern_id: Optional[int] = None,
    assessment_type: Optional[str] = None,
    week: Optional[int] = None,
    db_path: Path = DATABASE_PATH
) -> List[Dict[str, Any]]:
    """Fetches stored assessments with filtering."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    query = """
        SELECT a.*, i.name as intern_name, i.role, i.experience_level
        FROM intern_assessments a
        JOIN interns i ON a.intern_id = i.intern_id
        WHERE 1=1
    """
    params: List[Any] = []

    if intern_id is not None:
        query += " AND a.intern_id = ?"
        params.append(intern_id)
    if assessment_type:
        query += " AND a.assessment_type = ?"
        params.append(assessment_type)
    if week is not None:
        query += " AND a.week = ?"
        params.append(week)

    query += " ORDER BY a.created_at DESC, a.assessment_id DESC"

    cur.execute(query, params)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    conn.close()

    return [dict(zip(cols, r)) for r in rows]


def get_assessment_summary_stats(db_path: Path = DATABASE_PATH) -> Dict[str, Any]:
    """Calculates live assessment statistics derived from database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*), COUNT(DISTINCT intern_id), AVG(percentage_score) FROM intern_assessments")
    tot, unique_interns, avg_pct = cur.fetchone()

    cur.execute("SELECT assessment_type, COUNT(*) FROM intern_assessments GROUP BY assessment_type")
    type_counts = dict(cur.fetchall())

    cur.execute("SELECT MAX(created_at) FROM intern_assessments")
    latest_ts = cur.fetchone()[0]

    conn.close()
    return {
        "total_assessments": tot or 0,
        "unique_interns": unique_interns or 0,
        "average_percentage": round(avg_pct, 2) if avg_pct is not None else 0.0,
        "by_type": type_counts,
        "latest_timestamp": latest_ts
    }
