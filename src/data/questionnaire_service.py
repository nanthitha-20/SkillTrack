"""
SkillTrack-ML: Questionnaire Service
Handles intern self-assessment questionnaire submission, persistence,
feature extraction, ML observation creation, and prediction triggering.

Questions elicit ML-relevant signals: learning effort, task activity,
debugging confidence, code review quality, and technical focus skill.
These map directly to existing feature columns in intern_performance
and the 43-feature prediction schema.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


class QuestionnaireValidationError(Exception):
    """Raised when questionnaire inputs fail validation."""
    pass


# The questionnaire questions definition — maps to ML feature schema
QUESTIONNAIRE_QUESTIONS: List[Dict[str, Any]] = [
    {
        "key": "learning_hours",
        "label": "How many hours did you spend learning this week?",
        "type": "slider",
        "min": 0.0,
        "max": 40.0,
        "default": 4.0,
        "step": 0.5,
        "unit": "hours",
        "feature": "learning_hours",
        "feature_type": "numeric",
        "required": True,
        "help": "Include all study, videos, guides, and hands-on coding time."
    },
    {
        "key": "tasks_completed",
        "label": "How many coding tasks / exercises did you complete?",
        "type": "slider",
        "min": 0,
        "max": 20,
        "default": 2,
        "step": 1,
        "unit": "tasks",
        "feature": "tasks_completed",
        "feature_type": "integer",
        "required": True,
        "help": "Count only tasks you completed, not just attempted."
    },
    {
        "key": "videos_completed",
        "label": "How many instructional videos did you watch?",
        "type": "slider",
        "min": 0,
        "max": 20,
        "default": 2,
        "step": 1,
        "unit": "videos",
        "feature": "videos_completed",
        "feature_type": "integer",
        "required": True,
        "help": "Complete views, not partial."
    },
    {
        "key": "guides_completed",
        "label": "How many reading guides / documentation chapters did you complete?",
        "type": "slider",
        "min": 0,
        "max": 20,
        "default": 1,
        "step": 1,
        "unit": "guides",
        "feature": "guides_completed",
        "feature_type": "integer",
        "required": True,
        "help": "Official documentation, textbook chapters, or learning guides."
    },
    {
        "key": "average_time_minutes",
        "label": "On average, how many minutes did you spend per coding exercise?",
        "type": "slider",
        "min": 5.0,
        "max": 240.0,
        "default": 45.0,
        "step": 5.0,
        "unit": "minutes",
        "feature": "average_time_minutes",
        "feature_type": "numeric",
        "required": True,
        "help": "Average time from starting to submitting a coding exercise."
    },
    {
        "key": "debugging_confidence",
        "label": "How confident do you feel about debugging code? (0 = not at all, 100 = very confident)",
        "type": "slider",
        "min": 0.0,
        "max": 100.0,
        "default": 60.0,
        "step": 1.0,
        "unit": "/100",
        "feature": "debugging_score",
        "feature_type": "numeric",
        "required": True,
        "help": "Your self-assessed debugging skill level."
    },
    {
        "key": "code_review_score",
        "label": "How would you rate the quality of your code this week? (0 = poor, 100 = excellent)",
        "type": "slider",
        "min": 0.0,
        "max": 100.0,
        "default": 65.0,
        "step": 1.0,
        "unit": "/100",
        "feature": "code_review_score",
        "feature_type": "numeric",
        "required": True,
        "help": "Consider code readability, structure, and adherence to best practices."
    },
    {
        "key": "technical_focus_skill",
        "label": "Which technical area did you focus on most this week?",
        "type": "selectbox",
        "options": [
            "JavaScript Core",
            "React & State Management",
            "CSS & Responsive Design",
            "HTML & Semantics",
            "API Integration",
            "Data Structures & Algorithms",
            "Testing & Debugging",
            "Version Control & Git"
        ],
        "default": "JavaScript Core",
        "feature": "skill",
        "feature_type": "categorical",
        "required": True,
        "help": "Select the primary skill area you practiced this week."
    },
    {
        "key": "challenge_encountered",
        "label": "Describe the main challenge you faced this week (optional)",
        "type": "text_area",
        "max_length": 500,
        "default": "",
        "feature": None,
        "feature_type": "text_only",
        "required": False,
        "help": "This helps your mentor understand your blockers. Not used as an ML feature."
    }
]


def validate_questionnaire_response(answers: Dict[str, Any]) -> None:
    """Validates questionnaire answers against expected ranges and types."""
    for q in QUESTIONNAIRE_QUESTIONS:
        key = q["key"]
        if q["required"] and key not in answers:
            raise QuestionnaireValidationError(f"Required field missing: '{key}'")

        val = answers.get(key)
        if val is None:
            if q["required"]:
                raise QuestionnaireValidationError(f"Required field '{key}' is None")
            continue

        if q["type"] == "slider":
            try:
                fval = float(val)
            except (TypeError, ValueError):
                raise QuestionnaireValidationError(f"Field '{key}' must be numeric, got: {val!r}")
            if fval < q["min"] or fval > q["max"]:
                raise QuestionnaireValidationError(
                    f"Field '{key}' value {fval} is out of range [{q['min']}, {q['max']}]"
                )

        elif q["type"] == "selectbox":
            if val not in q["options"]:
                raise QuestionnaireValidationError(
                    f"Field '{key}' value '{val}' is not a valid option. Allowed: {q['options']}"
                )

        elif q["type"] == "text_area":
            if isinstance(val, str) and len(val) > q.get("max_length", 1000):
                raise QuestionnaireValidationError(
                    f"Field '{key}' exceeds maximum length of {q['max_length']} characters"
                )


def extract_ml_features(answers: Dict[str, Any], intern_week: int) -> Dict[str, Any]:
    """
    Extracts the ML-relevant features from questionnaire answers.
    Maps questionnaire keys to intern_performance column names.
    Returns a dict suitable for updating intern_performance.
    """
    return {
        "learning_hours": float(answers.get("learning_hours", 0.0)),
        "tasks_completed": int(answers.get("tasks_completed", 0)),
        "videos_completed": int(answers.get("videos_completed", 0)),
        "guides_completed": int(answers.get("guides_completed", 0)),
        "average_time_minutes": float(answers.get("average_time_minutes", 30.0)),
        "debugging_score": float(answers.get("debugging_confidence", 0.0)),
        "code_review_score": float(answers.get("code_review_score", 0.0)),
        "skill": str(answers.get("technical_focus_skill", "JavaScript Core")),
        "week": intern_week,
    }


def build_feature_inspection_table(answers: Dict[str, Any], intern_week: int) -> List[Dict[str, Any]]:
    """
    Builds a human-readable feature inspection table showing:
    raw input, transformed value, feature name, feature type, and inclusion status.
    """
    ml_feats = extract_ml_features(answers, intern_week)
    rows = []
    for q in QUESTIONNAIRE_QUESTIONS:
        raw_val = answers.get(q["key"], "")
        mapped_feat = q["feature"]
        if mapped_feat and mapped_feat in ml_feats:
            transformed_val = ml_feats[mapped_feat]
            included = True
        else:
            transformed_val = "N/A (text only)"
            included = False

        rows.append({
            "Question": q["label"][:60] + ("..." if len(q["label"]) > 60 else ""),
            "Raw Input": str(raw_val),
            "ML Feature": mapped_feat or "—",
            "Transformed Value": transformed_val,
            "Feature Type": q["feature_type"],
            "Used in Prediction": "✅" if included else "❌"
        })
    return rows


def save_questionnaire_response(
    intern_id: int,
    week: int,
    answers: Dict[str, Any],
    db_path: Path = DATABASE_PATH,
    allow_update: bool = True
) -> Tuple[int, bool]:
    """
    Validates and persists a questionnaire response to questionnaire_responses.
    Updates intern_performance for the given week with the extracted ML features.
    Creates an ml_observations record for lineage tracking.
    Triggers a prediction via the ML pipeline.

    Returns: (response_id, was_updated_existing)
    Raises: QuestionnaireValidationError on invalid input.
    """
    validate_questionnaire_response(answers)
    ml_feats = extract_ml_features(answers, week)
    now_str = datetime.now().isoformat(timespec="seconds")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        # Verify intern exists
        cur.execute("SELECT intern_id FROM interns WHERE intern_id = ?", (intern_id,))
        if cur.fetchone() is None:
            raise QuestionnaireValidationError(f"Intern ID {intern_id} not found in database.")

        # Check for existing response for this intern/week
        cur.execute(
            "SELECT response_id FROM questionnaire_responses WHERE intern_id = ? AND week = ?",
            (intern_id, week)
        )
        existing = cur.fetchone()
        was_update = existing is not None

        if was_update and not allow_update:
            response_id = existing[0]
            conn.close()
            return response_id, True  # Return existing without re-saving

        if was_update:
            # Update existing response
            cur.execute("""
                UPDATE questionnaire_responses
                SET learning_hours = ?, tasks_completed = ?, videos_completed = ?,
                    guides_completed = ?, average_time_minutes = ?,
                    debugging_confidence = ?, code_review_score = ?,
                    technical_focus_skill = ?, challenge_encountered = ?,
                    submitted_at = ?
                WHERE intern_id = ? AND week = ?
            """, (
                ml_feats["learning_hours"],
                ml_feats["tasks_completed"],
                ml_feats["videos_completed"],
                ml_feats["guides_completed"],
                ml_feats["average_time_minutes"],
                ml_feats["debugging_score"],
                ml_feats["code_review_score"],
                ml_feats["skill"],
                answers.get("challenge_encountered", ""),
                now_str,
                intern_id, week
            ))
            response_id = existing[0]
        else:
            # Insert new response
            cur.execute("""
                INSERT INTO questionnaire_responses
                (intern_id, week, learning_hours, tasks_completed, videos_completed,
                 guides_completed, average_time_minutes, debugging_confidence,
                 code_review_score, technical_focus_skill, challenge_encountered, submitted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                intern_id,
                week,
                ml_feats["learning_hours"],
                ml_feats["tasks_completed"],
                ml_feats["videos_completed"],
                ml_feats["guides_completed"],
                ml_feats["average_time_minutes"],
                ml_feats["debugging_score"],
                ml_feats["code_review_score"],
                ml_feats["skill"],
                answers.get("challenge_encountered", ""),
                now_str
            ))
            response_id = cur.lastrowid

        # Sync extracted ML features to intern_performance (upsert for the week)
        cur.execute(
            "SELECT performance_id FROM intern_performance WHERE intern_id = ? AND week = ?",
            (intern_id, week)
        )
        perf_row = cur.fetchone()
        if perf_row:
            cur.execute("""
                UPDATE intern_performance
                SET learning_hours = ?,
                    tasks_completed = ?,
                    videos_completed = ?,
                    guides_completed = ?,
                    average_time_minutes = ?,
                    debugging_score = ?,
                    code_review_score = ?,
                    skill = ?
                WHERE intern_id = ? AND week = ?
            """, (
                ml_feats["learning_hours"],
                ml_feats["tasks_completed"],
                ml_feats["videos_completed"],
                ml_feats["guides_completed"],
                ml_feats["average_time_minutes"],
                ml_feats["debugging_score"],
                ml_feats["code_review_score"],
                ml_feats["skill"],
                intern_id, week
            ))

        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    # Create ml_observations record for questionnaire submission
    try:
        from src.data.live_data_ingestion import ingest_observation
        composite_score = round(
            0.4 * ml_feats["debugging_score"]
            + 0.3 * ml_feats["code_review_score"]
            + 0.3 * min(ml_feats["learning_hours"] * 3.0, 100.0),
            2
        )
        ingest_observation(
            intern_id=intern_id,
            week=week,
            source="questionnaire",
            event_type="questionnaire_submitted",
            metric_name="questionnaire_score",
            metric_value=composite_score,
            payload={
                "response_id": response_id,
                "learning_hours": ml_feats["learning_hours"],
                "tasks_completed": ml_feats["tasks_completed"],
                "videos_completed": ml_feats["videos_completed"],
                "guides_completed": ml_feats["guides_completed"],
                "debugging_score": ml_feats["debugging_score"],
                "code_review_score": ml_feats["code_review_score"],
                "skill": ml_feats["skill"],
                "week": week
            },
            db_path=db_path,
            label_status="IMMEDIATE_LABELED",
            training_eligible=1,
            outcome_value=composite_score,
            source_record_id=response_id
        )
    except Exception:
        # Observation ingestion failure does not block questionnaire save
        pass

    return response_id, was_update


def get_questionnaire_response(
    intern_id: int,
    week: int,
    db_path: Path = DATABASE_PATH
) -> Optional[Dict[str, Any]]:
    """Returns the questionnaire response for an intern/week, or None if not found."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT response_id, intern_id, week, learning_hours, tasks_completed,
               videos_completed, guides_completed, average_time_minutes,
               debugging_confidence, code_review_score, technical_focus_skill,
               challenge_encountered, submitted_at
        FROM questionnaire_responses
        WHERE intern_id = ? AND week = ?
        ORDER BY response_id DESC LIMIT 1
    """, (intern_id, week))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    cols = [
        "response_id", "intern_id", "week", "learning_hours", "tasks_completed",
        "videos_completed", "guides_completed", "average_time_minutes",
        "debugging_confidence", "code_review_score", "technical_focus_skill",
        "challenge_encountered", "submitted_at"
    ]
    return dict(zip(cols, row))


def get_all_questionnaire_responses(
    intern_id: int,
    db_path: Path = DATABASE_PATH
) -> List[Dict[str, Any]]:
    """Returns all questionnaire responses for an intern, ordered by week."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT response_id, intern_id, week, learning_hours, tasks_completed,
               videos_completed, guides_completed, average_time_minutes,
               debugging_confidence, code_review_score, technical_focus_skill,
               challenge_encountered, submitted_at
        FROM questionnaire_responses
        WHERE intern_id = ?
        ORDER BY week ASC
    """, (intern_id,))
    rows = cur.fetchall()
    conn.close()
    cols = [
        "response_id", "intern_id", "week", "learning_hours", "tasks_completed",
        "videos_completed", "guides_completed", "average_time_minutes",
        "debugging_confidence", "code_review_score", "technical_focus_skill",
        "challenge_encountered", "submitted_at"
    ]
    return [dict(zip(cols, row)) for row in rows]


def get_questionnaire_summary_stats(db_path: Path = DATABASE_PATH) -> Dict[str, Any]:
    """Returns aggregate statistics about questionnaire submissions."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM questionnaire_responses")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT intern_id) FROM questionnaire_responses")
    unique_interns = cur.fetchone()[0]
    cur.execute("SELECT MAX(submitted_at) FROM questionnaire_responses")
    latest = cur.fetchone()[0]
    conn.close()
    return {
        "total_responses": total,
        "unique_interns": unique_interns,
        "latest_submission": latest or "None"
    }
