"""
Unit & Integration Tests for Assessment Ingestion, Rubric-Based Hiring Decision Engine,
and Observation Lineage in SkillTrack-ML.
"""

import unittest
import sqlite3
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from src.data.assessment_service import (
    validate_and_save_assessment,
    get_intern_assessments,
    get_assessment_summary_stats,
    AssessmentValidationError,
    VALID_ASSESSMENT_TYPES
)
from src.ml.hiring_engine import (
    compute_hiring_recommendation,
    save_mentor_hiring_decision,
    get_latest_hiring_decision,
    RUBRIC_WEIGHTS
)
from src.database.migrate_assessment_and_hiring import run_assessment_and_hiring_migration

PROD_DB_PATH = Path("data/database/skilltrack.db")


class TestAssessmentService(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_skilltrack.db"
        shutil.copy(PROD_DB_PATH, self.db_path)
        run_assessment_and_hiring_migration(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_valid_assessment_entry_and_observation_emission(self):
        result = validate_and_save_assessment(
            intern_id=10001,
            week=3,
            assessment_type="Technical Assessment",
            assessment_name="Graph Algorithms & Dynamic Programming",
            marks_obtained=88.5,
            max_marks=100.0,
            technical_details={
                "tasks_completed": 5,
                "test_cases_total": 10,
                "test_cases_passed": 10,
                "code_quality_score": 92.0,
                "completion_time_minutes": 45.0
            },
            db_path=self.db_path
        )
        self.assertIn("assessment_id", result)
        self.assertIn("observation_id", result)
        self.assertEqual(result["percentage_score"], 88.5)

        # Verify record in intern_assessments
        assessments = get_intern_assessments(10001, db_path=self.db_path)
        self.assertGreaterEqual(len(assessments), 1)
        found = any(a["assessment_name"] == "Graph Algorithms & Dynamic Programming" for a in assessments)
        self.assertTrue(found)

        # Verify immediate-labeled emission in ml_observations
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT intern_id, event_type, outcome_value, label_status, training_eligible FROM ml_observations WHERE observation_id = ?", (result["observation_id"],))
        obs = cur.fetchone()
        conn.close()

        self.assertIsNotNone(obs)
        self.assertEqual(obs[0], 10001)
        self.assertEqual(obs[1], "task_submitted")
        self.assertEqual(obs[2], 88.5)
        self.assertEqual(obs[3], "IMMEDIATE_LABELED")
        self.assertEqual(obs[4], 1)

    def test_invalid_assessment_validations(self):
        # 1. Nonexistent intern
        with self.assertRaises(AssessmentValidationError):
            validate_and_save_assessment(
                intern_id=999999,
                week=1,
                assessment_type="Technical Assessment",
                assessment_name="Test",
                marks_obtained=80.0,
                db_path=self.db_path
            )

        # 2. Invalid assessment type
        with self.assertRaises(AssessmentValidationError):
            validate_and_save_assessment(
                intern_id=10001,
                week=1,
                assessment_type="Unknown Assessment Type",
                assessment_name="Test",
                marks_obtained=80.0,
                db_path=self.db_path
            )

        # 3. Invalid score (negative)
        with self.assertRaises(AssessmentValidationError):
            validate_and_save_assessment(
                intern_id=10001,
                week=1,
                assessment_type="MCQ Quiz",
                assessment_name="Test",
                marks_obtained=-5.0,
                db_path=self.db_path
            )

        # 4. Invalid score (greater than max_marks)
        with self.assertRaises(AssessmentValidationError):
            validate_and_save_assessment(
                intern_id=10001,
                week=1,
                assessment_type="MCQ Quiz",
                assessment_name="Test",
                marks_obtained=105.0,
                max_marks=100.0,
                db_path=self.db_path
            )

        # 5. Invalid week number
        with self.assertRaises(AssessmentValidationError):
            validate_and_save_assessment(
                intern_id=10001,
                week=25,
                assessment_type="MCQ Quiz",
                assessment_name="Test",
                marks_obtained=80.0,
                db_path=self.db_path
            )

    def test_summary_stats(self):
        validate_and_save_assessment(
            intern_id=10001,
            week=1,
            assessment_type="MCQ Quiz",
            assessment_name="Python Fundamentals",
            marks_obtained=90.0,
            db_path=self.db_path
        )
        stats = get_assessment_summary_stats(self.db_path)
        self.assertGreaterEqual(stats["total_assessments"], 1)
        self.assertGreaterEqual(stats["average_percentage"], 0.0)


class TestHiringEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_hiring.db"
        shutil.copy(PROD_DB_PATH, self.db_path)
        run_assessment_and_hiring_migration(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("src.ml.hiring_engine.get_intern_ml_forecast")
    def test_minimum_evidence_checks(self, mock_forecast):
        mock_forecast.return_value = {
            "predicted_performance": 72.0,
            "predicted_skill_gap": "Moderate",
            "model_version": "Ridge Regression Production"
        }
        # Create a new intern with week=1 and 0 assessments
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("INSERT INTO interns (intern_id, name, email, current_week, average_score, final_performance_score) VALUES (88888, 'New Candidate', 'new@test.edu', 1, 70.0, 70.0)")
        conn.commit()
        conn.close()

        rec = compute_hiring_recommendation(88888, db_path=self.db_path)
        self.assertFalse(rec["min_evidence_met"])
        self.assertEqual(rec["recommendation"], "INSUFFICIENT_EVIDENCE")
        self.assertIn("minimum of 3 active weeks", rec["rationale"])

    @patch("src.ml.hiring_engine.get_intern_ml_forecast")
    def test_full_rubric_recommendation_hire(self, mock_forecast):
        mock_forecast.return_value = {
            "predicted_performance": 91.0,
            "predicted_skill_gap": "Strong",
            "model_version": "Ridge Regression Production"
        }

        # Create a candidate in week 8 with high evaluations
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("INSERT INTO interns (intern_id, name, email, current_week, average_score, final_performance_score) VALUES (77777, 'Top Candidate', 'top@test.edu', 8, 92.0, 93.0)")
        cur.execute("""
            INSERT INTO mentor_evaluations (intern_id, mentor_id, technical_score, communication_score, discipline_score, overall_score, decision, remarks, evaluated_at)
            VALUES (77777, 1, 95.0, 90.0, 95.0, 94.0, 'HIRE', 'Consistently outstanding code.', '2026-10-10 12:00:00')
        """)
        cur.execute("""
            INSERT INTO questionnaire_responses (
                intern_id, week, learning_hours, tasks_completed, videos_completed,
                guides_completed, average_time_minutes, debugging_confidence,
                code_review_score, technical_focus_skill, challenge_encountered, submitted_at
            ) VALUES (
                77777, 8, 20, 10, 5,
                4, 30.0, 95,
                90.0, 'Python', 'None', '2026-10-10'
            )
        """)
        conn.commit()
        conn.close()

        # Add 2 validated assessments
        validate_and_save_assessment(
            intern_id=77777,
            week=6,
            assessment_type="Technical Assessment",
            assessment_name="Distributed Systems",
            marks_obtained=95.0,
            db_path=self.db_path
        )
        validate_and_save_assessment(
            intern_id=77777,
            week=7,
            assessment_type="MCQ Quiz",
            assessment_name="System Architecture",
            marks_obtained=92.0,
            db_path=self.db_path
        )

        rec = compute_hiring_recommendation(77777, db_path=self.db_path)
        self.assertTrue(rec["min_evidence_met"])
        self.assertGreaterEqual(rec["composite_score"], 80.0)
        self.assertEqual(rec["recommendation"], "HIRE")
        self.assertEqual(rec["confidence_level"], "High")

        # Verify rubric weights sum to 1.0
        weights = rec["evidence_summary"]["rubric_weights"]
        self.assertAlmostEqual(sum(weights.values()), 1.0)

        # Save formal hiring decision
        decision_id = save_mentor_hiring_decision(
            intern_id=77777,
            mentor_id=1,
            decision_type=rec["recommendation"],
            composite_readiness_score=rec["composite_score"],
            confidence_level=rec["confidence_level"],
            decision_rationale=rec["rationale"],
            strengths="Architecture, High test pass rate",
            growth_areas="Mentoring junior peers",
            is_final=True,
            evidence_summary=rec["evidence_summary"],
            db_path=self.db_path
        )
        self.assertGreater(decision_id, 0)

        # Retrieve saved decision
        decision = get_latest_hiring_decision(77777, db_path=self.db_path)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["decision_type"], "HIRE")
        self.assertEqual(decision["mentor_id"], 1)
        self.assertAlmostEqual(decision["composite_readiness_score"], rec["composite_score"])


class TestRealDatabaseIntegration(unittest.TestCase):
    def test_intern_10001_hiring_recommendation(self):
        # Intern 10001 in production db has real weeks & evaluations
        rec = compute_hiring_recommendation(10001)
        self.assertIn("recommendation", rec)
        self.assertIn("composite_score", rec)
        self.assertIn("evidence_summary", rec)
        self.assertIn("min_evidence_met", rec)


if __name__ == "__main__":
    unittest.main()
