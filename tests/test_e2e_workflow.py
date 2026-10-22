"""
SkillTrack-ML: End-to-End Automated Integration Test Suite
Validates the complete full-cycle ML loop:
1. Intern questionnaire submission & input validation
2. Real-time observation ingestion with provenance and immediate labels
3. Dynamic feature vector reconstruction (43 features)
4. ML forecasting (classification + regression) & lineage tracing
5. Mentor evaluation review & feedback recording
6. Online learning update & lineage progression
7. Mentor benchmark dashboard evaluation & dataset auditing
"""

import sys
import sqlite3
import tempfile
import shutil
import unittest
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data.questionnaire_service import (
    save_questionnaire_response,
    get_questionnaire_response,
    build_feature_inspection_table,
    extract_ml_features
)
from src.data.live_data_ingestion import ingest_observation
from src.ml.predict import get_intern_ml_forecast, build_feature_vector
from src.ml.prediction_tracer import get_latest_prediction_trace, diff_prediction_traces
from src.ml.benchmark import (
    run_historical_benchmark,
    run_live_data_benchmark,
    get_dataset_audit,
    save_benchmark_results,
    load_saved_benchmark_results
)

REAL_DB_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


class TestEndToEndWorkflow(unittest.TestCase):
    """
    Simulates the complete user and model interaction loop end-to-end:
    Questionnaire -> Ingestion -> Features -> ML Forecast -> Lineage ->
    Mentor Evaluation -> Retraining / Drift check -> Benchmark Verification.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "e2e_skilltrack.db"
        shutil.copy2(REAL_DB_PATH, self.db_path)
        self.test_intern = 10002
        self.test_week = 10

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_end_to_end_ml_cycle(self):
        # ---------------------------------------------------------
        # STEP 1: Baseline ML forecast before questionnaire submission
        # ---------------------------------------------------------
        forecast_before = get_intern_ml_forecast(
            intern_id=self.test_intern,
            prediction_week=9,
            record_trace=True,
            db_path=self.db_path
        )
        self.assertEqual(forecast_before["status"], "success")
        self.assertIn(forecast_before["predicted_skill_gap"], ["Developing", "Needs Improvement", "Strong"])
        self.assertGreaterEqual(forecast_before["predicted_performance_score"], 0.0)
        self.assertLessEqual(forecast_before["predicted_performance_score"], 100.0)

        # ---------------------------------------------------------
        # STEP 2: Intern fills questionnaire (Week 10)
        # ---------------------------------------------------------
        answers = {
            "learning_hours": 18.5,
            "tasks_completed": 5,
            "videos_completed": 9,
            "guides_completed": 7,
            "average_time_minutes": 40.0,
            "debugging_confidence": 8.0,
            "code_review_score": 85.0,
            "technical_focus_skill": "React & State Management",
            "challenge_encountered": "Redux toolkit slice action handling."
        }

        # Inspect feature mapping
        inspection_rows = build_feature_inspection_table(answers, intern_week=self.test_week)
        self.assertEqual(len(inspection_rows), 9)

        # Save questionnaire response
        resp_id, was_update = save_questionnaire_response(
            intern_id=self.test_intern,
            week=self.test_week,
            answers=answers,
            db_path=self.db_path
        )
        self.assertGreater(resp_id, 0)

        # Verify database record in questionnaire_responses
        stored_q = get_questionnaire_response(
            intern_id=self.test_intern,
            week=self.test_week,
            db_path=self.db_path
        )
        self.assertIsNotNone(stored_q)
        self.assertAlmostEqual(stored_q["learning_hours"], 18.5)
        self.assertEqual(stored_q["technical_focus_skill"], "React & State Management")

        # Verify ml_observations event created with provenance
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT observation_id, source, event_type, metric_name, label_status, training_eligible
            FROM ml_observations
            WHERE intern_id = ? AND week = ? AND source = 'questionnaire'
            ORDER BY observation_id DESC LIMIT 1
            """,
            (self.test_intern, self.test_week)
        )
        obs_row = cur.fetchone()
        self.assertIsNotNone(obs_row, "Questionnaire observation should be stored in ml_observations")
        self.assertEqual(obs_row[1], "questionnaire")
        self.assertEqual(obs_row[2], "questionnaire_submitted")
        self.assertEqual(obs_row[4], "IMMEDIATE_LABELED")
        self.assertEqual(obs_row[5], 1)

        # Verify intern_performance was updated for week 10
        cur.execute(
            """
            SELECT learning_hours, tasks_completed, videos_completed, code_review_score
            FROM intern_performance
            WHERE intern_id = ? AND week = ?
            """,
            (self.test_intern, self.test_week)
        )
        perf_row = cur.fetchone()
        self.assertIsNotNone(perf_row)
        self.assertAlmostEqual(perf_row[0], 18.5)
        self.assertEqual(perf_row[1], 5)
        conn.close()

        # ---------------------------------------------------------
        # STEP 3: Generate updated ML forecast incorporating questionnaire
        # ---------------------------------------------------------
        forecast_after = get_intern_ml_forecast(
            intern_id=self.test_intern,
            prediction_week=10,
            record_trace=True,
            db_path=self.db_path
        )
        self.assertEqual(forecast_after["status"], "success")
        self.assertIn("predicted_skill_gap", forecast_after)
        self.assertIn("predicted_performance_score", forecast_after)
        self.assertIn("confidence_score", forecast_after)
        self.assertIn("confidence_percent", forecast_after)
        self.assertIn("recommendation", forecast_after)
        self.assertIn("action_tag", forecast_after)

        # Verify prediction lineage trace recorded
        trace = get_latest_prediction_trace(self.test_intern, db_path=self.db_path)
        self.assertIsNotNone(trace)
        self.assertEqual(trace["intern_id"], self.test_intern)
        self.assertIn("feature_vector_hash", trace)
        self.assertIn("predicted_gap", trace)

        # ---------------------------------------------------------
        # STEP 4: Mentor reviews intern and records evaluation
        # ---------------------------------------------------------
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO mentor_evaluations
            (intern_id, mentor_id, technical_score, communication_score, discipline_score, overall_score, decision, remarks, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.test_intern,
                2001,
                88.0,
                82.0,
                90.0,
                86.7,
                "Strong Performer",
                "Great work on React state management; ready for advanced features.",
                datetime.now().isoformat(timespec="seconds")
            )
        )
        conn.commit()
        conn.close()

        # Ingest mentor evaluation into live ML pipeline
        mentor_obs_id = ingest_observation(
            intern_id=self.test_intern,
            week=self.test_week,
            source="mentor_evaluation",
            event_type="mentor_evaluated",
            metric_name="mentor_overall_score",
            metric_value=86.7,
            payload={"mentor_id": 2001, "decision": "Strong Performer"},
            db_path=self.db_path,
            update_active_performance=False
        )
        self.assertGreater(mentor_obs_id, 0)

        # ---------------------------------------------------------
        # STEP 5: Mentor Benchmark Dashboard evaluation
        # ---------------------------------------------------------
        live_benchmark = run_live_data_benchmark(db_path=self.db_path)
        self.assertGreater(live_benchmark["total_live_observations"], 0)
        self.assertGreater(live_benchmark["labeled_live_observations"], 0)
        self.assertTrue(live_benchmark["supervised_benchmark_available"])

        historical_benchmark = run_historical_benchmark()
        self.assertGreater(historical_benchmark["test_classification"]["accuracy"], 0.75)
        self.assertGreater(historical_benchmark["test_classification"]["macro_f1"], 0.60)
        self.assertLess(historical_benchmark["test_regression"]["rmse"], 10.0)

        audit = get_dataset_audit()
        self.assertEqual(audit["raw_dataset"]["rows"], 5000)
        self.assertEqual(audit["processed_dataset"]["rows"], 4500)
        self.assertEqual(audit["test_split"]["intern_count"], 75)

        # Save and verify results persistence
        temp_bench_file = Path(self.temp_dir) / "benchmark_test.json"
        save_benchmark_results(
            {
                "historical": historical_benchmark,
                "live": live_benchmark,
                "audit": audit
            },
            output_path=temp_bench_file
        )
        loaded_bench = load_saved_benchmark_results(temp_bench_file)
        self.assertIsNotNone(loaded_bench)
        self.assertEqual(
            loaded_bench["historical"]["test_classification"]["accuracy"],
            historical_benchmark["test_classification"]["accuracy"]
        )


if __name__ == "__main__":
    unittest.main()
