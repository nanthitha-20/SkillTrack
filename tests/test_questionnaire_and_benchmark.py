"""
SkillTrack-ML: Questionnaire and Benchmark Test Suite
Validates:
1. Questionnaire validation and error handling
2. Feature extraction and inspection table construction
3. Database persistence and upsert semantics
4. Live ML pipeline observation ingestion with provenance
5. Benchmark engine (historical, live, dataset audit, save/load)
"""

import sys
import sqlite3
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data.questionnaire_service import (
    QUESTIONNAIRE_QUESTIONS,
    validate_questionnaire_response,
    extract_ml_features,
    build_feature_inspection_table,
    save_questionnaire_response,
    get_questionnaire_response,
    get_all_questionnaire_responses,
    get_questionnaire_summary_stats,
    QuestionnaireValidationError,
)
from src.ml.benchmark import (
    run_historical_benchmark,
    run_live_data_benchmark,
    get_dataset_audit,
    save_benchmark_results,
    load_saved_benchmark_results,
    BENCHMARK_RESULTS_PATH,
)


class TestQuestionnaireValidation(unittest.TestCase):
    """Unit tests for questionnaire validation logic."""

    def setUp(self):
        self.valid_response = {
            "learning_hours": 15.5,
            "tasks_completed": 4,
            "videos_completed": 8,
            "guides_completed": 6,
            "average_time_minutes": 45.0,
            "debugging_confidence": 7.5,
            "code_review_score": 82.0,
            "technical_focus_skill": "React & State Management",
            "challenge_encountered": "Understanding component lifecycle and hooks."
        }

    def test_valid_response_passes(self):
        # Valid response should pass validation without error
        try:
            validate_questionnaire_response(self.valid_response)
        except QuestionnaireValidationError:
            self.fail("validate_questionnaire_response raised QuestionnaireValidationError unexpectedly!")

    def test_out_of_range_slider_rejected_high(self):
        bad_response = dict(self.valid_response)
        bad_response["learning_hours"] = 80.0  # max is 60.0
        with self.assertRaises(QuestionnaireValidationError) as ctx:
            validate_questionnaire_response(bad_response)
        self.assertIn("learning_hours", str(ctx.exception))

    def test_out_of_range_slider_rejected_low(self):
        bad_response = dict(self.valid_response)
        bad_response["debugging_confidence"] = -1.0  # min is 1.0
        with self.assertRaises(QuestionnaireValidationError) as ctx:
            validate_questionnaire_response(bad_response)
        self.assertIn("debugging_confidence", str(ctx.exception))

    def test_invalid_skill_option_rejected(self):
        bad_response = dict(self.valid_response)
        bad_response["technical_focus_skill"] = "Quantum Computing"
        with self.assertRaises(QuestionnaireValidationError) as ctx:
            validate_questionnaire_response(bad_response)
        self.assertIn("technical_focus_skill", str(ctx.exception))

    def test_challenge_exceeding_max_length_rejected(self):
        bad_response = dict(self.valid_response)
        bad_response["challenge_encountered"] = "A" * 600  # max is 500
        with self.assertRaises(QuestionnaireValidationError) as ctx:
            validate_questionnaire_response(bad_response)
        self.assertIn("challenge_encountered", str(ctx.exception))

    def test_missing_field_rejected(self):
        bad_response = dict(self.valid_response)
        del bad_response["code_review_score"]
        with self.assertRaises(QuestionnaireValidationError) as ctx:
            validate_questionnaire_response(bad_response)
        self.assertIn("code_review_score", str(ctx.exception))


class TestQuestionnaireFeatureExtraction(unittest.TestCase):
    """Tests for extracting ML features from questionnaire responses."""

    def setUp(self):
        self.response = {
            "learning_hours": 12.0,
            "tasks_completed": 3,
            "videos_completed": 5,
            "guides_completed": 4,
            "average_time_minutes": 50.0,
            "debugging_confidence": 8.0,
            "code_review_score": 85.0,
            "technical_focus_skill": "JavaScript Core",
            "challenge_encountered": "Optimizing array transformations and closures."
        }

    def test_extract_ml_features_keys_and_values(self):
        features = extract_ml_features(self.response, intern_week=8)
        self.assertIn("learning_hours", features)
        self.assertIn("tasks_completed", features)
        self.assertIn("videos_completed", features)
        self.assertIn("guides_completed", features)
        self.assertIn("average_time_minutes", features)
        self.assertIn("debugging_score", features)
        self.assertIn("code_review_score", features)
        self.assertEqual(features["learning_hours"], 12.0)
        self.assertEqual(features["videos_completed"], 5)
        self.assertEqual(features["guides_completed"], 4)
        self.assertEqual(features["average_time_minutes"], 50.0)
        self.assertEqual(features["debugging_score"], 8.0)
        self.assertEqual(features["code_review_score"], 85.0)

    def test_build_feature_inspection_table(self):
        table_rows = build_feature_inspection_table(self.response, intern_week=8)
        self.assertEqual(len(table_rows), len(QUESTIONNAIRE_QUESTIONS))
        first_row = table_rows[0]
        self.assertIn("Question", first_row)
        self.assertIn("Raw Input", first_row)
        self.assertIn("ML Feature", first_row)
        self.assertIn("Transformed Value", first_row)
        self.assertIn("Feature Type", first_row)
        self.assertIn("Used in Prediction", first_row)


class TestQuestionnairePersistenceAndIngestion(unittest.TestCase):
    """Tests for saving, upserting, retrieving questionnaire responses and ML ingestion."""

    def setUp(self):
        self.db_path = BASE_DIR / "data" / "database" / "skilltrack.db"
        self.test_intern = 10001
        self.test_week = 10
        self.valid_response = {
            "learning_hours": 18.0,
            "tasks_completed": 5,
            "videos_completed": 10,
            "guides_completed": 7,
            "average_time_minutes": 42.0,
            "debugging_confidence": 8.5,
            "code_review_score": 88.0,
            "technical_focus_skill": "JavaScript Core",
            "challenge_encountered": "Fine-tuning regression hyperparameters."
        }

    def test_save_and_retrieve_questionnaire(self):
        resp_id, was_update = save_questionnaire_response(
            intern_id=self.test_intern,
            week=self.test_week,
            answers=self.valid_response,
            db_path=self.db_path
        )
        self.assertGreater(resp_id, 0)

        retrieved = get_questionnaire_response(
            intern_id=self.test_intern,
            week=self.test_week,
            db_path=self.db_path
        )
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["intern_id"], self.test_intern)
        self.assertEqual(retrieved["week"], self.test_week)
        self.assertAlmostEqual(retrieved["learning_hours"], 18.0)
        self.assertEqual(retrieved["tasks_completed"], 5)
        self.assertEqual(retrieved["technical_focus_skill"], "JavaScript Core")

    def test_upsert_questionnaire_response(self):
        # Save first version
        id1, _ = save_questionnaire_response(
            intern_id=self.test_intern,
            week=self.test_week,
            answers=self.valid_response,
            db_path=self.db_path
        )
        # Update with new values for same intern and week
        updated_data = dict(self.valid_response)
        updated_data["learning_hours"] = 22.5
        id2, was_update = save_questionnaire_response(
            intern_id=self.test_intern,
            week=self.test_week,
            answers=updated_data,
            db_path=self.db_path
        )
        self.assertEqual(id1, id2)
        self.assertTrue(was_update)

        retrieved = get_questionnaire_response(
            intern_id=self.test_intern,
            week=self.test_week,
            db_path=self.db_path
        )
        self.assertAlmostEqual(retrieved["learning_hours"], 22.5)

    def test_questionnaire_creates_live_ml_observation(self):
        save_questionnaire_response(
            intern_id=self.test_intern,
            week=self.test_week,
            answers=self.valid_response,
            db_path=self.db_path
        )

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT observation_id, source, event_type, metric_name, label_status, training_eligible
            FROM ml_observations
            WHERE intern_id = ? AND week = ? AND source = 'questionnaire'
            ORDER BY observation_id DESC LIMIT 1
            """,
            (self.test_intern, self.test_week)
        )
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row, "Expected ml_observations record for questionnaire")
        obs_id, source, event_type, metric_name, label_status, training_eligible = row
        self.assertEqual(source, "questionnaire")
        self.assertEqual(event_type, "questionnaire_submitted")
        self.assertEqual(metric_name, "questionnaire_score")
        self.assertEqual(label_status, "IMMEDIATE_LABELED")
        self.assertEqual(training_eligible, 1)

    def test_summary_stats_and_all_responses(self):
        # Ensure at least 1 response exists
        save_questionnaire_response(
            intern_id=self.test_intern,
            week=self.test_week,
            answers=self.valid_response,
            db_path=self.db_path
        )

        stats = get_questionnaire_summary_stats(db_path=self.db_path)
        self.assertIn("total_responses", stats)
        self.assertIn("unique_interns", stats)
        self.assertGreaterEqual(stats["total_responses"], 1)

        all_resps = get_all_questionnaire_responses(intern_id=self.test_intern, db_path=self.db_path)
        self.assertGreaterEqual(len(all_resps), 1)


class TestBenchmarkEngine(unittest.TestCase):
    """Tests for the benchmark engine modules and auditing."""

    def test_historical_benchmark_schema_and_metrics(self):
        result = run_historical_benchmark()
        self.assertEqual(result["evaluation_type"], "historical_held_out")
        self.assertIn("test_classification", result)
        self.assertIn("test_regression", result)
        self.assertIn("baseline_classification", result)
        self.assertIn("temporal_leakage_check", result)

        cls_res = result["test_classification"]
        self.assertIn("accuracy", cls_res)
        self.assertIn("macro_f1", cls_res)
        self.assertIn("weighted_f1", cls_res)
        self.assertIn("balanced_accuracy", cls_res)
        self.assertIn("confusion_matrix", cls_res)
        self.assertIn("per_class_precision", cls_res)
        self.assertIn("per_class_recall", cls_res)
        self.assertIn("per_class_f1", cls_res)
        self.assertGreater(cls_res["accuracy"], 0.70)
        self.assertGreater(cls_res["macro_f1"], 0.60)

        reg_res = result["test_regression"]
        self.assertIn("rmse", reg_res)
        self.assertIn("mae", reg_res)
        self.assertIn("r2", reg_res)
        self.assertLess(reg_res["rmse"], 15.0)

        # Leakage check must confirm verified with 0 train-test overlap
        leakage = result["temporal_leakage_check"]
        self.assertTrue(leakage["verified"])
        self.assertEqual(leakage["train_test_overlap"], 0)

    def test_live_benchmark_schema(self):
        result = run_live_data_benchmark()
        self.assertIn("total_live_observations", result)
        self.assertIn("labeled_live_observations", result)
        self.assertIn("pending_future_label_observations", result)
        self.assertIn("supervised_benchmark_available", result)
        self.assertIn("observation_time_range", result)

    def test_dataset_audit_schema(self):
        audit = get_dataset_audit()
        self.assertIn("raw_dataset", audit)
        self.assertIn("processed_dataset", audit)
        self.assertIn("train_split", audit)
        self.assertIn("validation_split", audit)
        self.assertIn("test_split", audit)

        raw = audit["raw_dataset"]
        proc = audit["processed_dataset"]
        self.assertEqual(raw["rows"], 5000)
        self.assertEqual(proc["rows"], 4500)
        self.assertEqual(proc["feature_columns"], 43)

    def test_benchmark_save_and_load(self):
        dummy_results = {
            "historical": {"test_classification": {"accuracy": 0.8015}},
            "live": {"total_live_observations": 10},
            "audit": {"raw_dataset": {"rows": 5000}}
        }
        save_benchmark_results(dummy_results, BENCHMARK_RESULTS_PATH)
        self.assertTrue(BENCHMARK_RESULTS_PATH.exists())
        loaded = load_saved_benchmark_results(BENCHMARK_RESULTS_PATH)
        self.assertIsNotNone(loaded)
        self.assertIn("historical", loaded)
        self.assertIn("live", loaded)
        self.assertIn("audit", loaded)
        self.assertEqual(loaded["historical"]["test_classification"]["accuracy"], 0.8015)


if __name__ == "__main__":
    unittest.main()
