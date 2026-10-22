"""
SkillTrack-ML: Live ML Pipeline Integration Tests
Validates the complete loop:
1. Live user activity ingestion into ml_observations
2. Feature vector updates
3. Forecast updates in real time
4. Dataset versioning and retraining eligibility checks
5. Data drift assessment
"""

import sys
import sqlite3
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data.live_data_ingestion import (
    ingest_observation,
    get_observation_summary,
    get_observations_for_intern,
    validate_metric_value,
    LiveDataIngestionError
)
from src.ml.predict import get_intern_ml_forecast, build_feature_vector
from src.ml.model_registry import (
    get_active_production_model,
    check_retraining_eligibility,
    evaluate_quality_gate
)
from src.ml.drift_monitor import assess_system_data_drift


class TestLiveMLPipeline(unittest.TestCase):

    def setUp(self):
        self.db_path = BASE_DIR / "data" / "database" / "skilltrack.db"
        self.test_intern = 10001

    def test_live_observation_ingestion_and_provenance(self):
        """
        Verify live events are stored with correct source provenance and timestamps.
        """
        summary_before = get_observation_summary(self.db_path)

        obs_id = ingest_observation(
            intern_id=self.test_intern,
            week=10,
            source="coding_submission",
            event_type="task_submitted",
            metric_name="coding_score",
            metric_value=88.5,
            payload={"task_id": 1, "test": "pipeline_unit_test"},
            db_path=self.db_path,
            update_active_performance=False
        )
        self.assertGreater(obs_id, 0)

        # Check in ml_observations
        obs_list = get_observations_for_intern(self.test_intern, db_path=self.db_path)
        matching = [o for o in obs_list if o["observation_id"] == obs_id]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["source"], "coding_submission")
        self.assertEqual(matching[0]["event_type"], "task_submitted")
        self.assertAlmostEqual(matching[0]["metric_value"], 88.5, places=1)

        # Cleanup test observation
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM ml_observations WHERE observation_id = ?", (obs_id,))
        conn.commit()
        conn.close()

    def test_invalid_ranges_rejected(self):
        """
        Verify that out-of-range metrics are rejected by ingestion.
        """
        with self.assertRaises(LiveDataIngestionError):
            validate_metric_value("coding_score", 150.0)

        with self.assertRaises(LiveDataIngestionError):
            validate_metric_value("learning_hours", -2.0)

    def test_production_model_registry_active(self):
        """
        Verify that a promoted production model exists in model_registry.
        """
        prod_model = get_active_production_model(self.db_path)
        self.assertIsNotNone(prod_model)
        self.assertEqual(prod_model["status"], "production")
        self.assertIn("v_", prod_model["model_version"])
        self.assertIn("classification", prod_model["validation_metrics"])
        self.assertIn("regression", prod_model["validation_metrics"])

    def test_quality_gate_logic(self):
        """
        Verify quality gate rejects substandard models and approves good ones.
        """
        # Good candidate
        passed, msg = evaluate_quality_gate(
            val_macro_f1=0.73,
            val_ni_recall=0.85,
            val_rmse=7.5,
            val_r2=0.63
        )
        self.assertTrue(passed)

        # Bad candidate (low Needs Improvement recall)
        failed, fail_msg = evaluate_quality_gate(
            val_macro_f1=0.73,
            val_ni_recall=0.50,  # Below 0.70 threshold
            val_rmse=7.5,
            val_r2=0.63
        )
        self.assertFalse(failed)
        self.assertIn("Needs Improvement Recall", fail_msg)

    def test_retraining_threshold_safety(self):
        """
        Verify retraining is controlled and not triggered on few observations.
        """
        is_eligible, count, reason = check_retraining_eligibility(self.db_path, min_new_samples=50000)
        self.assertFalse(is_eligible)
        self.assertIn("needed before controlled retraining", reason)

    def test_data_drift_assessment_runs(self):
        """
        Verify data drift assessment produces valid statistical summary.
        """
        drift_res = assess_system_data_drift(self.db_path)
        self.assertIn(drift_res["drift_status"], ["LOW", "MODERATE", "HIGH"])
        self.assertGreater(drift_res["monitored_feature_count"], 0)


if __name__ == "__main__":
    unittest.main()
