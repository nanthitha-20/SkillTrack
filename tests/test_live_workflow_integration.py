"""
End-to-End Workflow & Integration Test:
Verifies seamless data flow across Live ML Dataset, Live ML Lab, and End-to-End Demo.
"""

import unittest
import sqlite3
import tempfile
import shutil
from pathlib import Path

from src.data.assessment_service import (
    validate_and_save_assessment,
    get_intern_assessments,
    get_assessment_summary_stats
)
from src.ml.predict import (
    get_intern_ml_forecast,
    load_models
)
from src.ml.prediction_tracer import (
    record_prediction_trace,
    get_recent_prediction_traces,
    get_trace_by_id,
    get_traces_for_intern,
    diff_prediction_traces
)
from src.ml.hiring_engine import (
    compute_hiring_recommendation,
    save_mentor_hiring_decision,
    get_latest_hiring_decision
)
from src.ml.online_learner import get_online_learner, get_online_learning_status
from src.ml.drift_monitor import assess_system_data_drift
from src.ml.model_registry import get_active_production_model, get_all_registered_models
from src.database.migrate_assessment_and_hiring import run_assessment_and_hiring_migration

PROD_DB_PATH = Path("data/database/skilltrack.db")


class TestCompleteWorkflowIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_flow.db"
        shutil.copy(PROD_DB_PATH, self.db_path)
        run_assessment_and_hiring_migration(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_e2e_lifecycle(self):
        intern_id = 10001
        mentor_id = 1

        # 1. Baseline prediction before assessment
        pred_before = get_intern_ml_forecast(intern_id, db_path=self.db_path)
        score_1 = pred_before.get("predicted_performance_score") or pred_before.get("predicted_performance")
        self.assertIsNotNone(score_1)
        self.assertIn("predicted_skill_gap", pred_before)

        # 2. Record initial baseline trace
        trace_id_1 = record_prediction_trace(
            intern_id=intern_id,
            predicted_gap=pred_before["predicted_skill_gap"],
            predicted_performance=score_1,
            confidence_score=pred_before.get("confidence_score", 0.85),
            features_dict=pred_before.get("features", {}),
            model_version=pred_before.get("model_version", "v1.0.0"),
            db_path=self.db_path
        )
        self.assertGreater(trace_id_1, 0)

        # 3. Submit real assessment via Live ML Dataset pipeline
        ass_result = validate_and_save_assessment(
            intern_id=intern_id,
            week=5,
            assessment_type="Technical Assessment",
            assessment_name="Distributed Transactions & ACID Properties",
            marks_obtained=96.0,
            max_marks=100.0,
            technical_details={
                "tasks_completed": 8,
                "test_cases_total": 12,
                "test_cases_passed": 12,
                "code_quality_score": 98.0,
                "completion_time_minutes": 35.0
            },
            db_path=self.db_path
        )
        self.assertIn("assessment_id", ass_result)
        self.assertIn("observation_id", ass_result)

        # 4. Verify observation is in ml_observations with IMMEDIATE_LABELED
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT label_status, training_eligible, outcome_value FROM ml_observations WHERE observation_id = ?",
            (ass_result["observation_id"],)
        )
        obs_row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(obs_row)
        self.assertEqual(obs_row[0], "IMMEDIATE_LABELED")
        self.assertEqual(obs_row[1], 1)
        self.assertEqual(obs_row[2], 96.0)

        # 5. Subsequent prediction after new live observation
        pred_after = get_intern_ml_forecast(intern_id, db_path=self.db_path)
        score_2 = pred_after.get("predicted_performance_score") or pred_after.get("predicted_performance")
        trace_id_2 = record_prediction_trace(
            intern_id=intern_id,
            predicted_gap=pred_after["predicted_skill_gap"],
            predicted_performance=score_2,
            confidence_score=pred_after.get("confidence_score", 0.90),
            features_dict=pred_after.get("features", {}),
            model_version=pred_after.get("model_version", "v1.0.0"),
            db_path=self.db_path
        )
        self.assertGreater(trace_id_2, trace_id_1)

        # 6. Verify trace differential analysis
        diff = diff_prediction_traces(trace_id_1, trace_id_2, db_path=self.db_path)
        self.assertNotIn("error", diff)
        self.assertIn("prediction_delta", diff)

        # 7. Compute transparent hiring recommendation
        hiring_rec = compute_hiring_recommendation(intern_id, db_path=self.db_path)
        self.assertTrue(hiring_rec["min_evidence_met"])
        self.assertIn(hiring_rec["recommendation"], ["HIRE", "EXTEND_INTERNSHIP", "DO_NOT_RECOMMEND"])
        self.assertIn("evidence_summary", hiring_rec)

        # 8. Persist formal mentor hiring authorization
        decision_id = save_mentor_hiring_decision(
            intern_id=intern_id,
            mentor_id=mentor_id,
            decision_type=hiring_rec["recommendation"],
            composite_readiness_score=hiring_rec["composite_score"],
            confidence_level=hiring_rec["confidence_level"],
            decision_rationale=hiring_rec["rationale"],
            strengths="Advanced distributed systems mastery",
            growth_areas="Mentorship capabilities",
            is_final=True,
            evidence_summary=hiring_rec["evidence_summary"],
            db_path=self.db_path
        )
        self.assertGreater(decision_id, 0)

        # 9. Verify retrieval of formal decision
        latest_dec = get_latest_hiring_decision(intern_id, db_path=self.db_path)
        self.assertIsNotNone(latest_dec)
        self.assertEqual(latest_dec["decision_type"], hiring_rec["recommendation"])
        self.assertEqual(latest_dec["decision_id"], decision_id)

        # 10. Verify Live ML Lab can inspect models and registries
        active_model = get_active_production_model(self.db_path)
        self.assertIsNotNone(active_model)
        self.assertEqual(active_model["status"], "production")

        all_models = get_all_registered_models(self.db_path)
        self.assertGreaterEqual(len(all_models), 1)


if __name__ == "__main__":
    unittest.main()
