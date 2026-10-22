"""
SkillTrack-ML: Automated Test Suite for Live ML Data Lineage & Prediction Traces.
Verifies all 14 requirements from Part 12.
"""

import unittest
import sqlite3
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime

from src.data.live_data_ingestion import (
    ingest_observation,
    create_pending_forecasting_sample
)
from src.ml.predict import get_intern_ml_forecast, build_feature_vector
from src.ml.prediction_tracer import (
    record_prediction_trace,
    get_latest_prediction_trace,
    diff_prediction_traces,
    compute_feature_vector_hash,
    get_observation_ids_for_prediction
)
from src.ml.online_learner import OnlineEventLearner, get_online_learning_status

BASE_DIR = Path(__file__).resolve().parents[1]
REAL_DB_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


class TestPredictionLineage(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_skilltrack.db"
        shutil.copy2(REAL_DB_PATH, self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_application_event_creates_observation_with_provenance_and_timestamp(self):
        """Req 1, 2, 3: New app event creates observation with provenance and ISO timestamp."""
        ts = "2026-10-07T12:00:00"
        obs_id = ingest_observation(
            intern_id=10001,
            week=10,
            source="quiz",
            event_type="mcq_attempted",
            metric_name="mcq_score",
            metric_value=85.0,
            timestamp=ts,
            db_path=self.db_path,
            label_status="IMMEDIATE_LABELED",
            training_eligible=1,
            outcome_value=85.0
        )
        self.assertGreater(obs_id, 0)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT source, timestamp, label_status FROM ml_observations WHERE observation_id = ?", (obs_id,))
        row = cur.fetchone()
        conn.close()

        self.assertEqual(row[0], "quiz")
        self.assertEqual(row[1], ts)
        self.assertEqual(row[2], "IMMEDIATE_LABELED")

    def test_temporal_causality_prediction_before_and_after_event(self):
        """Req 4, 5, 8, 9: Prediction before event excludes observation; prediction after includes it."""
        t0 = "2026-10-07T10:00:00"
        t1 = "2026-10-07T11:00:00"
        t2 = "2026-10-07T12:00:00"

        # T0: Initial prediction
        pred_a = get_intern_ml_forecast(
            intern_id=10001,
            data_cutoff_timestamp=t0,
            record_trace=True,
            db_path=self.db_path
        )
        obs_used_a = pred_a.get("observation_ids_used", [])

        # T1: Ingest new application event
        obs_id = ingest_observation(
            intern_id=10001,
            week=10,
            source="quiz",
            event_type="mcq_attempted",
            metric_name="mcq_score",
            metric_value=90.0,
            timestamp=t1,
            db_path=self.db_path,
            label_status="IMMEDIATE_LABELED",
            training_eligible=1,
            outcome_value=90.0
        )

        # T2: Prediction after event
        pred_b = get_intern_ml_forecast(
            intern_id=10001,
            data_cutoff_timestamp=t2,
            record_trace=True,
            db_path=self.db_path
        )
        obs_used_b = pred_b.get("observation_ids_used", [])

        # Verification: OBS was unavailable at T0, available at T2
        self.assertNotIn(obs_id, obs_used_a, "Future observation leaked into earlier prediction A!")
        self.assertIn(obs_id, obs_used_b, "New observation was not captured in subsequent prediction B!")
        self.assertEqual(pred_a["data_cutoff_timestamp"], t0)
        self.assertEqual(pred_b["data_cutoff_timestamp"], t2)

    def test_feature_vector_changes_only_when_appropriate(self):
        """Req 6: Feature vector changes only in affected variables."""
        f_before, _, _, _ = build_feature_vector(10001, db_path=self.db_path)
        mcq_att_before = int(f_before["mcq_attempts"].iloc[0])
        role_before = str(f_before["role"].iloc[0])

        # Submit quiz
        ingest_observation(
            intern_id=10001,
            week=10,
            source="quiz",
            event_type="mcq_attempted",
            metric_name="mcq_score",
            metric_value=95.0,
            db_path=self.db_path,
            label_status="IMMEDIATE_LABELED",
            training_eligible=1,
            outcome_value=95.0
        )

        f_after, _, _, _ = build_feature_vector(10001, db_path=self.db_path)
        mcq_att_after = int(f_after["mcq_attempts"].iloc[0])
        role_after = str(f_after["role"].iloc[0])

        # MCQ attempts must increase by exactly 1
        self.assertEqual(mcq_att_after, mcq_att_before + 1)
        # Static role must remain completely unaffected
        self.assertEqual(role_after, role_before)

    def test_prediction_trace_stores_model_version_and_source(self):
        """Req 7, 13, 14: Trace records model version, model type, and prediction source."""
        pred = get_intern_ml_forecast(intern_id=10001, record_trace=True, db_path=self.db_path)
        trace_id = pred["trace_id"]
        self.assertIsNotNone(trace_id)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT model_version, model_type, prediction_source, feature_vector_hash
            FROM prediction_traces
            WHERE trace_id = ?
        """, (trace_id,))
        row = cur.fetchone()
        conn.close()

        self.assertTrue(row[0])  # model_version exists
        self.assertEqual(row[1], "Balanced Logistic Regression + Ridge Regression")
        self.assertEqual(row[2], "production_batch")  # explicitly not online shadow
        self.assertTrue(row[3])  # feature hash exists

    def test_immediate_vs_pending_forecasting_label_isolation(self):
        """Req 10, 11: Immediate labeled row reaches online learner; pending forecast cannot be used as target."""
        # 1. Immediate labeled row
        obs_id_imm = ingest_observation(
            intern_id=10001,
            week=10,
            source="quiz",
            event_type="mcq_attempted",
            metric_name="mcq_score",
            metric_value=88.0,
            db_path=self.db_path,
            label_status="IMMEDIATE_LABELED",
            training_eligible=1,
            outcome_value=88.0
        )

        # 2. Pending forecasting row
        obs_id_pend = create_pending_forecasting_sample(
            intern_id=10001,
            origin_week=10,
            db_path=self.db_path
        )

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT label_status, training_eligible, outcome_value FROM ml_observations WHERE observation_id = ?", (obs_id_imm,))
        row_imm = cur.fetchone()
        cur.execute("SELECT label_status, training_eligible, outcome_value FROM ml_observations WHERE observation_id = ?", (obs_id_pend,))
        row_pend = cur.fetchone()
        conn.close()

        self.assertEqual(row_imm[0], "IMMEDIATE_LABELED")
        self.assertEqual(row_imm[1], 1)
        self.assertEqual(row_imm[2], 88.0)

        self.assertEqual(row_pend[0], "PENDING_FUTURE_LABEL")
        self.assertEqual(row_pend[1], 0)
        self.assertIsNone(row_pend[2])  # Outcome is NULL - strictly non-trainable

    def test_online_model_update_logged_with_lineage(self):
        """Req 12: Online learning update logs model version before/after and observation ID."""
        model_file = Path(self.temp_dir) / "test_online.pkl"
        learner = OnlineEventLearner(model_path=model_file)
        v_before = f"online_sgd_v{learner.cumulative_samples}"

        unique_obs_id = 88899
        res = learner.update_with_event(
            intern_id=10001,
            outcome_value=91.0,
            event_type="mcq_attempted",
            db_path=self.db_path,
            observation_id=unique_obs_id
        )

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["model_version_before"], v_before)
        self.assertEqual(res["observation_id"], f"OBS-{unique_obs_id}")

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT observation_ids, model_version_before, model_version_after FROM online_learning_updates WHERE observation_ids = ? ORDER BY update_id DESC LIMIT 1",
            (f"OBS-{unique_obs_id}",)
        )
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], f"OBS-{unique_obs_id}")
        self.assertEqual(row[1], v_before)
        self.assertEqual(row[2], res["model_version_after"])


if __name__ == "__main__":
    unittest.main()
