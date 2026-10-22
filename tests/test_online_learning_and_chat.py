"""
SkillTrack-ML: Online Learning, Dual Ingestion, Resources, and Chat Service Tests.
Verifies Part 1 to Part 37 requirements using standard unittest framework.
"""

import unittest
import sqlite3
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.data.live_data_ingestion import (
    ingest_observation,
    create_pending_forecasting_sample,
    resolve_pending_forecasting_labels,
    get_observation_summary,
    get_recent_dataset_records
)
from src.ml.online_learner import OnlineEventLearner
from src.data.chat_service import (
    send_chat_message,
    get_conversation_history,
    mark_messages_as_read,
    get_mentor_unread_counts,
    get_chat_engagement_metadata
)
from src.data.resource_service import (
    get_curated_resources,
    get_available_filters,
    get_recommended_resources
)

BASE_DIR = Path(__file__).resolve().parents[1]
REAL_DB_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


class TestOnlineLearningAndPlatform(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_skilltrack.db"
        shutil.copy2(REAL_DB_PATH, self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_immediate_supervised_observation(self):
        """Verifies immediate event-level supervised row creation (Part 2, Part 6)."""
        initial_summary = get_observation_summary(db_path=self.db_path)

        obs_id = ingest_observation(
            intern_id=10001,
            week=10,
            source="quiz",
            event_type="mcq_attempted",
            metric_name="mcq_score",
            metric_value=88.5,
            payload={"topic": "Java Collections", "score": 88.5},
            db_path=self.db_path,
            label_status="IMMEDIATE_LABELED",
            training_eligible=1,
            outcome_value=88.5
        )

        self.assertGreater(obs_id, 0)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT label_status, training_eligible, outcome_value FROM ml_observations WHERE observation_id = ?",
            (obs_id,)
        )
        row = cur.fetchone()
        conn.close()

        self.assertEqual(row[0], "IMMEDIATE_LABELED")
        self.assertEqual(row[1], 1)
        self.assertEqual(row[2], 88.5)

        # Verify dataset counter grew dynamically by 1
        new_summary = get_observation_summary(db_path=self.db_path)
        self.assertEqual(
            new_summary["total_active_observations"],
            initial_summary["total_active_observations"] + 1
        )

    def test_pending_forecasting_sample_no_leakage(self):
        """Verifies forecasting samples are created immediately but remain pending (Part 3, Part 6, Part 32)."""
        obs_id = create_pending_forecasting_sample(
            intern_id=10001,
            origin_week=10,
            db_path=self.db_path 
        )

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT label_status, training_eligible, outcome_value, target_week FROM ml_observations WHERE observation_id = ?",
            (obs_id,)
        )
        row = cur.fetchone()
        conn.close()

        self.assertEqual(row[0], "PENDING_FUTURE_LABEL")
        self.assertEqual(row[1], 0)  # NOT training eligible
        self.assertIsNone(row[2])  # Outcome UNKNOWN - no label fabrication!
        self.assertEqual(row[3], 11)

    def test_automated_label_resolution(self):
        """Verifies that pending forecasting samples resolve when target week arrives (Part 35)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO ml_observations (
                intern_id, week, timestamp, source, event_type,
                metric_name, metric_value, payload_json, created_at,
                label_status, training_eligible, outcome_value, target_week
            ) VALUES (10001, 9, '2026-10-07T10:00:00', 'application', 'weekly_forecast',
                      'performance_forecast_w_plus_1', NULL, '{}', '2026-10-07T10:00:00',
                      'PENDING_FUTURE_LABEL', 0, NULL, 10)
        """)
        pending_id = cur.lastrowid
        conn.commit()
        conn.close()

        resolved_count = resolve_pending_forecasting_labels(db_path=self.db_path)
        self.assertGreaterEqual(resolved_count, 1)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT label_status, training_eligible, outcome_value FROM ml_observations WHERE observation_id = ?",
            (pending_id,)
        )
        row = cur.fetchone()
        conn.close()

        self.assertEqual(row[0], "LABELED")
        self.assertEqual(row[1], 1)
        self.assertIsNotNone(row[2])
        self.assertGreater(row[2], 0.0)

    def test_online_learner_incremental_update(self):
        """Verifies online model updates incrementally with partial_fit() without full retraining (Part 10, Part 11)."""
        model_file = Path(self.temp_dir) / "test_online_model.pkl"
        learner = OnlineEventLearner(model_path=model_file)
        initial_samples = learner.cumulative_samples

        res = learner.update_with_event(
            intern_id=10001,
            outcome_value=92.0,
            event_type="mcq_attempted",
            db_path=self.db_path
        )

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["cumulative_samples"], initial_samples + 1)
        self.assertEqual(learner.cumulative_samples, initial_samples + 1)
        self.assertTrue(model_file.exists())

    def test_chat_persistence_and_unread(self):
        """Verifies intern to mentor messaging, persistence, and unread management (Part 22-25)."""
        msg_id = send_chat_message(
            sender_user_id=1,
            receiver_user_id=2,
            intern_id=10001,
            mentor_id=1,
            message="Hello mentor, I have a question about Java OOP polymorphism.",
            db_path=self.db_path
        )
        self.assertGreater(msg_id, 0)

        # Mentor checks unread
        unread_counts = get_mentor_unread_counts(mentor_user_id=2, db_path=self.db_path)
        self.assertGreaterEqual(unread_counts.get(10001, 0), 1)

        # Mentor views conversation
        history = get_conversation_history(
            intern_id=10001,
            viewer_user_id=2,
            viewer_role="mentor",
            db_path=self.db_path
        )
        self.assertGreaterEqual(len(history), 1)
        self.assertTrue(any("polymorphism" in m["message"] for m in history))

        # Mentor marks as read
        affected = mark_messages_as_read(intern_id=10001, reader_user_id=2, db_path=self.db_path)
        self.assertGreaterEqual(affected, 1)

        # Mentor replies
        reply_id = send_chat_message(
            sender_user_id=2,
            receiver_user_id=1,
            intern_id=10001,
            mentor_id=1,
            message="Polymorphism allows objects of different classes to be treated through a common interface.",
            db_path=self.db_path
        )
        self.assertGreater(reply_id, 0)

        # Intern views history
        intern_history = get_conversation_history(
            intern_id=10001,
            viewer_user_id=1,
            viewer_role="intern",
            db_path=self.db_path
        )
        self.assertGreaterEqual(len(intern_history), 2)

    def test_chat_privacy_boundaries(self):
        """Verifies that intern A cannot view intern B's messages (Part 27)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO users (username, password, role, intern_id, mentor_id)
            VALUES ('intern2', 'password123', 'intern', 10002, NULL)
        """)
        other_user_id = cur.lastrowid
        conn.commit()
        conn.close()

        send_chat_message(1, 2, 10001, 1, "Private intern 10001 note", db_path=self.db_path)

        with self.assertRaises(PermissionError):
            get_conversation_history(
                intern_id=10001,
                viewer_user_id=other_user_id,
                viewer_role="intern",
                db_path=self.db_path
            )

    def test_curated_learning_resources_and_recommendations(self):
        """Verifies curated resources and deterministic skill-gap recommendation (Part 18, Part 20, Part 21)."""
        resources = get_curated_resources(db_path=self.db_path)
        self.assertGreaterEqual(len(resources), 20)

        for res in resources:
            self.assertTrue(res["title"])
            self.assertTrue(res["url"].startswith("http"))
            self.assertTrue(res["category"])
            self.assertTrue(res["difficulty"])

        java_basics = get_curated_resources(category="Java Basics", db_path=self.db_path)
        self.assertGreaterEqual(len(java_basics), 2)

        recs = get_recommended_resources(intern_id=10001, db_path=self.db_path)
        self.assertTrue(recs["gap_label"])
        self.assertGreaterEqual(len(recs["recommendations"]), 1)


if __name__ == "__main__":
    unittest.main()
