"""
SkillTrack-ML: Automated Temporal Leakage Test Suite
Validates that feature extraction strictly adheres to temporal boundaries (t <= T).
Guarantees that information from future weeks or post-prediction timestamps
never leaks into historical feature vectors or earlier forecasts.
"""

import sys
import sqlite3
import unittest
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.predict import build_feature_vector, get_intern_ml_forecast
from src.data.live_data_ingestion import ingest_observation, get_observations_for_intern


class TestTemporalLeakage(unittest.TestCase):
    """
    Formal tests verifying zero forward-looking data leakage.
    """

    def setUp(self):
        self.db_path = BASE_DIR / "data" / "database" / "skilltrack.db"
        self.test_intern_id = 10001

    def test_sql_temporal_filter_enforces_strict_cutoff(self):
        """
        Verify that querying features for Week W only uses records where week <= W.
        """
        origin_week = 4
        f_vec, origin, skill, err = build_feature_vector(
            intern_id=self.test_intern_id,
            prediction_week=origin_week,
            db_path=self.db_path
        )
        self.assertIsNone(err, f"Feature extraction failed: {err}")
        self.assertIsNotNone(f_vec)
        self.assertEqual(origin, origin_week)

        # The week feature itself must equal origin_week
        self.assertEqual(int(f_vec["week"].iloc[0]), origin_week)

        # Cumulative learning hours up to week 4 must strictly match sum of weeks 1..4
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT SUM(learning_hours) FROM intern_performance WHERE intern_id = ? AND week <= ?",
            (self.test_intern_id, origin_week)
        )
        expected_cum_hours = cur.fetchone()[0]
        conn.close()

        actual_cum_hours = float(f_vec["cumulative_learning_hours"].iloc[0])
        self.assertAlmostEqual(actual_cum_hours, expected_cum_hours, places=2)

    def test_future_activity_does_not_alter_past_forecast(self):
        """
        Critical Test: Adding future activity at Week 10 MUST NOT alter
        an earlier historical prediction for Week 3 or Week 4.
        """
        # 1. Baseline prediction at Week 3
        forecast_w3_before = get_intern_ml_forecast(
            intern_id=self.test_intern_id,
            prediction_week=3,
            db_path=self.db_path
        )
        self.assertEqual(forecast_w3_before["status"], "success")

        # 2. Ingest mock future observation at Week 10
        obs_id = ingest_observation(
            intern_id=self.test_intern_id,
            week=10,
            source="coding_submission",
            event_type="task_submitted",
            metric_name="coding_score",
            metric_value=99.0,
            payload={"test_leakage": True},
            timestamp="2026-10-07T23:59:59",
            db_path=self.db_path,
            update_active_performance=False
        )

        # 3. Forecast at Week 3 after future activity
        forecast_w3_after = get_intern_ml_forecast(
            intern_id=self.test_intern_id,
            prediction_week=3,
            db_path=self.db_path
        )

        # Clean up test observation
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM ml_observations WHERE observation_id = ?", (obs_id,))
        conn.commit()
        conn.close()

        # Predictions at Week 3 must be 100% IDENTICAL
        self.assertEqual(
            forecast_w3_before["predicted_skill_gap"],
            forecast_w3_after["predicted_skill_gap"],
            "Future activity at Week 10 leaked into Week 3 skill gap prediction!"
        )
        self.assertAlmostEqual(
            forecast_w3_before["predicted_performance_score"],
            forecast_w3_after["predicted_performance_score"],
            places=4,
            msg="Future activity at Week 10 leaked into Week 3 performance score!"
        )

    def test_no_target_columns_in_feature_vector(self):
        """
        Verifies that no target label or direct ground truth proxy exists in the feature vector.
        """
        f_vec, _, _, err = build_feature_vector(
            intern_id=self.test_intern_id,
            prediction_week=5,
            db_path=self.db_path
        )
        self.assertIsNone(err)

        forbidden_leakage_columns = [
            "target_skill_gap",
            "target_performance_score",
            "target_resource_action",
            "target_skill_score"
        ]
        for col in forbidden_leakage_columns:
            self.assertNotIn(col, f_vec.columns, f"Target column '{col}' leaked into feature vector!")

    def test_observation_temporal_query_respects_cutoff(self):
        """
        Verifies get_observations_for_intern respects max_timestamp and max_week.
        """
        obs = get_observations_for_intern(
            intern_id=self.test_intern_id,
            max_week=9,
            db_path=self.db_path
        )
        for o in obs:
            self.assertLessEqual(o["week"], 9, "Observation week exceeded max_week cutoff!")


if __name__ == "__main__":
    unittest.main()
