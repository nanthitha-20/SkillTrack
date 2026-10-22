"""
SkillTrack-ML: Online / Incremental Learning Engine
Provides continuous model adaptation using scikit-learn SGD estimators with partial_fit().
Operates on immediate event-level supervised observations without modifying
the batch production forecasting baseline.
"""

import sys
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODELS_DIR = BASE_DIR / "models"
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"
ONLINE_MODEL_PATH = MODELS_DIR / "online_event_model.pkl"

# Event feature keys used for online incremental updates
ONLINE_FEATURE_NAMES = [
    "coding_score",
    "mcq_score",
    "debugging_score",
    "code_review_score",
    "engagement_rate",
    "compile_error_rate",
    "runtime_error_rate",
    "learning_hours",
    "coding_attempts",
    "mcq_attempts"
]


class OnlineEventLearner:
    """
    Incremental learner using SGDRegressor.
    Learns continuously from immediate event outcomes (quiz scores, coding evaluations).
    """

    def __init__(self, model_path: Path = ONLINE_MODEL_PATH):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.model = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=0.001,
            learning_rate="adaptive",
            eta0=0.01,
            random_state=42
        )
        self.cumulative_samples = 0
        self.is_initialized = False
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        if self.model_path.exists():
            try:
                data = joblib.load(self.model_path)
                self.model = data["model"]
                self.scaler = data["scaler"]
                self.cumulative_samples = data.get("cumulative_samples", 0)
                self.is_initialized = data.get("is_initialized", True)
                return
            except Exception:
                pass

        # Warm start initialize on synthetic baseline mean distributions
        # Create small calibration batch so partial_fit has initial weights
        X_calib = np.array([
            [70.0, 80.0, 75.0, 79.0, 0.75, 0.14, 0.09, 4.2, 3, 2],
            [50.0, 60.0, 55.0, 65.0, 0.50, 0.30, 0.20, 2.0, 2, 1],
            [90.0, 95.0, 90.0, 92.0, 0.90, 0.05, 0.02, 6.0, 4, 3],
            [35.0, 45.0, 40.0, 50.0, 0.35, 0.40, 0.30, 1.5, 1, 1],
        ], dtype=float)
        y_calib = np.array([76.0, 58.0, 92.0, 42.0], dtype=float)

        self.scaler.partial_fit(X_calib)
        X_scaled = self.scaler.transform(X_calib)
        self.model.partial_fit(X_scaled, y_calib)
        self.cumulative_samples = len(X_calib)
        self.is_initialized = True
        self.save()

    def save(self) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "cumulative_samples": self.cumulative_samples,
            "is_initialized": self.is_initialized,
            "version": f"online_sgd_v{self.cumulative_samples}",
            "updated_at": datetime.now().isoformat(timespec="seconds")
        }, self.model_path)

    def extract_event_features(self, intern_id: int, db_path: Path = DATABASE_PATH) -> np.ndarray:
        """
        Extracts 10 immediate numeric features from the intern's active week in SQLite.
        """
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT coding_score, mcq_score, debugging_score, code_review_score,
                   engagement_rate, compile_error_rate, runtime_error_rate,
                   learning_hours, coding_attempts, mcq_attempts
            FROM intern_performance
            WHERE intern_id = ?
            ORDER BY week DESC
            LIMIT 1
        """, (intern_id,))
        row = cur.fetchone()
        conn.close()

        if not row:
            # Fallback default features
            return np.array([[70.0, 80.0, 75.0, 78.0, 0.70, 0.15, 0.10, 4.0, 3, 2]], dtype=float)

        # Replace any None with default mean
        vals = [float(v) if v is not None else 70.0 for v in row]
        return np.array([vals], dtype=float)

    def update_with_event(
        self,
        intern_id: int,
        outcome_value: float,
        event_type: str,
        db_path: Path = DATABASE_PATH,
        observation_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Incrementally updates the online model with a new immediately labeled observation.
        """
        X = self.extract_event_features(intern_id, db_path=db_path)
        y = np.array([float(outcome_value)], dtype=float)

        # Partial fit scaler and model
        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)

        version_before = f"online_sgd_v{self.cumulative_samples}"

        # Compute prior prediction to estimate step loss
        pred_before = float(self.model.predict(X_scaled)[0])
        step_loss = float((pred_before - outcome_value) ** 2)

        # Perform incremental step
        self.model.partial_fit(X_scaled, y)
        self.cumulative_samples += 1
        version_after = f"online_sgd_v{self.cumulative_samples}"
        self.save()

        now_str = datetime.now().isoformat(timespec="seconds")
        features_dict = dict(zip(ONLINE_FEATURE_NAMES, [round(float(v), 2) for v in X[0]]))
        obs_tag = f"OBS-{observation_id}" if observation_id else "None"

        try:
            m_path_str = str(self.model_path.relative_to(BASE_DIR)).replace("\\", "/")
        except ValueError:
            m_path_str = str(self.model_path).replace("\\", "/")

        # Log detailed update to SQLite
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("""
                INSERT INTO online_learning_updates (
                    timestamp, model_name, event_type, samples_processed,
                    cumulative_samples, current_loss, model_path,
                    model_version_before, model_version_after, observation_ids,
                    training_row_count, loss, features_used, target_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_str,
                "SGDRegressor (Online)",
                event_type,
                1,
                self.cumulative_samples,
                round(step_loss, 4),
                m_path_str,
                version_before,
                version_after,
                obs_tag,
                1,
                round(step_loss, 4),
                json.dumps(features_dict),
                float(outcome_value)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            import traceback
            traceback.print_exc()

        return {
            "status": "success",
            "samples_processed": 1,
            "cumulative_samples": self.cumulative_samples,
            "step_loss": round(step_loss, 4),
            "updated_at": now_str,
            "model_version_before": version_before,
            "model_version_after": version_after,
            "model_version": version_after,
            "observation_id": obs_tag
        }

    def predict_event_outcome(self, intern_id: int, db_path: Path = DATABASE_PATH) -> float:
        """
        Predicts continuous performance capability using the online model.
        """
        X = self.extract_event_features(intern_id, db_path=db_path)
        X_scaled = self.scaler.transform(X)
        pred = float(self.model.predict(X_scaled)[0])
        return max(0.0, min(100.0, round(pred, 2)))


# Singleton instance
_ONLINE_LEARNER_INSTANCE: Optional[OnlineEventLearner] = None


def get_online_learner() -> OnlineEventLearner:
    global _ONLINE_LEARNER_INSTANCE
    if _ONLINE_LEARNER_INSTANCE is None:
        _ONLINE_LEARNER_INSTANCE = OnlineEventLearner()
    return _ONLINE_LEARNER_INSTANCE


def get_online_learning_status(db_path: Path = DATABASE_PATH) -> Dict[str, Any]:
    """
    Returns diagnostic metrics of the online incremental learning model.
    """
    learner = get_online_learner()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT count(*), MAX(timestamp), AVG(current_loss) FROM online_learning_updates")
    row = cur.fetchone()
    update_count = row[0] or 0
    latest_ts = row[1]
    avg_loss = round(float(row[2]), 4) if row[2] is not None else 0.0

    conn.close()

    return {
        "online_algorithm": "SGDRegressor (Adaptive Learning Rate, L2)",
        "online_model_version": f"online_sgd_v{learner.cumulative_samples}",
        "total_online_updates": update_count,
        "cumulative_samples": learner.cumulative_samples,
        "last_online_update": latest_ts or "Not yet updated",
        "average_step_loss": avg_loss,
        "model_artifact": str(ONLINE_MODEL_PATH.relative_to(BASE_DIR)).replace("\\", "/")
    }
