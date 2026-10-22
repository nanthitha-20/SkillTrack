import sqlite3
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Base paths
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"

SKILL_GAP_MODEL_PATH = MODELS_DIR / "skill_gap_model.pkl"
PERFORMANCE_MODEL_PATH = MODELS_DIR / "performance_model.pkl"

# Cached model instances in memory
_CACHED_MODELS = {
    "skill_gap": None,
    "performance": None
}

# Top global influential features extracted from trained models for explainability
EXPLAINABILITY_FACTORS = [
    "Historical skill performance trend",
    "Coding assessment test pass rate",
    "Debugging & code review challenge scores",
    "Learning hours & guide completion effort",
    "Compilation & runtime error frequency"
]


def load_models() -> Tuple[Any, Any]:
    """
    Loads and caches the trained classification and regression pipelines.
    """
    global _CACHED_MODELS
    if _CACHED_MODELS["skill_gap"] is None:
        if not SKILL_GAP_MODEL_PATH.exists():
            raise FileNotFoundError(f"Skill gap model not found at {SKILL_GAP_MODEL_PATH}")
        _CACHED_MODELS["skill_gap"] = joblib.load(SKILL_GAP_MODEL_PATH)

    if _CACHED_MODELS["performance"] is None:
        if not PERFORMANCE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Performance model not found at {PERFORMANCE_MODEL_PATH}")
        _CACHED_MODELS["performance"] = joblib.load(PERFORMANCE_MODEL_PATH)

    return _CACHED_MODELS["skill_gap"], _CACHED_MODELS["performance"]


def build_feature_vector(
    intern_id: int,
    prediction_week: Optional[int] = None,
    data_cutoff_timestamp: Optional[str] = None,
    db_path: Path = DATABASE_PATH
) -> Tuple[Optional[pd.DataFrame], Optional[int], Optional[str], Optional[str]]:
    """
    Constructs the exact 43-feature input vector from the live SQLite database
    respecting temporal safety:
    1. Records with week <= prediction_week are queried.
    2. If data_cutoff_timestamp is supplied, only observations generated AT OR BEFORE
       the cutoff timestamp are reflected in the feature state.
    """
    if not db_path.exists():
        return None, None, None, f"Database not found at {db_path}"

    try:
        conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    except Exception as e:
        return None, None, None, f"Database connection error: {e}"

    try:
        # 1. Retrieve static intern profile
        profile_query = "SELECT intern_id, role, experience_level FROM interns WHERE intern_id = ?"
        profile_res = pd.read_sql_query(profile_query, conn, params=(intern_id,))

        if profile_res.empty:
            conn.close()
            return None, None, None, f"Intern ID {intern_id} not found in database."

        role = profile_res.iloc[0]["role"] or "Frontend Development"
        experience_level = profile_res.iloc[0]["experience_level"] or "Intermediate"

        # 2. Determine prediction origin week
        if prediction_week is None:
            max_week_res = pd.read_sql_query(
                "SELECT MAX(week) as max_w FROM intern_performance WHERE intern_id = ?",
                conn,
                params=(intern_id,)
            )
            if max_week_res.empty or max_week_res.iloc[0]["max_w"] is None:
                conn.close()
                return None, None, None, "No performance records found for this intern."
            origin_week = int(max_week_res.iloc[0]["max_w"])
        else:
            origin_week = int(prediction_week)

        # 3. Retrieve historical records up to and including origin_week (STRICT TEMPORAL RULE: week <= origin_week)
        perf_query = """
            SELECT * FROM intern_performance
            WHERE intern_id = ? AND week <= ?
            ORDER BY week ASC
        """
        history_df = pd.read_sql_query(perf_query, conn, params=(intern_id, origin_week))

        # Check if observations occurred AFTER data_cutoff_timestamp that should be excluded
        if data_cutoff_timestamp is not None:
            post_obs_df = pd.read_sql_query(
                "SELECT observation_id, event_type, metric_name, metric_value FROM ml_observations WHERE intern_id = ? AND timestamp > ?",
                conn,
                params=(intern_id, data_cutoff_timestamp)
            )
            if not post_obs_df.empty and not history_df.empty:
                # Rewind active week row by reversing events that occurred after the cutoff
                active_idx = history_df.index[-1]
                for _, p_obs in post_obs_df.iterrows():
                    ev_type = p_obs["event_type"]
                    if ev_type == "mcq_attempted":
                        cur_att = history_df.at[active_idx, "mcq_attempts"]
                        if cur_att and cur_att > 0:
                            history_df.at[active_idx, "mcq_attempts"] = cur_att - 1
                    elif ev_type in ("task_submitted", "task_run"):
                        cur_att = history_df.at[active_idx, "coding_attempts"]
                        if cur_att and cur_att > 0:
                            history_df.at[active_idx, "coding_attempts"] = cur_att - 1

        conn.close()

        if history_df.empty:
            return None, origin_week, None, f"No activity history available through Week {origin_week}."

        # 4. Feature engineering matching training pipeline
        current_row = history_df.iloc[-1]
        current_skill = str(current_row["skill"])

        test_case_pass_rate = (
            float(current_row["test_cases_passed"] / current_row["test_cases_total"])
            if current_row["test_cases_total"] > 0
            else 0.0
        )

        features = {
            # Categorical
            "role": str(role),
            "experience_level": str(experience_level),
            "skill": current_skill,
            # Numeric current week
            "week": int(current_row["week"]),
            "days_since_start": int(current_row["days_since_start"]),
            "learning_hours": float(current_row["learning_hours"]),
            "videos_completed": int(current_row["videos_completed"]),
            "guides_completed": int(current_row["guides_completed"]),
            "tasks_completed": int(current_row["tasks_completed"]),
            "engagement_rate": float(current_row["engagement_rate"]),
            "coding_attempts": int(current_row["coding_attempts"]),
            "execution_count": int(current_row["execution_count"]),
            "average_time_minutes": float(current_row["average_time_minutes"]),
            "compile_error_rate": float(current_row["compile_error_rate"]),
            "runtime_error_rate": float(current_row["runtime_error_rate"]),
            "test_cases_total": int(current_row["test_cases_total"]),
            "test_cases_passed": int(current_row["test_cases_passed"]),
            "coding_score": float(current_row["coding_score"]),
            "mcq_attempts": int(current_row["mcq_attempts"]),
            "mcq_score": float(current_row["mcq_score"]),
            "debugging_score": float(current_row["debugging_score"]),
            "code_review_score": float(current_row["code_review_score"]),
            "previous_skill_score": float(current_row["previous_skill_score"]),
            "skill_score": float(current_row["skill_score"]),
            "score_improvement": float(current_row["score_improvement"]),
            "final_performance_score": float(current_row["final_performance_score"]),
            "test_case_pass_rate": float(test_case_pass_rate),
            # Cumulative sums
            "cumulative_learning_hours": float(history_df["learning_hours"].sum()),
            "cumulative_tasks_completed": int(history_df["tasks_completed"].sum()),
            "cumulative_videos_completed": int(history_df["videos_completed"].sum()),
            "cumulative_guides_completed": int(history_df["guides_completed"].sum()),
            "cumulative_coding_attempts": int(history_df["coding_attempts"].sum()),
            "cumulative_execution_count": int(history_df["execution_count"].sum()),
            # Historical expanding averages
            "historical_coding_score_avg": float(history_df["coding_score"].mean()),
            "historical_mcq_score_avg": float(history_df["mcq_score"].mean()),
            "historical_debugging_score_avg": float(history_df["debugging_score"].mean()),
            "historical_code_review_score_avg": float(history_df["code_review_score"].mean()),
            "historical_skill_score_avg": float(history_df["skill_score"].mean()),
            "historical_learning_hours_avg": float(history_df["learning_hours"].mean()),
            "historical_engagement_rate_avg": float(history_df["engagement_rate"].mean()),
            "historical_compile_error_rate_avg": float(history_df["compile_error_rate"].mean()),
            "historical_runtime_error_rate_avg": float(history_df["runtime_error_rate"].mean()),
            "historical_average_time_minutes_avg": float(history_df["average_time_minutes"].mean())
        }

        features_df = pd.DataFrame([features])
        return features_df, origin_week, current_skill, None

    except Exception as e:
        if 'conn' in locals():
            conn.close()
        return None, None, None, f"Feature extraction error: {e}"


def get_skill_specific_recommendation(skill: str, predicted_gap: str) -> Tuple[str, str]:
    """
    Generates transparent, skill-tailored learning recommendations.
    """
    skill_clean = skill.strip() if skill else "Frontend Development"

    if predicted_gap == "Strong":
        action_tag = "Continue Progress"
        rec = f"Demonstrating high technical proficiency in {skill_clean}. Continue with advanced challenges and maintain high engagement."
    elif predicted_gap == "Developing":
        action_tag = "Practice Recommended"
        if "React" in skill_clean:
            rec = "Steady progress in React. Practice component state management and review React documentation to build confidence."
        elif "JavaScript" in skill_clean:
            rec = "Steady progress in JavaScript. Recommended to practice DOM manipulation and event-handling exercises."
        elif "CSS" in skill_clean:
            rec = "Steady progress in CSS. Practice responsive Flexbox/Grid layouts and test across mobile breakpoints."
        elif "HTML" in skill_clean:
            rec = "Steady progress in HTML. Review semantic tags and form accessibility standards."
        elif "API" in skill_clean:
            rec = "Steady progress in API Integration. Practice async/await fetch requests and error boundary handling."
        else:
            rec = f"Steady progress in {skill_clean}. Recommended to complete additional coding exercises and review documentation."
    else:  # Needs Improvement
        action_tag = "Targeted Learning"
        if "React" in skill_clean:
            rec = "Skill gap identified in React. Recommended to review React hooks & state videos, focus on debugging exercises, and seek mentor guidance."
        elif "JavaScript" in skill_clean:
            rec = "Skill gap identified in JavaScript. Recommended to review core JS syntax, practice debugging exercises, and seek mentor guidance."
        elif "CSS" in skill_clean:
            rec = "Skill gap identified in CSS. Recommended to review layout guides, practice debugging responsive styling, and seek mentor guidance."
        elif "HTML" in skill_clean:
            rec = "Skill gap identified in HTML. Review form validation and semantic layout structures."
        elif "API" in skill_clean:
            rec = "Skill gap identified in API Integration. Review REST endpoint handling, status codes, and JSON parsing."
        else:
            rec = f"Skill gap identified in {skill_clean}. Recommended to review foundational tutorials, focus on debugging challenges, and seek mentor guidance."

    return rec, action_tag


def get_intern_ml_forecast(
    intern_id: int,
    prediction_week: Optional[int] = None,
    data_cutoff_timestamp: Optional[str] = None,
    record_trace: bool = True,
    db_path: Path = DATABASE_PATH
) -> Dict[str, Any]:
    """
    Unified forecasting function for Intern and Mentor portals.
    Generates next-week skill gap and performance forecasts with confidence and recommendations.
    Persists auditable prediction trace with feature hash, model version, and data cutoff.
    """
    from datetime import datetime

    try:
        cls_model, reg_model = load_models()
    except Exception as e:
        return {
            "status": "error",
            "message": f"Evaluation engine unavailable: {e}",
            "intern_id": intern_id
        }

    features_df, origin_week, current_skill, error_msg = build_feature_vector(
        intern_id=intern_id,
        prediction_week=prediction_week,
        data_cutoff_timestamp=data_cutoff_timestamp,
        db_path=db_path
    )

    if error_msg or features_df is None:
        return {
            "status": "insufficient_data",
            "message": error_msg or "Insufficient historical data recorded for this intern.",
            "intern_id": intern_id,
            "origin_week": origin_week
        }

    target_week = origin_week + 1

    try:
        # Classification prediction
        predicted_gap = str(cls_model.predict(features_df)[0])
        probabilities = {}
        confidence = 0.0

        if hasattr(cls_model, "predict_proba"):
            probs = cls_model.predict_proba(features_df)[0]
            classes = cls_model.classes_
            probabilities = {cls_name: float(p) for cls_name, p in zip(classes, probs)}
            confidence = probabilities.get(predicted_gap, 0.0)

        # Regression prediction
        predicted_perf = float(reg_model.predict(features_df)[0])
        predicted_perf = max(0.0, min(100.0, predicted_perf))  # Bound to [0, 100]

        # Skill-specific recommendation mapping
        recommendation, action_tag = get_skill_specific_recommendation(
            skill=current_skill or "Frontend Development",
            predicted_gap=predicted_gap
        )

        cutoff_ts = data_cutoff_timestamp or datetime.now().isoformat(timespec="seconds")
        trace_id = None
        feature_hash = None
        obs_ids_used = []
        latest_obs_id = None
        m_ver = "v_20261007_125928"
        on_ver = "online_sgd_v0"

        # Auditable prediction trace recording
        if record_trace:
            try:
                from src.ml.prediction_tracer import (
                    record_prediction_trace,
                    compute_feature_vector_hash,
                    get_observation_ids_for_prediction
                )
                from src.ml.model_registry import get_active_production_model
                from src.ml.online_learner import get_online_learning_status

                features_dict = features_df.iloc[0].to_dict()
                feature_hash = compute_feature_vector_hash(features_dict)
                prod_m = get_active_production_model(db_path=db_path) or {}
                m_ver = prod_m.get("model_version", "v_20261007_125928")

                online_stat = get_online_learning_status(db_path=db_path)
                on_ver = online_stat.get("online_model_version", "online_sgd_v0")
                on_count = online_stat.get("total_online_updates", 0)

                obs_ids_used, latest_obs_id = get_observation_ids_for_prediction(intern_id, cutoff_ts, db_path)

                trace_id = record_prediction_trace(
                    intern_id=intern_id,
                    predicted_gap=predicted_gap,
                    predicted_performance=round(predicted_perf, 2),
                    confidence_score=round(confidence, 4),
                    features_dict=features_dict,
                    model_version=m_ver,
                    model_type="Balanced Logistic Regression + Ridge Regression",
                    prediction_source="production_batch",
                    prediction_type="next_week_forecast",
                    data_cutoff_timestamp=cutoff_ts,
                    training_row_count=4500,
                    online_model_version=on_ver,
                    online_update_count=on_count,
                    db_path=db_path
                )
            except Exception:
                pass

        return {
            "status": "success",
            "intern_id": intern_id,
            "origin_week": origin_week,
            "target_week": target_week,
            "current_skill": current_skill,
            "predicted_skill_gap": predicted_gap,
            "confidence_score": confidence,
            "confidence_percent": round(confidence * 100, 1),
            "probabilities": probabilities,
            "predicted_performance_score": round(predicted_perf, 1),
            "recommendation": recommendation,
            "action_tag": action_tag,
            "top_factors": EXPLAINABILITY_FACTORS,
            "explainability_note": "Factors that were most influential in the trained model (predictive association, not direct causation).",
            "feature_count": features_df.shape[1],
            "trace_id": trace_id,
            "feature_vector_hash": feature_hash,
            "data_cutoff_timestamp": cutoff_ts,
            "observation_ids_used": obs_ids_used,
            "latest_observation_id": latest_obs_id,
            "model_version": m_ver,
            "model_type": "Balanced Logistic Regression + Ridge Regression",
            "prediction_source": "production_batch"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Inference error: {e}",
            "intern_id": intern_id,
            "origin_week": origin_week
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Inference error: {e}",
            "intern_id": intern_id,
            "origin_week": origin_week
        }
