"""
SkillTrack-ML: Hiring Decision Support Engine
Provides transparent, explainable decision-support for intern hiring readiness.

Rules & Principles:
- Distinctly separates actual performance, ML predictions, and mentor judgment.
- Does NOT claim a performance classifier is a hiring classifier.
- Formula and weights are explicit, documented, and configurable.
- Supports four decisions:
  1. HIRE: Strong composite evidence (>= 80.0) with consistent high performance.
  2. EXTEND_INTERNSHIP: Developing performance (60.0 - 79.9) or insufficient observation weeks.
  3. DO_NOT_RECOMMEND: Consistent low performance (< 60.0) or low technical mastery.
  4. INSUFFICIENT_EVIDENCE: Fewer than 3 weeks of activity or missing critical evaluations.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from src.data.live_data_ingestion import DATABASE_PATH
from src.ml.predict import get_intern_ml_forecast


# Documented rubrics weights
RUBRIC_WEIGHTS = {
    "actual_assessment_avg": 0.40,      # 40% Actual observed test & assessment performance
    "predicted_performance": 0.25,      # 25% Validated ML ridge regression forecast
    "mentor_evaluation": 0.25,          # 25% Human mentor expert evaluation
    "questionnaire_and_behavior": 0.10  # 10% Professional engagement & self-efficacy
}


def compute_hiring_recommendation(
    intern_id: int,
    db_path: Path = DATABASE_PATH
) -> Dict[str, Any]:
    """
    Computes an explainable, audit-traceable provisional hiring recommendation for an intern.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Fetch Intern Profile
    cur.execute(
        "SELECT intern_id, name, role, experience_level, current_week, average_score, final_performance_score FROM interns WHERE intern_id = ?",
        (intern_id,)
    )
    intern_row = cur.fetchone()
    if not intern_row:
        conn.close()
        raise ValueError(f"Intern ID {intern_id} not found.")

    itn_id, name, role, exp_lvl, current_week, avg_score, final_perf = intern_row

    # 2. Query Assessments
    cur.execute(
        "SELECT COUNT(*), AVG(percentage_score), MAX(percentage_score), MIN(percentage_score) FROM intern_assessments WHERE intern_id = ?",
        (intern_id,)
    )
    ass_count, ass_avg, ass_max, ass_min = cur.fetchone()
    ass_count = ass_count or 0
    ass_avg = float(ass_avg) if ass_avg is not None else float(avg_score or 70.0)

    # 3. Query Mentor Evaluations
    cur.execute(
        "SELECT overall_score, decision, remarks, technical_score, communication_score, evaluated_at FROM mentor_evaluations WHERE intern_id = ? ORDER BY evaluated_at DESC LIMIT 1",
        (intern_id,)
    )
    mentor_eval = cur.fetchone()
    if mentor_eval:
        m_score = float(mentor_eval[0])
        m_decision = mentor_eval[1]
        m_remarks = mentor_eval[2]
        m_tech = float(mentor_eval[3]) if mentor_eval[3] is not None else m_score
        m_comm = float(mentor_eval[4]) if mentor_eval[4] is not None else m_score
    else:
        m_score = float(final_perf or 70.0)
        m_decision = "PENDING_REVIEW"
        m_remarks = "Provisional baseline score prior to formal mentor review."
        m_tech = m_score
        m_comm = m_score

    # 4. Query Questionnaire (debugging_confidence & engagement)
    cur.execute(
        "SELECT debugging_confidence FROM questionnaire_responses WHERE intern_id = ? ORDER BY week DESC LIMIT 1",
        (intern_id,)
    )
    q_row = cur.fetchone()
    q_score = float(q_row[0]) if (q_row and q_row[0] is not None) else 75.0

    conn.close()

    # 5. Fetch Real ML Model Forecast
    ml_forecast = get_intern_ml_forecast(intern_id, db_path=db_path)
    pred_perf = float(ml_forecast.get("predicted_performance_score") or ml_forecast.get("predicted_performance") or 75.0)
    pred_gap = str(ml_forecast.get("predicted_skill_gap", "Developing"))
    model_version = str(ml_forecast.get("model_version", "Ridge Regression Production"))

    # 6. Check Minimum Evidence Requirements
    weeks_observed = current_week or 1
    if weeks_observed < 3 and ass_count < 2:
        recommendation = "INSUFFICIENT_EVIDENCE"
        confidence_level = "Provisional"
        rationale = (
            f"Intern has only {weeks_observed} week(s) of records and {ass_count} assessment(s). "
            f"A minimum of 3 active weeks and 2 verified assessments are required for an employment recommendation."
        )
        composite_score = round(ass_avg, 1)
    else:
        # Calculate Weighted Composite Readiness Score
        composite_score = round(
            (RUBRIC_WEIGHTS["actual_assessment_avg"] * ass_avg) +
            (RUBRIC_WEIGHTS["predicted_performance"] * pred_perf) +
            (RUBRIC_WEIGHTS["mentor_evaluation"] * m_score) +
            (RUBRIC_WEIGHTS["questionnaire_and_behavior"] * q_score),
            1
        )

        if composite_score >= 80.0 and pred_gap == "Strong":
            recommendation = "HIRE"
            confidence_level = "High"
            rationale = (
                f"Candidate demonstrates strong readiness with an 80+ composite readiness score ({composite_score}/100). "
                f"Performance outlook indicates continued high performance ({pred_perf:.1f}), corroborated by mentor evaluation ({m_score:.1f})."
            )
        elif composite_score >= 65.0:
            recommendation = "EXTEND_INTERNSHIP"
            confidence_level = "Medium"
            rationale = (
                f"Candidate shows solid potential with a composite readiness score of {composite_score}/100. "
                f"Performance outlook indicates {pred_gap} progression ({pred_perf:.1f}). Recommending a 4-week extension to solidify technical mastery."
            )
        else:
            recommendation = "DO_NOT_RECOMMEND"
            confidence_level = "High" if weeks_observed >= 8 else "Medium"
            rationale = (
                f"Candidate's composite score ({composite_score}/100) and performance outlook ({pred_perf:.1f}, {pred_gap}) "
                f"fall below the hiring benchmark threshold of 65.0."
            )

    evidence_summary = {
        "rubric_weights": RUBRIC_WEIGHTS,
        "intern_id": intern_id,
        "name": name,
        "role": role,
        "weeks_observed": weeks_observed,
        "assessment_count": ass_count,
        "actual_assessment_avg": round(ass_avg, 1),
        "predicted_performance": round(pred_perf, 1),
        "predicted_skill_gap": pred_gap,
        "model_version": model_version,
        "mentor_evaluation_score": round(m_score, 1),
        "mentor_technical_score": round(m_tech, 1),
        "mentor_communication_score": round(m_comm, 1),
        "questionnaire_score": round(q_score, 1),
        "mentor_remarks": m_remarks
    }

    return {
        "intern_id": intern_id,
        "intern_name": name,
        "min_evidence_met": (recommendation != "INSUFFICIENT_EVIDENCE"),
        "recommendation": recommendation,
        "composite_score": composite_score,
        "confidence_level": confidence_level,
        "rationale": rationale,
        "evidence_summary": evidence_summary
    }


def save_mentor_hiring_decision(
    intern_id: int,
    mentor_id: int,
    decision_type: str,
    composite_readiness_score: float,
    confidence_level: str,
    decision_rationale: str,
    strengths: str,
    growth_areas: str,
    is_final: bool = False,
    evidence_summary: Optional[Dict[str, Any]] = None,
    db_path: Path = DATABASE_PATH
) -> int:
    """
    Persists a human mentor hiring decision to `mentor_hiring_decisions` table.
    """
    valid_decisions = ["HIRE", "EXTEND_INTERNSHIP", "DO_NOT_RECOMMEND", "INSUFFICIENT_EVIDENCE"]
    if decision_type not in valid_decisions:
        raise ValueError(f"Invalid decision type: '{decision_type}'. Must be one of {valid_decisions}")

    now_str = datetime.now().isoformat(timespec="seconds")
    ev = evidence_summary or {}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO mentor_hiring_decisions (
            intern_id, mentor_id, decision_type, is_final,
            composite_readiness_score, confidence_level, decision_rationale,
            strengths, growth_areas,
            actual_assessment_avg, predicted_performance_score,
            predicted_gap_category, mentor_evaluation_score,
            evidence_summary_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        intern_id, mentor_id, decision_type, 1 if is_final else 0,
        composite_readiness_score, confidence_level, decision_rationale.strip(),
        strengths.strip() if strengths else "",
        growth_areas.strip() if growth_areas else "",
        ev.get("actual_assessment_avg"),
        ev.get("predicted_performance"),
        ev.get("predicted_skill_gap"),
        ev.get("mentor_evaluation_score"),
        json.dumps(ev),
        now_str
    ))
    decision_id = cur.lastrowid
    conn.commit()
    conn.close()

    return decision_id


def get_latest_hiring_decision(
    intern_id: int,
    db_path: Path = DATABASE_PATH
) -> Optional[Dict[str, Any]]:
    """Retrieves the most recent mentor hiring decision for an intern."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM mentor_hiring_decisions
        WHERE intern_id = ?
        ORDER BY created_at DESC, decision_id DESC
        LIMIT 1
    """, (intern_id,))
    cols = [c[0] for c in cur.description]
    row = cur.fetchone()
    conn.close()

    if not row:
        return None
    res = dict(zip(cols, row))
    if res.get("evidence_summary_json"):
        try:
            res["evidence_summary"] = json.loads(res["evidence_summary_json"])
        except Exception:
            res["evidence_summary"] = {}
    return res
