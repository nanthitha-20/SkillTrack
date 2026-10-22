"""
SkillTrack-ML: End-to-End Intern Lifecycle Demonstration View
Allows an academic evaluator to follow ONE intern through the complete 10-stage journey:
1. Intern Profile & Initial Data
2. Assessment Submission & Ingestion
3. ML Feature Engineering & Preprocessing
4. Dual ML Model Predictions (Classification + Regression)
5. Prediction vs Actual Performance Comparison
6. Mentor Review & Rating
7. Explainable Hiring Recommendation (Readiness Rubrics)
8. Mentor Feedback & Online Learning Eligibility
9. Subsequent Prediction & Model State Delta
10. Complete Audit Timeline
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from pathlib import Path

from src.data.live_data_ingestion import (
    ingest_observation,
    DATABASE_PATH
)
from src.data.assessment_service import (
    validate_and_save_assessment,
    get_intern_assessments
)
from src.ml.predict import (
    get_intern_ml_forecast,
    build_feature_vector,
    load_models
)
from src.ml.hiring_engine import (
    compute_hiring_recommendation,
    save_mentor_hiring_decision,
    get_latest_hiring_decision,
    RUBRIC_WEIGHTS
)
from src.ml.online_learner import get_online_learning_status


def render_e2e_demo_page(current_user: tuple, get_connection_fn):
    user_id = current_user[0]
    mentor_id = current_user[4] or 1

    st.markdown(
        """
        <div>
            <div class="page-eyebrow">CENTRAL FACULTY DEMONSTRATION WORKFLOW</div>
            <div class="page-heading">End-to-End Intern Lifecycle & Hiring Decision Simulation</div>
            <div class="page-subheading">Follow ONE engineering intern through the complete closed-loop lifecycle from raw assessment to performance evaluation, mentor review, hiring decision support, and adaptive updates.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    conn = get_connection_fn()
    intern_rows = conn.execute("SELECT intern_id, name, role, experience_level, current_week, average_score, final_performance_score FROM interns ORDER BY intern_id ASC").fetchall()
    conn.close()

    intern_dict = {r[0]: f"Intern #{r[0]} - {r[1]} ({r[2]}, {r[3]}, Wk {r[4]})" for r in intern_rows}

    # Intern Selection Header
    st.markdown(
        """
        <div class="section-card">
            <div class="section-card-title">Select Demonstration Subject</div>
        """,
        unsafe_allow_html=True
    )
    sel_intern_id = st.selectbox(
        "Candidate Intern",
        options=list(intern_dict.keys()),
        format_func=lambda x: intern_dict[x],
        key="e2e_demo_intern_select"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    target_intern = next((r for r in intern_rows if r[0] == sel_intern_id), None)
    if not target_intern:
        st.warning("Please select a valid intern.")
        return

    # 10 STAGE PROGRESSION IN ACCORDION / STEPPERS
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # STAGE 1: INTERN PROFILE & INITIAL DATA
    # ---------------------------------------------------------
    with st.expander("📍 STAGE 1: Intern Profile & Initial Baseline State", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Intern ID", f"#{target_intern[0]}")
            st.caption(f"Name: {target_intern[1]}")
        with c2:
            st.metric("Role & Level", target_intern[2])
            st.caption(f"Experience: {target_intern[3]}")
        with c3:
            st.metric("Current Week", f"Week {target_intern[4]}")
            st.caption("Active Sprint")
        with c4:
            st.metric("Historical Avg", f"{target_intern[5]:.1f}%")
            st.caption(f"Final Score: {target_intern[6]:.1f}")

        # Prior Assessments
        prior_ass = get_intern_assessments(intern_id=sel_intern_id)
        st.markdown(f"**Verified Assessments on Record ({len(prior_ass)}):**")
        if prior_ass:
            p_df = pd.DataFrame(prior_ass)[["assessment_id", "week", "assessment_type", "assessment_name", "percentage_score", "created_at"]]
            st.dataframe(p_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No custom assessments logged yet for this intern.")

    # ---------------------------------------------------------
    # STAGE 2: ASSESSMENT SUBMISSION
    # ---------------------------------------------------------
    with st.expander("📝 STAGE 2: Enter Live Assessment Event", expanded=True):
        st.markdown("Submit a live technical coding challenge or comprehensive assessment for this intern:")
        with st.form("e2e_assessment_quick_form", clear_on_submit=False):
            ca1, ca2, ca3 = st.columns(3)
            with ca1:
                e2e_ass_name = st.text_input("Assessment Name", value="Live Sprint 10 Architecture Review")
            with ca2:
                e2e_ass_type = st.selectbox("Type", ["Technical Assessment", "MCQ Quiz", "Comprehensive Review"])
            with ca3:
                e2e_ass_score = st.number_input("Score Obtained (0-100)", min_value=0.0, max_value=100.0, value=88.0)

            e2e_sub_btn = st.form_submit_button("Record Assessment & Ingest Event", use_container_width=True)
            if e2e_sub_btn:
                res = validate_and_save_assessment(
                    intern_id=sel_intern_id,
                    week=int(target_intern[4]),
                    assessment_type=e2e_ass_type,
                    assessment_name=e2e_ass_name,
                    marks_obtained=float(e2e_ass_score),
                    max_marks=100.0,
                    technical_details={"test_cases_total": 10, "test_cases_passed": 9, "code_quality_score": 88.0}
                )
                st.session_state[f"e2e_last_ass_{sel_intern_id}"] = res
                st.success(f"Ingested Assessment #{res['assessment_id']} &bull; Observation OBS-{res['observation_id']} created with label_status=IMMEDIATE_LABELED")
                st.rerun()

        last_ass = st.session_state.get(f"e2e_last_ass_{sel_intern_id}")
        if last_ass:
            st.info(f"Active Demonstration Assessment: **#{last_ass['assessment_id']}** ({last_ass['assessment_type']} - Score: **{last_ass['percentage_score']}%**)")

    # ---------------------------------------------------------
    # STAGE 3 & 4: PERFORMANCE EVALUATION & OUTLOOK
    # ---------------------------------------------------------
    with st.expander("⚙️ STAGE 3 & 4: Metric Evaluation & Performance Outlook", expanded=True):
        v_df, ow, cs, err = build_feature_vector(sel_intern_id)
        if err:
            st.error(err)
        elif v_df is not None:
            st.markdown(f"**Evaluated Performance Indicators (Week {ow}):**")
            st.dataframe(v_df.iloc[:, :12], use_container_width=True)
            st.caption(f"... and {len(v_df.columns)-12} more progression indicators including rolling test pass rates and cumulative task completion metrics.")

            # Evaluation Predictions
            cls_model, reg_model = load_models()
            pred_cls = cls_model.predict(v_df)[0]
            pred_reg = float(reg_model.predict(v_df)[0])

            col_pred1, col_pred2 = st.columns(2)
            with col_pred1:
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 4px solid #4F46E5; border-radius: 8px; padding: 14px;">
                        <div style="font-size: 11px; font-weight: 700; color: #4F46E5; text-transform: uppercase;">Development Area Outlook</div>
                        <div style="font-size: 20px; font-weight: 700; color: #0F172A; margin: 4px 0;">{pred_cls}</div>
                        <div style="font-size: 12px; color: #64748B;">Evaluation Pipeline: Performance Categorization</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_pred2:
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 4px solid #0EA5E9; border-radius: 8px; padding: 14px;">
                        <div style="font-size: 11px; font-weight: 700; color: #0EA5E9; text-transform: uppercase;">Upcoming Performance Outlook</div>
                        <div style="font-size: 20px; font-weight: 700; color: #0F172A; margin: 4px 0;">{pred_reg:.1f} / 100</div>
                        <div style="font-size: 12px; color: #64748B;">Evaluation Pipeline: Regularized Progression Metric</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ---------------------------------------------------------
    # STAGE 5 & 6: ESTIMATED OUTLOOK VS RECORDED SCORE & MENTOR REVIEW
    # ---------------------------------------------------------
    with st.expander("🔍 STAGE 5 & 6: Estimated Outlook vs Recorded Score & Mentor Review", expanded=True):
        st.markdown(
            f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-bottom: 12px; font-size: 13px; color: #334155;">
                <strong>Comparative Performance Evaluation:</strong><br/>
                &bull; <strong>Performance Outlook Estimate:</strong> {pred_reg:.1f} / 100<br/>
                &bull; <strong>Recent Observed Assessment Score:</strong> {last_ass['percentage_score'] if last_ass else target_intern[5]:.1f}%<br/>
                &bull; <strong>Variance:</strong> {(last_ass['percentage_score'] if last_ass else target_intern[5]) - pred_reg:+.2f} points
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Mentor Evaluation Review")
        with st.form(f"e2e_mentor_eval_form_{sel_intern_id}", clear_on_submit=False):
            ce1, ce2 = st.columns(2)
            with ce1:
                e_tech = st.slider("Technical Mastery (0-100)", min_value=0.0, max_value=100.0, value=85.0)
                e_comm = st.slider("Communication & Collaboration (0-100)", min_value=0.0, max_value=100.0, value=90.0)
            with ce2:
                e_disc = st.slider("Discipline & Reliability (0-100)", min_value=0.0, max_value=100.0, value=95.0)
                e_overall = round(0.4 * e_tech + 0.3 * e_comm + 0.3 * e_disc, 1)
                st.metric("Overall Evaluation Score", f"{e_overall:.1f}%")

            e_remarks = st.text_area("Supervisory Remarks", value="Exhibits strong engineering aptitude, proactive communication, and quick learning.")
            sub_eval_btn = st.form_submit_button("Record Mentor Review", use_container_width=True)

            if sub_eval_btn:
                conn = get_connection_fn()
                now_str = datetime.now().isoformat(timespec="seconds")
                conn.execute("""
                    INSERT INTO mentor_evaluations (
                        intern_id, mentor_id, technical_score, communication_score,
                        discipline_score, overall_score, decision, remarks, evaluated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (sel_intern_id, mentor_id, e_tech, e_comm, e_disc, e_overall, "FULL_TIME", e_remarks, now_str))
                conn.commit()
                conn.close()

                # Also ingest observation for mentor evaluation
                ingest_observation(
                    intern_id=sel_intern_id,
                    week=int(target_intern[4]),
                    source="mentor_evaluation",
                    event_type="mentor_evaluated",
                    metric_name="mentor_overall_score",
                    metric_value=e_overall,
                    payload={"technical": e_tech, "remarks": e_remarks},
                    label_status="IMMEDIATE_LABELED",
                    training_eligible=1,
                    outcome_value=e_overall
                )
                st.success("Mentor evaluation recorded successfully!")
                st.rerun()

    # ---------------------------------------------------------
    # STAGE 7: HIRING RECOMMENDATION SUPPORT
    # ---------------------------------------------------------
    with st.expander("🎯 STAGE 7: Decision Support & Hiring Readiness Evaluation", expanded=True):
        hire_data = compute_hiring_recommendation(sel_intern_id)
        ev_sum = hire_data["evidence_summary"]

        rec_style = {
            "HIRE": ("#ECFDF5", "#047857", "#10B981"),
            "EXTEND_INTERNSHIP": ("#FFFBEB", "#B45309", "#F59E0B"),
            "DO_NOT_RECOMMEND": ("#FEF2F2", "#B91C1C", "#EF4444"),
            "INSUFFICIENT_EVIDENCE": ("#F8FAFC", "#475569", "#94A3B8")
        }.get(hire_data["recommendation"], ("#F8FAFC", "#475569", "#94A3B8"))

        st.markdown(
            f"""
            <div style="background: {rec_style[0]}; border: 2px solid {rec_style[2]}; border-radius: 12px; padding: 18px; margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 11px; font-weight: 700; color: {rec_style[1]}; text-transform: uppercase;">Provisional Decision-Support Recommendation</div>
                        <div style="font-size: 24px; font-weight: 800; color: {rec_style[1]}; margin: 4px 0;">{hire_data['recommendation']}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 11px; color: #64748B;">Composite Readiness Score</div>
                        <div style="font-size: 26px; font-weight: 800; color: #0F172A;">{hire_data['composite_score']}<span style="font-size: 14px; font-weight: 500; color: #64748B;">/100</span></div>
                        <div style="font-size: 11px; font-weight: 600; color: {rec_style[1]};">Estimate Reliability: {hire_data['confidence_level']}</div>
                    </div>
                </div>
                <div style="font-size: 13.5px; color: #1E293B; margin-top: 10px; line-height: 1.5; border-top: 1px solid rgba(0,0,0,0.06); padding-top: 10px;">
                    <strong>Decision Rationale:</strong> {hire_data['rationale']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Rubric Weights & Evidence Breakdown Table
        st.markdown("<strong>Documented Rubric Weights & Evidence Breakdown:</strong>", unsafe_allow_html=True)
        r_rows = [
            {"Component": "Actual Assessments Average", "Weight": "40%", "Observed Value": f"{ev_sum.get('actual_assessment_avg')}%", "Contribution": f"{ev_sum.get('actual_assessment_avg', 0)*0.4:.1f} pts"},
            {"Component": "Upcoming Performance Outlook", "Weight": "25%", "Observed Value": f"{ev_sum.get('predicted_performance')}/100", "Contribution": f"{ev_sum.get('predicted_performance', 0)*0.25:.1f} pts"},
            {"Component": "Mentor Evaluation Score", "Weight": "25%", "Observed Value": f"{ev_sum.get('mentor_evaluation_score')}%", "Contribution": f"{ev_sum.get('mentor_evaluation_score', 0)*0.25:.1f} pts"},
            {"Component": "Professional & Questionnaire", "Weight": "10%", "Observed Value": f"{ev_sum.get('questionnaire_score')}%", "Contribution": f"{ev_sum.get('questionnaire_score', 0)*0.10:.1f} pts"}
        ]
        st.dataframe(pd.DataFrame(r_rows), use_container_width=True, hide_index=True)

        # Formal Mentor Decision Authorization
        st.markdown("#### Authorize Final Employment Recommendation")
        with st.form("mentor_formal_hiring_decision_form"):
            ch1, ch2 = st.columns(2)
            with ch1:
                formal_dec = st.selectbox("Final Authorized Decision", ["HIRE", "EXTEND_INTERNSHIP", "DO_NOT_RECOMMEND"])
                formal_final_chk = st.checkbox("Sign as Binding Supervisory Decision", value=True)
            with ch2:
                formal_rationale = st.text_area("Supervisory Justification", value=hire_data["rationale"])

            f_strength = st.text_input("Candidate Core Strengths", value="Exceeds technical bar in frontend state management, consistent sprint delivery.")
            f_growth = st.text_input("Areas for Post-Hire Development", value="Deepen knowledge of distributed caching and micro-frontend deployments.")

            save_dec_btn = st.form_submit_button("Save & Persist Formal Hiring Decision", use_container_width=True)
            if save_dec_btn:
                d_id = save_mentor_hiring_decision(
                    intern_id=sel_intern_id,
                    mentor_id=mentor_id,
                    decision_type=formal_dec,
                    composite_readiness_score=hire_data["composite_score"],
                    confidence_level=hire_data["confidence_level"],
                    decision_rationale=formal_rationale,
                    strengths=f_strength,
                    growth_areas=f_growth,
                    is_final=formal_final_chk,
                    evidence_summary=ev_sum
                )
                st.success(f"Formal Hiring Decision #{d_id} persisted in `mentor_hiring_decisions` table with complete evidence provenance!")
                st.rerun()

        latest_dec = get_latest_hiring_decision(sel_intern_id)
        if latest_dec:
            st.markdown(f"**Latest Persisted Hiring Decision:** `ID #{latest_dec['decision_id']}` &bull; Type: **{latest_dec['decision_type']}** &bull; Timestamp: `{latest_dec['created_at']}`")

    # ---------------------------------------------------------
    # STAGE 8 & 9: EVALUATION UPDATES & SUBSEQUENT OUTLOOK
    # ---------------------------------------------------------
    with st.expander("🔄 STAGE 8 & 9: Evaluation Updates & Subsequent Outlook Delta", expanded=True):
        onl_stat = get_online_learning_status() if get_online_learning_status else {}
        st.markdown(
            f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-bottom: 12px; font-size: 13px; color: #334155;">
                <strong>Evaluation Engine Status:</strong><br/>
                &bull; <strong>Engine Version:</strong> <code>{onl_stat.get('model_version', 'eval_engine_v1')}</code><br/>
                &bull; <strong>Incremental Updates:</strong> {onl_stat.get('total_updates', 0)}<br/>
                &bull; <strong>Verification Rule:</strong> Only validated observations update the performance baseline.
            </div>
            """,
            unsafe_allow_html=True
        )

        fc_subsequent = get_intern_ml_forecast(sel_intern_id)
        sc1, sc2 = st.columns(2)
        with sc1:
            st.metric("Initial Baseline Score", f"{target_intern[6]:.1f}")
        with sc2:
            st.metric("Subsequent Outlook", f"{fc_subsequent.get('predicted_performance', 0):.1f}", delta=f"{fc_subsequent.get('predicted_performance', 0) - target_intern[6]:+.2f} delta")

    # ---------------------------------------------------------
    # STAGE 10: COMPLETE AUDIT TIMELINE
    # ---------------------------------------------------------
    with st.expander("📋 STAGE 10: Complete Single-Intern Milestone & Activity Timeline", expanded=True):
        st.markdown(f"**Unified Activity Stream for Intern #{sel_intern_id}:**")
        conn = get_connection_fn()
        obs_rows = conn.execute("""
            SELECT observation_id, timestamp, source, event_type, metric_name, metric_value, label_status, training_eligible
            FROM ml_observations
            WHERE intern_id = ?
            ORDER BY timestamp DESC, observation_id DESC
        """, (sel_intern_id,)).fetchall()
        conn.close()

        if obs_rows:
            obs_df = pd.DataFrame(obs_rows, columns=["Obs ID", "Timestamp", "Source", "Event Type", "Metric", "Value", "Label Status", "Training Eligible"])
            st.dataframe(obs_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No live observations logged for this intern yet.")
