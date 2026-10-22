"""
SkillTrack-ML: Live Dataset Management Page Component
Provides interactive data entry, filtering, and real-time inspection of assessments and ML observations.
"""

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from src.data.live_data_ingestion import (
    get_observation_summary,
    get_recent_dataset_records,
    resolve_pending_forecasting_labels,
    DATABASE_PATH
)
from src.data.assessment_service import (
    validate_and_save_assessment,
    get_intern_assessments,
    get_assessment_summary_stats,
    AssessmentValidationError,
    VALID_ASSESSMENT_TYPES
)
from src.ml.predict import get_intern_ml_forecast


def render_live_dataset_page(current_user: tuple, get_connection_fn):
    user_id = current_user[0]
    mentor_id = current_user[4] or 1

    st.markdown(
        """
        <div>
            <div class="page-eyebrow">COHORT ASSESSMENT RECORDS & WORKSPACE</div>
            <div class="page-heading">Assessment Records & Data Center</div>
            <div class="page-subheading">Interactive data entry, validation, milestone tracking, and row-level inspection of real intern assessment records.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------
    # 1. LIVE DATASET STATISTICS (DATABASE-DERIVED COUNTS)
    # ---------------------------------------------------------
    obs_summary = get_observation_summary() if get_observation_summary else {}
    ass_summary = get_assessment_summary_stats() if get_assessment_summary_stats else {}

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("Total Observations", f"{obs_summary.get('total_active_observations', 5000):,}")
        st.caption("All Active Cohort Records")
    with k2:
        st.metric("Historical Baseline", f"{obs_summary.get('synthetic_baseline_count', 5000):,}")
        st.caption("Preserved 500x10 Rows")
    with k3:
        st.metric("Application Live", f"{obs_summary.get('application_generated_count', 0):,}")
        st.caption("Live Ingested Events")
    with k4:
        st.metric("Assessments Recorded", f"{ass_summary.get('total_assessments', 0):,}")
        st.caption(f"{ass_summary.get('unique_interns', 0)} Unique Interns")
    with k5:
        st.metric("Immediate Labeled", f"{obs_summary.get('immediate_supervised_count', 0):,}")
        st.caption("Ground-Truth Ready")
    with k6:
        st.metric("Pending Labels", f"{obs_summary.get('pending_future_label_count', 0):,}")
        st.caption("Zero Leakage Protected")

    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px; margin-top: 10px; margin-bottom: 1.25rem; font-size: 12.5px; color: #475569;">
            <div><strong>Active Dataset:</strong> <code>data/processed/intern_ml_dataset.csv</code> (4,500 transition instances, 43 features)</div>
            <div><strong>Latest Event:</strong> <code>{}</code></div>
            <div><strong>Ground Truth Policy:</strong> Dual-label architecture (Zero label fabrication)</div>
        </div>
        """.format(ass_summary.get("latest_timestamp") or "Initial Baseline"),
        unsafe_allow_html=True
    )

    # Two Main Workspace Tabs
    tab_entry, tab_view, tab_lifecycle = st.tabs([
        "📥 Enter Real Assessment",
        "🔍 Live Dataset & Observation Inspector",
        "🔄 Dataset Lifecycle & Operational Controls"
    ])

    # ---------------------------------------------------------
    # TAB 1: ENTER REAL ASSESSMENT
    # ---------------------------------------------------------
    with tab_entry:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-card-title">Direct Assessment Entry & Feature Ingestion</div>
                <div class="section-card-subtitle">Submit verified technical evaluations, quizzes, or behavioral ratings for an intern. Automatically validates ranges, writes to <code>intern_assessments</code>, constructs observation provenance, and alerts inference pipelines.</div>
            """,
            unsafe_allow_html=True
        )

        conn = get_connection_fn()
        intern_rows = conn.execute("SELECT intern_id, name, role, experience_level, current_week, average_score FROM interns ORDER BY intern_id ASC").fetchall()
        conn.close()

        intern_dict = {r[0]: f"Intern #{r[0]} - {r[1]} ({r[2]}, {r[3]}, Wk {r[4]})" for r in intern_rows}

        col_itn, col_type = st.columns(2)
        with col_itn:
            sel_intern_id = st.selectbox(
                "Select Candidate Intern",
                options=list(intern_dict.keys()),
                format_func=lambda x: intern_dict[x],
                key="live_ds_intern_select"
            )
        with col_type:
            sel_ass_type = st.selectbox(
                "Assessment Type",
                options=VALID_ASSESSMENT_TYPES,
                key="live_ds_assessment_type"
            )

        # Show intern snapshot
        sel_row = next((r for r in intern_rows if r[0] == sel_intern_id), None)
        if sel_row:
            st.markdown(
                f"""
                <div style="background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; font-size: 13px; color: #1E1B4B; display: flex; justify-content: space-between;">
                    <div><strong>Intern ID:</strong> {sel_row[0]} | <strong>Name:</strong> {sel_row[1]}</div>
                    <div><strong>Current Week:</strong> {sel_row[4]} | <strong>Historical Average Score:</strong> {sel_row[5]:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.form("live_dataset_assessment_form", clear_on_submit=False):
            st.markdown("<strong>1. Assessment Metadata</strong>", unsafe_allow_html=True)
            c_meta1, c_meta2, c_meta3 = st.columns(3)
            with c_meta1:
                ass_name = st.text_input("Assessment Title", value=f"Sprint {sel_row[4] if sel_row else 10} {sel_ass_type}")
            with c_meta2:
                ass_week = st.number_input("Training Week", min_value=1, max_value=20, value=int(sel_row[4] if sel_row else 10))
            with c_meta3:
                attempt_num = st.number_input("Attempt Number", min_value=1, max_value=5, value=1)

            c_sc1, c_sc2 = st.columns(2)
            with c_sc1:
                max_marks = st.number_input("Maximum Marks", min_value=1.0, max_value=500.0, value=100.0, step=10.0)
            with c_sc2:
                marks_obtained = st.number_input("Marks Obtained", min_value=0.0, max_value=500.0, value=85.0, step=1.0)

            pct_calc = round((marks_obtained / max_marks) * 100.0, 2) if max_marks > 0 else 0.0
            st.caption(f"Calculated Percentage Score: **{pct_calc:.2f}%**")

            # Conditional fields depending on assessment type
            tech_dict = {}
            quiz_dict = {}
            prof_dict = {}
            mentor_dict = {}

            if sel_ass_type == "Technical Assessment":
                st.markdown("<hr style='margin: 10px 0;'><strong>2. Technical Performance Details</strong>", unsafe_allow_html=True)
                ct1, ct2, ct3, ct4 = st.columns(4)
                with ct1:
                    t_cases_tot = st.number_input("Total Test Cases", min_value=0, max_value=50, value=10)
                with ct2:
                    t_cases_pass = st.number_input("Passed Test Cases", min_value=0, max_value=50, value=9)
                with ct3:
                    code_qual = st.slider("Code Quality Score", min_value=0.0, max_value=100.0, value=88.0)
                with ct4:
                    comp_time = st.number_input("Completion Time (Mins)", min_value=1.0, max_value=300.0, value=45.0)

                tech_dict = {
                    "tasks_completed": 1 if marks_obtained >= (max_marks * 0.6) else 0,
                    "test_cases_total": t_cases_tot,
                    "test_cases_passed": t_cases_pass,
                    "code_quality_score": code_qual,
                    "completion_time_minutes": comp_time
                }

            elif sel_ass_type == "MCQ Quiz":
                st.markdown("<hr style='margin: 10px 0;'><strong>2. Quiz & Skill Category Details</strong>", unsafe_allow_html=True)
                cq1, cq2, cq3 = st.columns(3)
                with cq1:
                    q_att = st.number_input("Questions Attempted", min_value=1, max_value=50, value=20)
                with cq2:
                    q_corr = st.number_input("Correct Answers", min_value=0, max_value=50, value=17)
                with cq3:
                    q_inc = q_att - q_corr
                    st.text_input("Incorrect Answers (Auto)", value=str(max(0, q_inc)), disabled=True)
                q_topic = st.selectbox("Topic Skill", ["Java Basics", "OOP Principles", "Collections & Maps", "Streams & Lambda", "Concurrency", "Database & JDBC"])

                quiz_dict = {
                    "questions_attempted": q_att,
                    "correct_answers": q_corr,
                    "incorrect_answers": max(0, q_inc),
                    "topic_category": q_topic
                }

            elif sel_ass_type == "Professional & Attendance":
                st.markdown("<hr style='margin: 10px 0;'><strong>2. Professional Execution & Discipline</strong>", unsafe_allow_html=True)
                cp1, cp2, cp3 = st.columns(3)
                with cp1:
                    t_rate = st.slider("Task Completion Rate", min_value=0.0, max_value=1.0, value=0.92)
                with cp2:
                    att_rate = st.slider("Attendance Rate", min_value=0.0, max_value=1.0, value=0.98)
                with cp3:
                    comm_rate = st.slider("Communication Rating", min_value=0.0, max_value=100.0, value=85.0)
                prof_dict = {
                    "task_completion_rate": t_rate,
                    "attendance_rate": att_rate,
                    "communication_rating": comm_rate
                }

            elif sel_ass_type == "Comprehensive Review":
                st.markdown("<hr style='margin: 10px 0;'><strong>2. Supervisory Mentor Evaluation Review</strong>", unsafe_allow_html=True)
                cm1, cm2 = st.columns(2)
                with cm1:
                    m_eval_sc = st.number_input("Mentor Review Score (0-100)", min_value=0.0, max_value=100.0, value=pct_calc)
                with cm2:
                    m_final_chk = st.checkbox("Mark as Final Period Evaluation", value=False)
                m_remarks = st.text_area("Supervisory Remarks", value="Demonstrates solid problem-solving fundamentals and code readability.")
                m_strengths = st.text_input("Key Strengths", value="Clean modular architecture, excellent unit testing habits.")
                m_areas = st.text_input("Growth Areas", value="Deepen knowledge of concurrent execution and asynchronous handlers.")

                mentor_dict = {
                    "mentor_id": mentor_id,
                    "mentor_score": m_eval_sc,
                    "mentor_remarks": m_remarks,
                    "strengths": m_strengths,
                    "areas_for_improvement": m_areas,
                    "is_final": 1 if m_final_chk else 0
                }

            submit_assessment_btn = st.form_submit_button("Submit Assessment & Ingest Observation", use_container_width=True)

            if submit_assessment_btn:
                try:
                    res = validate_and_save_assessment(
                        intern_id=sel_intern_id,
                        week=int(ass_week),
                        assessment_type=sel_ass_type,
                        assessment_name=ass_name,
                        marks_obtained=float(marks_obtained),
                        max_marks=float(max_marks),
                        attempt_number=int(attempt_num),
                        technical_details=tech_dict,
                        quiz_details=quiz_dict,
                        professional_details=prof_dict,
                        mentor_details=mentor_dict
                    )
                    st.success(f"Assessment saved successfully! Assessment ID: **#{res['assessment_id']}** &bull; Linked Ingested Observation: **OBS-{res['observation_id']}**")

                    # Update performance outlook on fresh data
                    fc = get_intern_ml_forecast(sel_intern_id)
                    st.info(f"Updated Performance Outlook for Intern #{sel_intern_id}: **Estimated Performance: {fc.get('predicted_performance', 0):.1f} / 100** | **Development Area: {fc.get('predicted_skill_gap', 'N/A')}**")
                    st.rerun()

                except AssessmentValidationError as ave:
                    st.error(f"Validation Error: {ave}")
                except Exception as ex:
                    st.error(f"Unexpected Ingestion Error: {ex}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 2: LIVE DATASET & OBSERVATION INSPECTOR
    # ---------------------------------------------------------
    with tab_view:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-card-title">Live Assessment Records & Lineage Filter</div>
                <div class="section-card-subtitle">Filter stored assessment submissions by intern, assessment type, or training week.</div>
            """,
            unsafe_allow_html=True
        )

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            all_intern_opts = ["All Interns"] + [str(r[0]) for r in intern_rows]
            f_intern = st.selectbox("Filter by Intern", all_intern_opts, key="filter_ass_intern")
        with f_col2:
            all_type_opts = ["All Types"] + VALID_ASSESSMENT_TYPES
            f_type = st.selectbox("Filter by Type", all_type_opts, key="filter_ass_type")
        with f_col3:
            f_week_opts = ["All Weeks"] + [str(w) for w in range(1, 21)]
            f_week = st.selectbox("Filter by Week", f_week_opts, key="filter_ass_week")

        req_intern_id = int(f_intern) if f_intern != "All Interns" else None
        req_type = f_type if f_type != "All Types" else None
        req_week = int(f_week) if f_week != "All Weeks" else None

        records = get_intern_assessments(intern_id=req_intern_id, assessment_type=req_type, week=req_week)
        if records:
            rec_df = pd.DataFrame(records)
            cols_to_show = [
                "assessment_id", "created_at", "intern_id", "intern_name", "week",
                "assessment_type", "assessment_name", "percentage_score", "observation_id"
            ]
            avail_cols = [c for c in cols_to_show if c in rec_df.columns]
            styled_rec_df = rec_df[avail_cols].rename(columns={
                "assessment_id": "ID",
                "created_at": "Timestamp",
                "intern_id": "Intern ID",
                "intern_name": "Name",
                "week": "Week",
                "assessment_type": "Type",
                "assessment_name": "Title",
                "percentage_score": "Score %",
                "observation_id": "Obs ID"
            })
            st.dataframe(styled_rec_df, use_container_width=True, hide_index=True)

            # Row detail inspector
            st.markdown("<strong>Detailed Row Inspector</strong>", unsafe_allow_html=True)
            sel_ass_id = st.selectbox(
                "Select Assessment to Inspect",
                options=[r["assessment_id"] for r in records],
                format_func=lambda x: f"Assessment #{x} ({next((r['assessment_name'] for r in records if r['assessment_id'] == x), '')})",
                key="select_ass_inspector"
            )
            inspect_item = next((r for r in records if r["assessment_id"] == sel_ass_id), None)
            if inspect_item:
                c_i1, c_i2, c_i3 = st.columns(3)
                with c_i1:
                    st.markdown(f"**Score:** {inspect_item['percentage_score']}% ({inspect_item['marks_obtained']}/{inspect_item['max_marks']})")
                    st.markdown(f"**Attempt #:** {inspect_item['attempt_number']}")
                    st.markdown(f"**Test Cases:** {inspect_item['test_cases_passed']} / {inspect_item['test_cases_total']}")
                with c_i2:
                    st.markdown(f"**Quiz Questions:** {inspect_item['questions_attempted']} (Correct: {inspect_item['correct_answers']})")
                    st.markdown(f"**Topic:** {inspect_item['topic_category'] or 'N/A'}")
                    st.markdown(f"**Linked Observation:** `OBS-{inspect_item['observation_id']}`")
                with c_i3:
                    st.markdown(f"**Mentor Score:** {inspect_item['mentor_score'] or 'N/A'}")
                    st.markdown(f"**Mentor Remarks:** {inspect_item['mentor_remarks'] or 'None'}")
                    st.markdown(f"**Strengths:** {inspect_item['strengths'] or 'None'}")
        else:
            st.info("No assessment records found matching the active filters.")

        st.markdown("</div>", unsafe_allow_html=True)

        # Ingestion Stream Table
        st.markdown(
            """
            <div class="section-card">
                <div class="section-card-title">Live Milestone & Activity Stream (ml_observations)</div>
                <div class="section-card-subtitle">Real-time log of raw events ingested from coding submissions, quizzes, questionnaires, and evaluations.</div>
            """,
            unsafe_allow_html=True
        )
        recent_obs = get_recent_dataset_records(limit=30) if get_recent_dataset_records else []
        if recent_obs:
            st.dataframe(pd.DataFrame(recent_obs), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 3: DATASET LIFECYCLE & OPERATIONAL CONTROLS
    # ---------------------------------------------------------
    with tab_lifecycle:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-card-title">Assessment Lifecycle & Outcome Resolution</div>
                <div class="section-card-subtitle">Operational controls for progressing observations through verified outcome milestones.</div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 1.25rem;">
                <h4 style="margin-top: 0; color: #0F172A;">Understanding the Closed-Loop Data Lifecycle:</h4>
                <ol style="margin-bottom: 0; padding-left: 20px; line-height: 1.6; font-size: 13.5px; color: #334155;">
                    <li><strong>Assessment / Event Submission:</strong> Intern completes task or mentor submits evaluation &rarr; Ingested into <code>intern_assessments</code> &amp; <code>ml_observations</code>.</li>
                    <li><strong>Metric State Synchronization:</strong> Immediate metrics sync to <code>intern_performance</code> &rarr; Indicator vector updates automatically.</li>
                    <li><strong>Outcome Status Segregation:</strong>
                        <ul>
                            <li><code>IMMEDIATE_LABELED</code>: Quizzes &amp; coding test scores have instant ground truth &rarr; Verified immediate outcomes.</li>
                            <li><code>PENDING_FUTURE_LABEL</code>: Next-week performance outlooks ($W \to W+1$) hold <code>outcome_value = NULL</code> until the target week is completed.</li>
                        </ul>
                    </li>
                    <li><strong>Automated Outcome Resolution:</strong> When an intern reaches and completes Week W+1, clicking <em>Sync Pending Evaluation Outcomes</em> attaches the authentic observed outcome.</li>
                    <li><strong>Periodic Evaluation Gate:</strong> Newly verified samples are added to the evaluation pool once safety thresholds are reached.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_res_btn, col_res_info = st.columns([1, 2])
        with col_res_btn:
            if st.button("🔄 Sync Pending Evaluation Outcomes Now", use_container_width=True, help="Scans pending records where target_week <= current intern week and attaches ground-truth outcomes."):
                if resolve_pending_forecasting_labels:
                    n_resolved = resolve_pending_forecasting_labels()
                    if n_resolved > 0:
                        st.success(f"Successfully updated {n_resolved} pending evaluation record(s)!")
                    else:
                        st.info("No pending evaluation records have reached their target milestone week yet.")
                    st.rerun()

        with col_res_info:
            st.caption("Syncing outcomes ensures that performance outlooks are evaluated only against genuinely observed future outcomes, preserving evaluation validity.")

        st.markdown("</div>", unsafe_allow_html=True)
