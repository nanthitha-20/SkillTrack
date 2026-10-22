"""
SkillTrack-ML: Deliverables Views
Provides the interactive UI components for:
1. Intern Deliverable Submission & Revision Lifecycle (versioning, self-assessment, PDF upload, history)
2. Mentor Review Hub (cohort overview, PDF viewer/downloader, rubric scoring, status transition)
"""

import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
import streamlit as st

from src.data.deliverable_service import (
    get_all_milestones,
    get_milestone_by_id,
    get_intern_deliverable_history,
    save_pdf_deliverable,
    record_mentor_deliverable_review,
    get_all_submissions_for_mentor,
    get_submission_by_id,
    get_cohort_deliverable_metrics,
    DeliverableValidationError
)


def render_pdf_preview(file_path_str: str, height: int = 420):
    """Safely renders an embedded PDF preview or fallback download button."""
    p = Path(file_path_str)
    if not p.exists():
        st.warning(f"Deliverable file not found at {p.name}")
        return

    try:
        with open(p, "rb") as f:
            pdf_bytes = f.read()

        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_display = f"""
            <iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="{height}px" type="application/pdf" style="border: 1px solid #E2E8F0; border-radius: 8px;">
                <p>Your browser does not support embedded PDF viewing. Please use the download button below.</p>
            </iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
        st.download_button(
            label=f"Download {p.name}",
            data=pdf_bytes,
            file_name=p.name,
            mime="application/pdf",
            key=f"dl_{p.name}"
        )
    except Exception as ex:
        st.info(f"PDF Preview unavailable: {ex}")


def render_intern_deliverables_page(intern_id: int):
    """
    Intern-facing view for submitting and tracking project deliverables across milestones.
    """
    st.markdown(
        """
        <div class="hero-banner">
            <div class="page-eyebrow">CURRICULUM DELIVERABLES</div>
            <div class="page-heading">Project Submissions & Milestone Reports</div>
            <div class="page-subheading">Submit your milestone PDF deliverables, complete self-assessments, and review mentor evaluation history.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    milestones = get_all_milestones()
    history = get_intern_deliverable_history(intern_id)

    # 1. Milestone Selection & Status Grid
    m_options = {f"Milestone {m['milestone_number']}: {m['title']}": m['milestone_id'] for m in milestones}
    selected_label = st.selectbox("Select Project Milestone", list(m_options.keys()))
    selected_milestone_id = m_options[selected_label]
    milestone = get_milestone_by_id(selected_milestone_id)

    # Check existing submissions for this milestone
    milestone_subs = [s for s in history if s["milestone_id"] == selected_milestone_id]
    latest_sub = milestone_subs[0] if milestone_subs else None

    # Milestone Requirements Card
    st.markdown(
        f"""
        <div class="section-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                <div>
                    <span class="badge-developing">{milestone['track']}</span>
                    <span class="badge-strong">Week {milestone['week']}</span>
                </div>
                <div>
                    <strong>Max Score: {milestone['max_score']:.0f} pts</strong>
                </div>
            </div>
            <div class="section-card-title">{milestone['title']}</div>
            <div style="font-size: 13.5px; color: #475569; margin-bottom: 12px; line-height: 1.5;">
                {milestone['description']}
            </div>
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 14px; font-size: 13px; color: #1E293B;">
                <strong>Deliverable Requirements:</strong> {milestone.get('deliverable_requirements', 'Standard PDF document.')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Lifecycle Status Banner if a submission exists
    if latest_sub:
        st.markdown(
            f"""
            <div class="section-card" style="border-left: 4px solid {'#10B981' if latest_sub['review_status'] == 'APPROVED' else ('#EF4444' if latest_sub['review_status'] == 'CHANGES_REQUESTED' else '#3B82F6')};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 12px; font-weight: 700; color: #64748B; text-transform: uppercase;">Current Review Lifecycle (Version {latest_sub['version_number']})</div>
                        <div style="font-size: 17px; font-weight: 700; color: #0F172A; margin-top: 2px;">{latest_sub['title']}</div>
                        <div style="font-size: 12px; color: #64748B;">Submitted on {latest_sub['submitted_at']}</div>
                    </div>
                    <div>
                        <span class="badge-{'approved' if latest_sub['review_status'] == 'APPROVED' else ('changes-requested' if latest_sub['review_status'] == 'CHANGES_REQUESTED' else 'submitted')}" style="font-size: 13px; padding: 4px 12px;">
                            {latest_sub['review_status'].replace('_', ' ')}
                        </span>
                    </div>
                </div>
            """,
            unsafe_allow_html=True
        )

        if latest_sub.get("mentor_feedback"):
            st.markdown(
                f"""
                <div class="rec-banner" style="margin-top: 12px;">
                    <strong>Mentor Feedback:</strong> {latest_sub['mentor_feedback']}<br/>
                    <strong>Score Awarded:</strong> {latest_sub['mentor_score'] if latest_sub['mentor_score'] is not None else 'Pending'} / {milestone['max_score']:.0f}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # 2. Upload / Resubmission Form
    can_submit = True
    form_title = f"Submit Deliverable — Version {len(milestone_subs) + 1}"
    if latest_sub and latest_sub["review_status"] == "APPROVED":
        st.success("This milestone has been approved by your mentor. No further revisions required.")
        can_submit = False

    if can_submit:
        with st.expander(form_title, expanded=(latest_sub is None or latest_sub["review_status"] == "CHANGES_REQUESTED")):
            with st.form(f"submit_pdf_form_{selected_milestone_id}"):
                sub_title = st.text_input(
                    "Deliverable Title",
                    value=f"{milestone['title']} - Report" if not latest_sub else f"{milestone['title']} (Revision v{len(milestone_subs)+1})",
                    help="Descriptive title of your submitted work."
                )
                sub_desc = st.text_area("Submission Overview / Abstract", placeholder="Briefly summarize the architecture, approaches, and findings in this deliverable...")

                col_u1, col_u2 = st.columns([1.2, 1])
                with col_u1:
                    uploaded_file = st.file_uploader(
                        "Upload Deliverable (PDF only, max 15MB)",
                        type=["pdf"],
                        help="Please ensure your file is a compiled PDF document."
                    )
                with col_u2:
                    self_score = st.slider("Self-Assessment Score", 0.0, 100.0, 85.0, step=1.0, help="Rate your confidence against the milestone criteria.")
                    intern_notes = st.text_area("Notes to Mentor (Optional)", placeholder="Notes regarding feedback addressed or specific questions...")

                submit_btn = st.form_submit_button("Submit Deliverable for Review", use_container_width=True)

                if submit_btn:
                    if not uploaded_file:
                        st.error("Please select a PDF file to upload.")
                    elif not sub_title.strip():
                        st.error("Please enter a deliverable title.")
                    else:
                        file_bytes = uploaded_file.read()
                        try:
                            res = save_pdf_deliverable(
                                intern_id=intern_id,
                                milestone_id=selected_milestone_id,
                                title=sub_title,
                                file_bytes=file_bytes,
                                original_filename=uploaded_file.name,
                                description=sub_desc,
                                intern_notes=intern_notes,
                                self_assessment_score=self_score
                            )
                            st.success(f"Deliverable Version {res['version_number']} submitted successfully!")
                            st.rerun()
                        except DeliverableValidationError as val_err:
                            st.error(f"Validation Error: {val_err}")
                        except Exception as ex:
                            st.error(f"Submission failed: {ex}")

    # 3. Version History & Previous Submissions
    if milestone_subs:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown(f"#### Submission History for Milestone {milestone['milestone_number']} ({len(milestone_subs)} versions)")

        for idx, sub in enumerate(milestone_subs):
            with st.expander(f"Version {sub['version_number']} &bull; {sub['submitted_at']} &bull; Status: {sub['review_status']}", expanded=(idx == 0)):
                c1, c2 = st.columns([1, 1.2])
                with c1:
                    st.markdown(
                        f"""
                        <div style="font-size: 13.5px; line-height: 1.6;">
                            <strong>Title:</strong> {sub['title']}<br/>
                            <strong>Original File:</strong> <code>{sub['original_filename']}</code> ({sub['file_size_bytes'] / 1024:.1f} KB)<br/>
                            <strong>Self-Assessment:</strong> {sub['self_assessment_score'] or 'N/A'}/100<br/>
                            <strong>Mentor Score:</strong> {sub['mentor_score'] if sub['mentor_score'] is not None else 'Pending'}<br/>
                            <strong>Status:</strong> {sub['review_status']}<br/>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if sub.get("intern_notes"):
                        st.caption(f"**Notes:** {sub['intern_notes']}")
                    if sub.get("mentor_feedback"):
                        st.info(f"**Mentor Remarks:** {sub['mentor_feedback']}")

                with c2:
                    st.markdown("<strong>Document Preview</strong>", unsafe_allow_html=True)
                    render_pdf_preview(sub["file_path"], height=320)


def render_mentor_review_hub(mentor_id: int):
    """
    Mentor-facing view for inspecting all cohort PDF deliverables, scoring rubrics, and updating review statuses.
    """
    st.markdown(
        """
        <div class="hero-banner">
            <div class="page-eyebrow">COHORT EVALUATIONS</div>
            <div class="page-heading">Deliverable Review Hub</div>
            <div class="page-subheading">Inspect submitted PDF deliverables, verify self-assessment criteria, provide actionable mentor feedback, and authorize milestone approvals.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    metrics = get_cohort_deliverable_metrics()

    # 4 Overview Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-label">Total Submissions</div>
                <div class="metric-card-value">{metrics['total_submissions']}</div>
                <div class="metric-card-caption">Across all milestones</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-label">Awaiting Review</div>
                <div class="metric-card-value" style="color: #EA580C;">{metrics['pending_reviews']}</div>
                <div class="metric-card-caption">Ready for evaluation</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-label">Approved</div>
                <div class="metric-card-value" style="color: #059669;">{metrics['approved_count']}</div>
                <div class="metric-card-caption">Completed deliverables</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-label">Changes Requested</div>
                <div class="metric-card-value" style="color: #DC2626;">{metrics['changes_requested_count']}</div>
                <div class="metric-card-caption">Pending intern revision</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns([1.5, 1, 1])
    with f_col1:
        search_query = st.text_input("Search Candidate or Deliverable", placeholder="Type intern name, ID, or title...")
    with f_col2:
        status_opts = ["All Statuses", "SUBMITTED", "UNDER_REVIEW", "CHANGES_REQUESTED", "APPROVED"]
        sel_status = st.selectbox("Filter Status", status_opts)
    with f_col3:
        milestone_opts = [0] + list(range(1, 11))
        sel_milestone = st.selectbox("Filter Milestone", milestone_opts, format_func=lambda x: "All Milestones" if x == 0 else f"Milestone {x}")

    submissions = get_all_submissions_for_mentor(
        status_filter=sel_status,
        milestone_filter=sel_milestone
    )

    if search_query.strip():
        q = search_query.strip().lower()
        submissions = [
            s for s in submissions
            if q in s["intern_name"].lower() or q in str(s["intern_id"]) or q in s["title"].lower()
        ]

    if not submissions:
        st.info("No deliverable submissions match the active filter.")
        return

    # Select submission to inspect
    sub_map = {
        f"#{s['submission_id']} &bull; {s['intern_name']} (M{s['milestone_number']} v{s['version_number']}) — {s['review_status']}": s['submission_id']
        for s in submissions
    }
    sel_sub_label = st.selectbox("Select Deliverable to Evaluate", list(sub_map.keys()))
    sel_sub_id = sub_map[sel_sub_label]

    sub = get_submission_by_id(sel_sub_id)
    if not sub:
        st.warning("Submission details could not be loaded.")
        return

    # Inspection Workspace (2 Columns: PDF Preview on Left, Mentor Evaluation Form on Right)
    col_pdf, col_form = st.columns([1.1, 1.2], gap="large")

    with col_pdf:
        st.markdown(
            f"""
            <div class="section-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div>
                        <span class="badge-developing">Milestone {sub['milestone_number']} &bull; {sub['track']}</span>
                        <span class="badge-strong">Version {sub['version_number']}</span>
                    </div>
                    <span class="badge-{'approved' if sub['review_status'] == 'APPROVED' else ('changes-requested' if sub['review_status'] == 'CHANGES_REQUESTED' else 'submitted')}">
                        {sub['review_status'].replace('_', ' ')}
                    </span>
                </div>
                <div class="section-card-title">{sub['title']}</div>
                <div style="font-size: 13px; color: #64748B; margin-bottom: 8px;">
                    <strong>Candidate:</strong> {sub['intern_name']} ({sub['intern_email']} &bull; ID: {sub['intern_id']})<br/>
                    <strong>Submitted:</strong> {sub['submitted_at']} &bull; File: <code>{sub['original_filename']}</code>
                </div>
                <div style="font-size: 13.5px; color: #334155; line-height: 1.5; margin-bottom: 12px;">
                    {sub['description'] or 'No abstract provided.'}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("##### PDF Document Viewer")
        render_pdf_preview(sub["file_path"], height=480)

    with col_form:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-card-title">Mentor Evaluation & Rubric Scoring</div>
                <div class="section-card-subtitle">Review submission contents, score milestone rubrics, and record feedback.</div>
            """,
            unsafe_allow_html=True
        )

        with st.form(f"eval_deliverable_form_{sub['submission_id']}"):
            st.markdown(f"**Self-Assessment Score:** `{sub['self_assessment_score'] or 'N/A'} / 100`")
            if sub.get("intern_notes"):
                st.caption(f"**Candidate Notes:** {sub['intern_notes']}")

            # Dynamic Rubric Sliders if defined
            rubric_scores = {}
            if sub.get("rubric"):
                st.markdown("###### Milestone Rubric Criteria Breakdown")
                for criterion, max_pts in sub["rubric"].items():
                    default_val = float(max_pts * 0.8)
                    rubric_scores[criterion] = st.slider(f"{criterion} (Max {max_pts} pts)", 0.0, float(max_pts), default_val, step=1.0)
                calc_total = sum(rubric_scores.values())
            else:
                calc_total = 80.0

            overall_score = st.slider("Overall Score (out of 100)", 0.0, 100.0, float(calc_total), step=1.0)

            review_decision = st.selectbox(
                "Review Decision",
                ["APPROVED", "CHANGES_REQUESTED", "UNDER_REVIEW"],
                index=0 if sub["review_status"] == "APPROVED" else (1 if sub["review_status"] == "CHANGES_REQUESTED" else 0)
            )

            feedback_text = st.text_area(
                "Written Feedback & Guidance",
                value=sub.get("mentor_feedback") or "",
                placeholder="Detail technical strengths, areas for refinement, code architecture clarity, and required revisions..."
            )

            submit_eval_btn = st.form_submit_button("Record Mentor Evaluation", use_container_width=True)

            if submit_eval_btn:
                if not feedback_text.strip():
                    st.error("Please enter written feedback for the intern before submitting.")
                else:
                    try:
                        record_mentor_deliverable_review(
                            submission_id=sub["submission_id"],
                            mentor_id=mentor_id,
                            review_status=review_decision,
                            score=overall_score,
                            feedback=feedback_text,
                            rubric_scores=rubric_scores
                        )
                        st.success(f"Evaluation recorded! Deliverable marked as {review_decision}.")
                        st.rerun()
                    except DeliverableValidationError as val_err:
                        st.error(f"Validation Error: {val_err}")
                    except Exception as ex:
                        st.error(f"Failed to record evaluation: {ex}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Audit History for this submission
        if sub.get("reviews"):
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            with st.expander(f"Review History Log ({len(sub['reviews'])})", expanded=False):
                for rev in sub["reviews"]:
                    st.markdown(
                        f"""
                        <div style="border-bottom: 1px solid #E2E8F0; padding: 8px 0; font-size: 13px;">
                            <strong>Reviewed At:</strong> {rev['reviewed_at']} &bull; <strong>Status:</strong> {rev['review_status']}<br/>
                            <strong>Score:</strong> {rev['score']}/100<br/>
                            <strong>Feedback:</strong> {rev['feedback']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
