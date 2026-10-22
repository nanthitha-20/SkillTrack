"""
SkillTrack-ML: Patch app.py
Updates:
1. Replaces login view with render_split_screen_login()
2. Replaces Coding nav item with Deliverables
3. Injects render_intern_deliverables_page() for Interns
4. Injects render_mentor_review_hub() for Mentors ("Reviews" nav)
5. Replaces blue/purple styling with get_orange_theme_css()
6. Adapts Intern and Mentor dashboard metrics to reflect PDF deliverables
"""

with open('app/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add view imports to top
view_import = """from src.data.deliverable_views import (
    render_intern_deliverables_page,
    render_mentor_review_hub,
    render_pdf_preview
)
from src.data.orange_theme import (
    get_orange_theme_css,
    render_split_screen_login
)"""

text = text.replace("from src.data.deliverable_service import (", view_import + "\nfrom src.data.deliverable_service import (", 1)

# 2. Replace Login page rendering
old_login_block = """if not st.session_state.logged_in:

    # Global Login Styles (Light Theme Commercial Design with Subtle Gradient Depth)"""

assert old_login_block in text, "old_login_block not found"

# Find end of login block (up to st.stop() before CURRENT USER)
login_start_idx = text.find("if not st.session_state.logged_in:")
current_user_marker = "# ============================================================\n# CURRENT USER\n# ============================================================"
current_user_idx = text.find(current_user_marker)

assert login_start_idx != -1 and current_user_idx != -1, "Login block boundaries not found"

new_login_block = """if not st.session_state.logged_in:
    login_err = st.session_state.get("login_error_msg", "")
    username_input, password_input, login_clicked = render_split_screen_login(error_msg=login_err)

    if login_clicked:
        if not username_input or not password_input:
            st.session_state["login_error_msg"] = "Please enter both username and password."
            st.rerun()
        else:
            user = authenticate_user(
                username_input.strip(),
                password_input.strip()
            )
            if user:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.session_state["login_error_msg"] = ""
                st.rerun()
            else:
                st.session_state["login_error_msg"] = "Invalid username or password."
                st.rerun()

    st.stop()


"""

text = text[:login_start_idx] + new_login_block + text[current_user_idx:]

# 3. Update Sidebar branding & navigation
old_sidebar_header = """    <div style="display: flex; align-items: center; gap: 10px; padding: 6px 0 16px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 16px;">
        <div style="width: 34px; height: 34px; border-radius: 8px; background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); display: flex; align-items: center; justify-content: center; color: #FFFFFF; box-shadow: 0 2px 6px rgba(79, 70, 229, 0.25);">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
                <path d="M6 12v5c3 3 9 3 12 0v-5"/>
            </svg>
        </div>
        <div>
            <div style="font-size: 15px; font-weight: 700; color: #0F172A; letter-spacing: -0.01em;">SkillTrack</div>
            <div style="font-size: 11px; color: #64748B; font-weight: 500;">Internship Platform</div>
        </div>
    </div>"""

new_sidebar_header = """    <div style="display: flex; align-items: center; gap: 10px; padding: 6px 0 16px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 16px;">
        <div style="width: 34px; height: 34px; border-radius: 8px; background: linear-gradient(180deg, #EA580C 0%, #C2410C 100%); display: flex; align-items: center; justify-content: center; color: #FFFFFF; box-shadow: 0 2px 6px rgba(234, 88, 12, 0.3);">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
                <path d="M6 12v5c3 3 9 3 12 0v-5"/>
            </svg>
        </div>
        <div>
            <div style="font-size: 15px; font-weight: 800; color: #0F172A; letter-spacing: -0.01em;">SkillTrack</div>
            <div style="font-size: 11px; color: #EA580C; font-weight: 600;">Engineering Portal</div>
        </div>
    </div>"""

text = text.replace(old_sidebar_header, new_sidebar_header, 1)

# Navigation items for Intern
old_intern_nav = """if role == "intern":
    nav_page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "My Skills",
            "Questionnaire",
            "Coding",
            "Assessments",
            "Learning",
            "Chat with Mentor",
            "Progress",
            "Profile"
        ],
        label_visibility="collapsed"
    )"""

new_intern_nav = """if role == "intern":
    nav_page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "My Skills",
            "Questionnaire",
            "Deliverables",
            "Assessments",
            "Learning",
            "Chat with Mentor",
            "Progress",
            "Profile"
        ],
        label_visibility="collapsed"
    )"""

text = text.replace(old_intern_nav, new_intern_nav, 1)

# Navigation items for Mentor
old_mentor_nav = """elif role == "mentor":
    nav_page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "End-to-End Demo",
            "Live ML",
            "Live ML Dataset",
            "Messages",
            "Interns",
            "Performance",
            "Skill Gaps",
            "Reports",
            "Benchmark",
            "Profile"
        ],
        label_visibility="collapsed"
    )"""

new_mentor_nav = """elif role == "mentor":
    nav_page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Deliverables",
            "End-to-End Demo",
            "Live ML",
            "Live ML Dataset",
            "Messages",
            "Interns",
            "Performance",
            "Skill Gaps",
            "Reports",
            "Benchmark",
            "Profile"
        ],
        label_visibility="collapsed"
    )"""

text = text.replace(old_mentor_nav, new_mentor_nav, 1)

# Sidebar account color: #4F46E5 -> #EA580C
text = text.replace('color: #4F46E5; font-weight: 500; margin-bottom: 12px;">{role.title()} Role</div>',
                    'color: #EA580C; font-weight: 600; margin-bottom: 12px;">{role.title()} Role</div>')

# 4. Replace Intern CSS injection with orange theme
old_intern_css_marker = "    # Custom Intern Styles (Light Theme Commercial System with Subtle Lavender Depth)\n    st.markdown("
assert old_intern_css_marker in text, "old_intern_css_marker not found"

# Find end of intern css block
intern_css_start = text.find(old_intern_css_marker)
intern_dashboard_marker = "    # --------------------------------------------------------\n    # PAGE 1: DASHBOARD"
intern_dashboard_idx = text.find(intern_dashboard_marker)

assert intern_css_start != -1 and intern_dashboard_idx != -1, "Intern CSS boundary not found"

new_intern_css_block = """    # Orange Theme Injection
    st.markdown(get_orange_theme_css(), unsafe_allow_html=True)
"""

text = text[:intern_css_start] + new_intern_css_block + text[intern_dashboard_idx:]

# 5. Replace Mentor CSS injection with orange theme
old_mentor_css_marker = "    # Custom Mentor Theme Styles (Light Theme with Subtle Lavender Depth)\n    st.markdown("
mentor_css_start = text.find(old_mentor_css_marker)
mentor_dashboard_marker = "    # --------------------------------------------------------\n    # MENTOR PAGE 1: DASHBOARD"
mentor_dashboard_idx = text.find(mentor_dashboard_marker)

assert mentor_css_start != -1 and mentor_dashboard_idx != -1, "Mentor CSS boundary not found"

new_mentor_css_block = """    # Orange Theme Injection
    st.markdown(get_orange_theme_css(), unsafe_allow_html=True)
"""

text = text[:mentor_css_start] + new_mentor_css_block + text[mentor_dashboard_idx:]

# 6. Replace intern coding page with Deliverables page
old_coding_start = "    # --------------------------------------------------------\n    # PAGE 3: CODING WORKSPACE\n    # --------------------------------------------------------\n    elif nav_page == \"Coding\":"
old_assessments_marker = "    # --------------------------------------------------------\n    # PAGE 4: ASSESSMENTS\n    # --------------------------------------------------------\n    elif nav_page == \"Assessments\":"

coding_start_idx = text.find(old_coding_start)
assessments_start_idx = text.find(old_assessments_marker)

assert coding_start_idx != -1 and assessments_start_idx != -1, "Coding page boundary not found"

new_deliverables_page = """    # --------------------------------------------------------
    # PAGE 3: PROJECT DELIVERABLES WORKSPACE
    # --------------------------------------------------------
    elif nav_page == "Deliverables":
        render_intern_deliverables_page(intern_id)

"""

text = text[:coding_start_idx] + new_deliverables_page + text[assessments_start_idx:]

# 7. Add Deliverables handler in mentor section
old_mentor_e2e_start = "    # --------------------------------------------------------\n    # MENTOR PAGE: END-TO-END DEMO (10-STAGE INTERN LIFECYCLE)\n    # --------------------------------------------------------\n    elif nav_page == \"End-to-End Demo\":"
mentor_e2e_idx = text.find(old_mentor_e2e_start)
assert mentor_e2e_idx != -1, "Mentor E2E boundary not found"

mentor_deliverables_block = """    # --------------------------------------------------------
    # MENTOR PAGE: PROJECT DELIVERABLES REVIEW HUB
    # --------------------------------------------------------
    elif nav_page == "Deliverables":
        render_mentor_review_hub(mentor_id)

"""

text = text[:mentor_e2e_idx] + mentor_deliverables_block + text[mentor_e2e_idx:]

# 8. Update Intern Dashboard Metrics (Card 2: Coding Tasks -> Project Deliverables)
old_intern_m2 = """        with m2:
            st.markdown(
                f\"\"\"
                <div class="metric-card">
                    <div class="metric-card-label">Coding Tasks</div>
                    <div class="metric-card-value">{completed_tasks} / {total_tasks}</div>
                    <div class="metric-card-caption">Completed problem sets</div>
                </div>
                \"\"\",
                unsafe_allow_html=True
            )"""

new_intern_m2 = """        deliverable_hist = get_intern_deliverable_history(intern_id) if get_intern_deliverable_history else []
        approved_delivs = len([d for d in deliverable_hist if d['review_status'] == 'APPROVED'])
        total_milestones = 10

        with m2:
            st.markdown(
                f\"\"\"
                <div class="metric-card">
                    <div class="metric-card-label">Project Deliverables</div>
                    <div class="metric-card-value">{len(deliverable_hist)} Submitted</div>
                    <div class="metric-card-caption">{approved_delivs} of {total_milestones} milestones approved</div>
                </div>
                \"\"\",
                unsafe_allow_html=True
            )"""

text = text.replace(old_intern_m2, new_intern_m2, 1)

# 9. Update Intern Dashboard Recent Activity Card (Replace Recent Submissions with Recent Deliverables)
old_recent_sub_card = """            if submissions:
                for sub in submissions[:4]:
                    sub_title = sub[2] or "Coding Challenge"
                    sub_lang = sub[6] or "N/A"
                    sub_time = sub[5]
                    sub_status = sub[4]
                    st.markdown(
                        f\"\"\"
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #F1F5F9;">
                            <div>
                                <div style="font-weight: 600; font-size: 13.5px; color: #0F172A;">{sub_title}</div>
                                <div style="font-size: 12px; color: #64748B;">{sub_lang} • {sub_time}</div>
                            </div>
                            <span class="badge-strong">{sub_status}</span>
                        </div>
                        \"\"\",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No coding submissions recorded yet.")"""

new_recent_sub_card = """            if deliverable_hist:
                for sub in deliverable_hist[:4]:
                    sub_title = sub['title']
                    sub_time = sub['submitted_at']
                    sub_status = sub['review_status']
                    badge_c = "badge-approved" if sub_status == "APPROVED" else ("badge-changes-requested" if sub_status == "CHANGES_REQUESTED" else "badge-submitted")
                    st.markdown(
                        f\"\"\"
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #F1F5F9;">
                            <div>
                                <div style="font-weight: 600; font-size: 13.5px; color: #0F172A;">{sub_title}</div>
                                <div style="font-size: 12px; color: #64748B;">Version {sub['version_number']} • {sub_time}</div>
                            </div>
                            <span class="{badge_c}">{sub_status.replace('_', ' ')}</span>
                        </div>
                        \"\"\",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No project deliverables submitted yet.")"""

text = text.replace(old_recent_sub_card, new_recent_sub_card, 1)

# 10. Update Mentor Intern Detail Inspection (Coding Submissions -> Project Deliverables)
old_mentor_intern_sub_card = """                    with sub_col:
                        st.markdown(
                            \"\"\"
                            <div class="section-card">
                                <div class="section-card-title">Coding Submissions</div>
                                <div class="section-card-subtitle">Recent code submissions and test outcomes.</div>
                            \"\"\",
                            unsafe_allow_html=True
                        )
                        if selected_submissions:
                            for sub in selected_submissions[:4]:
                                st.markdown(
                                    f\"\"\"
                                    <div style="padding: 8px 0; border-bottom: 1px solid #F1F5F9;">
                                        <div style="font-weight: 600; font-size: 13.5px; color: #0F172A;">{sub[2] or 'Coding Task'}</div>
                                        <div style="font-size: 12px; color: #64748B;">Lang: {sub[6] or 'N/A'} • Status: {sub[4]} • {sub[5]}</div>
                                    </div>
                                    \"\"\",
                                    unsafe_allow_html=True
                                )
                        else:
                            st.caption("No coding submissions recorded.")
                        st.markdown("</div>", unsafe_allow_html=True)"""

new_mentor_intern_sub_card = """                    with sub_col:
                        st.markdown(
                            \"\"\"
                            <div class="section-card">
                                <div class="section-card-title">Project Deliverables</div>
                                <div class="section-card-subtitle">Recent PDF milestone submissions and evaluation states.</div>
                            \"\"\",
                            unsafe_allow_html=True
                        )
                        cand_delivs = get_intern_deliverable_history(selected_intern_id) if get_intern_deliverable_history else []
                        if cand_delivs:
                            for sub in cand_delivs[:4]:
                                st.markdown(
                                    f\"\"\"
                                    <div style="padding: 8px 0; border-bottom: 1px solid #F1F5F9;">
                                        <div style="font-weight: 600; font-size: 13.5px; color: #0F172A;">{sub['title']}</div>
                                        <div style="font-size: 12px; color: #64748B;">v{sub['version_number']} • Status: {sub['review_status']} • {sub['submitted_at']}</div>
                                    </div>
                                    \"\"\",
                                    unsafe_allow_html=True
                                )
                        else:
                            st.caption("No project deliverables recorded.")
                        st.markdown("</div>", unsafe_allow_html=True)"""

text = text.replace(old_mentor_intern_sub_card, new_mentor_intern_sub_card, 1)

# Write modified app.py
with open('app/app.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Step 2 (app.py orange theme and deliverables integration) applied successfully!")
