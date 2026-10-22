"""
SkillTrack-ML: Orange Design System & Reusable Components
Implements the exact UI styling inspired by the reference screenshot:
- Palette: #EA580C (Primary Orange), #C2410C (Deeper Orange), #F8FAFC (Light Background), #FFFFFF (White Cards)
- Modern typography: Inter / System font stack
- Clean cards, accessible badges, responsive split-screen authentication, and status lifecycles.
"""

def get_orange_theme_css() -> str:
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        color: #0F172A;
    }

    .stApp {
        background-color: #F8FAFC !important;
        background-image: radial-gradient(#EA580C 0.5px, transparent 0.5px), radial-gradient(#EA580C 0.5px, #F8FAFC 0.5px) !important;
        background-size: 24px 24px !important;
        background-position: 0 0, 12px 12px !important;
        background-opacity: 0.03 !important;
    }

    /* Streamlit Global Header & Footer Cleanup */
    header { visibility: hidden !important; height: 0 !important; }
    footer { visibility: hidden !important; height: 0 !important; }
    #MainMenu { visibility: hidden !important; }
    .stDeployButton { display: none !important; }

    /* Layout Spacing */
    .block-container {
        max-width: 1200px !important;
        padding-top: 2.2rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Orange Hero Banner */
    .hero-banner {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #EA580C;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 4px 12px -2px rgba(234, 88, 12, 0.05);
    }

    .page-eyebrow {
        font-size: 11px;
        font-weight: 700;
        color: #EA580C;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .page-heading {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.25;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }

    .page-subheading {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 0;
        line-height: 1.5;
    }

    /* Clean Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.03);
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }

    .metric-card:hover {
        box-shadow: 0 4px 16px -2px rgba(234, 88, 12, 0.12);
        border-color: #FDBA74;
    }

    .metric-card-label {
        font-size: 11px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }

    .metric-card-value {
        font-size: 22px;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
        margin-bottom: 2px;
    }

    .metric-card-caption {
        font-size: 12px;
        color: #94A3B8;
    }

    /* White Section Card */
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.03);
        transition: all 0.2s ease;
    }

    .section-card-title {
        font-size: 17px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
    }

    .section-card-subtitle {
        font-size: 13px;
        color: #64748B;
        margin-bottom: 16px;
    }

    /* Next-Week Skill Outlook Card */
    .outlook-card {
        background: #FFFFFF;
        border: 1px solid #FED7AA;
        border-left: 4px solid #EA580C;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 4px 0 rgba(15, 23, 42, 0.03), 0 8px 20px -4px rgba(234, 88, 12, 0.08);
    }

    /* Status Badges */
    .badge-strong {
        display: inline-flex;
        align-items: center;
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 6px;
    }

    .badge-developing {
        display: inline-flex;
        align-items: center;
        background-color: #FFF7ED;
        color: #C2410C;
        border: 1px solid #FFEDD5;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 6px;
    }

    .badge-needs-improvement {
        display: inline-flex;
        align-items: center;
        background-color: #FEF2F2;
        color: #B91C1C;
        border: 1px solid #FECACA;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 6px;
    }

    .badge-submitted {
        display: inline-flex;
        align-items: center;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 6px;
    }

    .badge-under-review {
        display: inline-flex;
        align-items: center;
        background-color: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 6px;
    }

    .badge-changes-requested {
        display: inline-flex;
        align-items: center;
        background-color: #FEF2F2;
        color: #B91C1C;
        border: 1px solid #FECACA;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 6px;
    }

    .badge-approved {
        display: inline-flex;
        align-items: center;
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 6px;
    }

    .rec-banner {
        background: #FFF7ED;
        border-left: 3px solid #EA580C;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-top: 14px;
        font-size: 13.5px;
        color: #1E293B;
        line-height: 1.45;
    }

    /* Orange Buttons */
    .stButton button {
        background-color: #EA580C !important;
        background-image: linear-gradient(180deg, #EA580C 0%, #C2410C 100%) !important;
        color: #FFFFFF !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 2px 6px rgba(234, 88, 12, 0.28) !important;
        transition: all 0.2s ease !important;
    }

    .stButton button:hover {
        background-image: linear-gradient(180deg, #F97316 0%, #EA580C 100%) !important;
        box-shadow: 0 4px 14px rgba(234, 88, 12, 0.38) !important;
        transform: translateY(-1px) !important;
        color: #FFFFFF !important;
    }

    .stButton button:active {
        transform: translateY(0px) !important;
    }

    /* Sidebar Theme Polish */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    /* Active radio highlight in sidebar */
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background-color: #FFF7ED !important;
        border-radius: 8px !important;
        color: #EA580C !important;
        font-weight: 600 !important;
    }
    </style>
    """


def render_split_screen_login(error_msg: str = ""):
    """
    Renders the exact split-screen orange-themed login page matching the reference design.
    Eliminates raw HTML markdown indentation bugs and uses native Streamlit containers.
    """
    import streamlit as st

    login_css = (
        "<style>\n"
        "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');\n"
        "\n"
        "html, body, [class*=\"css\"], .stApp {\n"
        "    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;\n"
        "}\n"
        "\n"
        ".stApp, [data-testid=\"stAppViewContainer\"], [data-testid=\"stHeader\"] {\n"
        "    background-color: #F8FAFC !important;\n"
        "}\n"
        "\n"
        "header, footer, #MainMenu, .stDeployButton {\n"
        "    display: none !important;\n"
        "    visibility: hidden !important;\n"
        "}\n"
        "\n"
        ".block-container {\n"
        "    max-width: 1040px !important;\n"
        "    padding-top: 3.5rem !important;\n"
        "    padding-bottom: 2rem !important;\n"
        "}\n"
        "\n"
        "/* Orange Left Hero Box */\n"
        ".orange-hero-box, .orange-hero-card {\n"
        "    background: linear-gradient(160deg, #EA580C 0%, #C2410C 50%, #9A3412 100%) !important;\n"
        "    border-radius: 24px !important;\n"
        "    padding: 40px 32px !important;\n"
        "    color: #FFFFFF !important;\n"
        "    min-height: 540px !important;\n"
        "    display: flex !important;\n"
        "    flex-direction: column !important;\n"
        "    justify-content: space-between !important;\n"
        "    box-shadow: 0 12px 36px -8px rgba(194, 65, 12, 0.45) !important;\n"
        "    box-sizing: border-box !important;\n"
        "}\n"
        "\n"
        ".orange-brand-row, .orange-brand-header {\n"
        "    display: flex !important;\n"
        "    align-items: center !important;\n"
        "    gap: 12px !important;\n"
        "}\n"
        "\n"
        ".orange-brand-icon, .orange-icon-box {\n"
        "    width: 40px !important;\n"
        "    height: 40px !important;\n"
        "    border-radius: 10px !important;\n"
        "    background: rgba(255, 255, 255, 0.2) !important;\n"
        "    display: flex !important;\n"
        "    align-items: center !important;\n"
        "    justify-content: center !important;\n"
        "    font-size: 20px !important;\n"
        "    color: #FFFFFF !important;\n"
        "}\n"
        "\n"
        ".orange-brand-name, .orange-brand-title {\n"
        "    font-size: 18px !important;\n"
        "    font-weight: 800 !important;\n"
        "    color: #FFFFFF !important;\n"
        "    line-height: 1.1 !important;\n"
        "}\n"
        "\n"
        ".orange-brand-sub {\n"
        "    font-size: 12px !important;\n"
        "    color: rgba(255, 255, 255, 0.85) !important;\n"
        "    font-weight: 500 !important;\n"
        "}\n"
        "\n"
        ".orange-decor-circle, .orange-center-circle {\n"
        "    width: 110px !important;\n"
        "    height: 110px !important;\n"
        "    border-radius: 50% !important;\n"
        "    background: rgba(255, 255, 255, 0.12) !important;\n"
        "    border: 2px solid rgba(255, 255, 255, 0.25) !important;\n"
        "    margin: 32px auto 24px auto !important;\n"
        "    display: flex !important;\n"
        "    align-items: center !important;\n"
        "    justify-content: center !important;\n"
        "    font-size: 42px !important;\n"
        "    color: #FFFFFF !important;\n"
        "}\n"
        "\n"
        ".orange-headline, .orange-hero-heading {\n"
        "    font-size: 26px !important;\n"
        "    font-weight: 800 !important;\n"
        "    color: #FFFFFF !important;\n"
        "    text-align: center !important;\n"
        "    line-height: 1.3 !important;\n"
        "    letter-spacing: -0.02em !important;\n"
        "    margin-bottom: 14px !important;\n"
        "}\n"
        "\n"
        ".orange-support-text, .orange-hero-desc {\n"
        "    font-size: 13.5px !important;\n"
        "    color: rgba(255, 255, 255, 0.9) !important;\n"
        "    text-align: center !important;\n"
        "    line-height: 1.6 !important;\n"
        "    max-width: 330px !important;\n"
        "    margin: 0 auto !important;\n"
        "}\n"
        "\n"
        ".orange-footer-row, .orange-hero-footer {\n"
        "    border-top: 1px solid rgba(255, 255, 255, 0.22) !important;\n"
        "    padding-top: 18px !important;\n"
        "    margin-top: 28px !important;\n"
        "    display: flex !important;\n"
        "    justify-content: space-between !important;\n"
        "    font-size: 11.5px !important;\n"
        "    color: rgba(255, 255, 255, 0.82) !important;\n"
        "    font-weight: 600 !important;\n"
        "}\n"
        "\n"
        "/* Right Auth Card Styling */\n"
        "div[data-testid=\"stVerticalBlockBorderWrapper\"] {\n"
        "    background-color: #FFFFFF !important;\n"
        "    border: 1px solid #E2E8F0 !important;\n"
        "    border-radius: 24px !important;\n"
        "    padding: 32px 30px !important;\n"
        "    box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.04), 0 16px 32px -8px rgba(15, 23, 42, 0.07) !important;\n"
        "}\n"
        "\n"
        ".auth-header-title {\n"
        "    font-size: 26px !important;\n"
        "    font-weight: 800 !important;\n"
        "    color: #0F172A !important;\n"
        "    margin-bottom: 4px !important;\n"
        "    letter-spacing: -0.02em !important;\n"
        "}\n"
        "\n"
        ".auth-header-sub {\n"
        "    font-size: 13.5px !important;\n"
        "    color: #64748B !important;\n"
        "    margin-bottom: 20px !important;\n"
        "}\n"
        "\n"
        ".stTextInput label p {\n"
        "    font-size: 13px !important;\n"
        "    font-weight: 600 !important;\n"
        "    color: #334155 !important;\n"
        "    margin-bottom: 4px !important;\n"
        "}\n"
        "\n"
        ".stTextInput input {\n"
        "    border: 1px solid #CBD5E1 !important;\n"
        "    border-radius: 10px !important;\n"
        "    padding: 10px 14px !important;\n"
        "    font-size: 14px !important;\n"
        "    color: #0F172A !important;\n"
        "    background-color: #FFFFFF !important;\n"
        "}\n"
        "\n"
        ".stTextInput input:focus {\n"
        "    border-color: #EA580C !important;\n"
        "    box-shadow: 0 0 0 3px rgba(234, 88, 12, 0.15) !important;\n"
        "    outline: none !important;\n"
        "}\n"
        "\n"
        ".stCheckbox label p {\n"
        "    font-size: 13px !important;\n"
        "    color: #334155 !important;\n"
        "    font-weight: 500 !important;\n"
        "}\n"
        "\n"
        "/* Orange Sign In Button */\n"
        "div[data-testid=\"stButton\"] button {\n"
        "    background-color: #EA580C !important;\n"
        "    background-image: linear-gradient(180deg, #EA580C 0%, #C2410C 100%) !important;\n"
        "    color: #FFFFFF !important;\n"
        "    font-size: 15px !important;\n"
        "    font-weight: 700 !important;\n"
        "    padding: 12px 20px !important;\n"
        "    border-radius: 10px !important;\n"
        "    border: none !important;\n"
        "    box-shadow: 0 4px 12px rgba(234, 88, 12, 0.35) !important;\n"
        "    width: 100% !important;\n"
        "    margin-top: 10px !important;\n"
        "    transition: all 0.2s ease !important;\n"
        "}\n"
        "\n"
        "div[data-testid=\"stButton\"] button:hover {\n"
        "    background-image: linear-gradient(180deg, #F97316 0%, #EA580C 100%) !important;\n"
        "    box-shadow: 0 6px 18px rgba(234, 88, 12, 0.45) !important;\n"
        "    transform: translateY(-1px) !important;\n"
        "}\n"
        "\n"
        "/* Demo Box */\n"
        ".demo-box-container {\n"
        "    background: #F8FAFC !important;\n"
        "    border: 1px solid #E2E8F0 !important;\n"
        "    border-radius: 12px !important;\n"
        "    padding: 14px 16px !important;\n"
        "    margin-top: 20px !important;\n"
        "}\n"
        "\n"
        ".demo-box-label {\n"
        "    font-size: 10.5px !important;\n"
        "    font-weight: 700 !important;\n"
        "    color: #64748B !important;\n"
        "    text-transform: uppercase !important;\n"
        "    letter-spacing: 0.06em !important;\n"
        "    margin-bottom: 8px !important;\n"
        "}\n"
        "\n"
        ".demo-grid {\n"
        "    display: grid !important;\n"
        "    grid-template-columns: 1fr 1fr !important;\n"
        "    gap: 10px !important;\n"
        "}\n"
        "\n"
        ".demo-card {\n"
        "    background: #FFFFFF !important;\n"
        "    border: 1px solid #E2E8F0 !important;\n"
        "    border-radius: 8px !important;\n"
        "    padding: 8px 10px !important;\n"
        "    font-size: 12px !important;\n"
        "    color: #334155 !important;\n"
        "    line-height: 1.4 !important;\n"
        "}\n"
        "\n"
        ".demo-card strong {\n"
        "    display: block !important;\n"
        "    font-size: 10.5px !important;\n"
        "    text-transform: uppercase !important;\n"
        "    color: #0F172A !important;\n"
        "    letter-spacing: 0.04em !important;\n"
        "    margin-bottom: 2px !important;\n"
        "}\n"
        "\n"
        ".demo-card code {\n"
        "    font-size: 11px !important;\n"
        "    color: #C2410C !important;\n"
        "    background: #FFF7ED !important;\n"
        "    padding: 1px 4px !important;\n"
        "    border-radius: 4px !important;\n"
        "}\n"
        "</style>"
    )
    st.markdown(login_css, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.1], gap="large")

    with col_left:
        hero_html = (
            '<div class="orange-hero-box">'
            '<div>'
            '<div class="orange-brand-row">'
            '<div class="orange-brand-icon">&#127891;</div>'
            '<div>'
            '<div class="orange-brand-name">SkillTrack</div>'
            '<div class="orange-brand-sub">Engineering Cohort Hub</div>'
            '</div>'
            '</div>'
            '<div class="orange-decor-circle">&#9881;&#65039;</div>'
            '<div class="orange-headline">Build verified engineering capability.</div>'
            '<div class="orange-support-text">Structured technical milestones, PDF project submissions, objective MCQ grading, and mentor evaluations.</div>'
            '</div>'
            '<div class="orange-footer-row">'
            '<span>Curriculum Cohort 2026</span>'
            '<span>10 Milestones &bull; 6 Core Tracks</span>'
            '</div>'
            '</div>'
        )
        st.markdown(hero_html, unsafe_allow_html=True)

    with col_right:
        with st.container(border=True):
            brand_html = (
                '<div style="display:inline-flex;align-items:center;gap:8px;margin-bottom:16px;">'
                '<div style="width:26px;height:26px;border-radius:6px;background:#FFF7ED;border:1px solid #FFEDD5;display:flex;align-items:center;justify-content:center;color:#EA580C;font-size:14px;">&#127891;</div>'
                '<span style="font-size:14px;font-weight:700;color:#0F172A;">SkillTrack</span>'
                '</div>'
                '<div class="auth-header-title">Welcome back</div>'
                '<div class="auth-header-sub">Enter your credentials to login to your account</div>'
            )
            st.markdown(brand_html, unsafe_allow_html=True)

            username = st.text_input("Email or Username", placeholder="e.g. intern1 or mentor1", key="login_username")
            password = st.text_input("Password", type="password", placeholder="••••••••••••", key="login_password")

            st.checkbox("Remember me", value=True, key="login_remember")

            login_btn = st.button("Sign In", use_container_width=True, key="login_submit_btn")

            if error_msg:
                st.error(error_msg)

            demo_html = (
                '<div class="demo-box-container">'
                '<div class="demo-box-label">SUPPORTED DEMO ACCOUNTS</div>'
                '<div class="demo-grid">'
                '<div class="demo-card">'
                '<strong>INTERN ROLE</strong>'
                'User: <code>intern1</code><br>'
                'Pass: <code>intern123</code>'
                '</div>'
                '<div class="demo-card">'
                '<strong>MENTOR ROLE</strong>'
                'User: <code>mentor1</code><br>'
                'Pass: <code>mentor123</code>'
                '</div>'
                '</div>'
                '</div>'
            )
            st.markdown(demo_html, unsafe_allow_html=True)

        return username, password, login_btn
