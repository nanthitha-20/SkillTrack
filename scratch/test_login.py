import streamlit as st

st.set_page_config(page_title="SkillTrack Login Test", layout="wide")

# CSS Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F8FAFC !important;
    }

    header, footer, #MainMenu, .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }

    .block-container {
        max-width: 1040px !important;
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
    }

    /* Orange Left Hero Panel */
    .orange-hero-box {
        background: linear-gradient(180deg, #EA580C 0%, #C2410C 50%, #9A3412 100%);
        border-radius: 24px;
        padding: 40px 32px;
        color: #FFFFFF !important;
        min-height: 540px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 12px 36px -8px rgba(194, 65, 12, 0.45);
        box-sizing: border-box;
    }

    .orange-brand-row {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .orange-brand-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }

    .orange-brand-name {
        font-size: 18px;
        font-weight: 800;
        color: #FFFFFF !important;
        line-height: 1.1;
    }

    .orange-brand-sub {
        font-size: 12px;
        color: rgba(255, 255, 255, 0.85) !important;
        font-weight: 500;
    }

    .orange-decor-circle {
        width: 110px;
        height: 110px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.12);
        border: 2px solid rgba(255, 255, 255, 0.25);
        margin: 32px auto 24px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 42px;
    }

    .orange-headline {
        font-size: 26px;
        font-weight: 800;
        color: #FFFFFF !important;
        text-align: center;
        line-height: 1.3;
        letter-spacing: -0.02em;
        margin-bottom: 14px;
    }

    .orange-support-text {
        font-size: 13.5px;
        color: rgba(255, 255, 255, 0.9) !important;
        text-align: center;
        line-height: 1.6;
        max-width: 330px;
        margin: 0 auto;
    }

    .orange-footer-row {
        border-top: 1px solid rgba(255, 255, 255, 0.22);
        padding-top: 18px;
        margin-top: 28px;
        display: flex;
        justify-content: space-between;
        font-size: 11.5px;
        color: rgba(255, 255, 255, 0.82) !important;
        font-weight: 600;
    }

    /* Right Auth Card Styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 24px !important;
        padding: 32px 30px !important;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.04), 0 16px 32px -8px rgba(15, 23, 42, 0.07) !important;
    }

    .auth-header-title {
        font-size: 26px;
        font-weight: 800;
        color: #0F172A !important;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }

    .auth-header-sub {
        font-size: 13.5px;
        color: #64748B !important;
        margin-bottom: 20px;
    }

    .stTextInput label p {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #334155 !important;
        margin-bottom: 4px !important;
    }

    .stTextInput input {
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        font-size: 14px !important;
        color: #0F172A !important;
        background-color: #FFFFFF !important;
    }

    .stTextInput input:focus {
        border-color: #EA580C !important;
        box-shadow: 0 0 0 3px rgba(234, 88, 12, 0.15) !important;
        outline: none !important;
    }

    .stCheckbox label p {
        font-size: 13px !important;
        color: #334155 !important;
        font-weight: 500 !important;
    }

    /* Orange Sign In Button */
    div[data-testid="stButton"] button {
        background-color: #EA580C !important;
        background-image: linear-gradient(180deg, #EA580C 0%, #C2410C 100%) !important;
        color: #FFFFFF !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        padding: 12px 20px !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(234, 88, 12, 0.35) !important;
        width: 100% !important;
        margin-top: 10px !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stButton"] button:hover {
        background-image: linear-gradient(180deg, #F97316 0%, #EA580C 100%) !important;
        box-shadow: 0 6px 18px rgba(234, 88, 12, 0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* Demo Box */
    .demo-box-container {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 20px;
    }

    .demo-box-label {
        font-size: 10.5px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }

    .demo-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }

    .demo-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 8px 10px;
        font-size: 12px;
        color: #334155;
        line-height: 1.4;
    }

    .demo-card strong {
        display: block;
        font-size: 10.5px;
        text-transform: uppercase;
        color: #0F172A;
        letter-spacing: 0.04em;
        margin-bottom: 2px;
    }

    .demo-card code {
        font-size: 11px;
        color: #C2410C;
        background: #FFF7ED;
        padding: 1px 4px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col_left, col_right = st.columns([1, 1.1], gap="large")

with col_left:
    hero_html = (
        '<div class="orange-hero-box">'
        '<div>'
        '<div class="orange-brand-row">'
        '<div class="orange-brand-icon">&#127891;</div>'
        '<div><div class="orange-brand-name">SkillTrack</div><div class="orange-brand-sub">Engineering Cohort Hub</div></div>'
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
        st.markdown(
            '<div style="display:inline-flex;align-items:center;gap:8px;margin-bottom:16px;">'
            '<div style="width:26px;height:26px;border-radius:6px;background:#FFF7ED;border:1px solid #FFEDD5;display:flex;align-items:center;justify-content:center;color:#EA580C;font-size:14px;">&#127891;</div>'
            '<span style="font-size:14px;font-weight:700;color:#0F172A;">SkillTrack</span>'
            '</div>'
            '<div class="auth-header-title">Welcome back</div>'
            '<div class="auth-header-sub">Enter your credentials to login to your account</div>',
            unsafe_allow_html=True
        )

        username = st.text_input("Email or Username", placeholder="e.g. intern1 or mentor1", key="login_username")
        password = st.text_input("Password", type="password", placeholder="••••••••••••", key="login_password")
        remember = st.checkbox("Remember me", value=True, key="login_remember")
        login_btn = st.button("Sign In", use_container_width=True, key="login_btn")

        st.markdown(
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
            '</div>',
            unsafe_allow_html=True
        )
