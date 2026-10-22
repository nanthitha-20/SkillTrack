# pyrefly: ignore [missing-import]
import streamlit as st
import sqlite3
import subprocess
import tempfile
import shutil
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ML Forecast Module, Online Learning, Chat & Resource Services
try:
    from src.ml.predict import get_intern_ml_forecast
    from src.data.live_data_ingestion import (
        ingest_observation,
        get_observation_summary,
        get_observations_for_intern,
        create_pending_forecasting_sample,
        resolve_pending_forecasting_labels,
        get_recent_dataset_records
    )
    from src.ml.model_registry import (
        get_active_production_model,
        check_retraining_eligibility
    )
    from src.ml.drift_monitor import assess_system_data_drift
    from src.ml.online_learner import (
        get_online_learner,
        get_online_learning_status
    )
    from src.data.chat_service import (
        send_chat_message,
        get_conversation_history,
        mark_messages_as_read,
        get_mentor_unread_counts,
        get_chat_engagement_metadata
    )
    from src.data.resource_service import (
        get_curated_resources,
        get_available_filters,
        get_recommended_resources
    )
    from src.ml.prediction_tracer import (
        diff_prediction_traces,
        diff_feature_snapshots,
        get_recent_prediction_traces,
        get_trace_by_id,
        get_traces_for_intern,
        get_annotated_recent_observations,
        get_observation_lineage_details,
        get_intern_live_ml_status,
        get_mentor_live_ml_kpis
    )
    from src.ml.live_trace_demo import run_live_ml_trace_demo
    from src.data.questionnaire_service import (
        QUESTIONNAIRE_QUESTIONS,
        save_questionnaire_response,
        get_questionnaire_response,
        get_all_questionnaire_responses,
        build_feature_inspection_table,
        QuestionnaireValidationError
    )
    from src.ml.benchmark import (
        run_historical_benchmark,
        run_live_data_benchmark,
        get_dataset_audit,
        save_benchmark_results,
        load_saved_benchmark_results
    )
    from src.data.assessment_service import (
        validate_and_save_assessment,
        get_intern_assessments,
        get_assessment_summary_stats,
        AssessmentValidationError
    )
    from src.ml.hiring_engine import (
        compute_hiring_recommendation,
        save_mentor_hiring_decision,
        get_latest_hiring_decision,
        RUBRIC_WEIGHTS
    )
    from src.data.live_dataset_view import render_live_dataset_page
    from src.ml.live_ml_lab_view import render_live_ml_lab_page
    from src.ml.e2e_demo_view import render_e2e_demo_page
    from src.data.deliverable_views import (
        render_intern_deliverables_page,
        render_mentor_review_hub,
        render_pdf_preview
    )
    from src.data.orange_theme import (
        get_orange_theme_css,
        render_split_screen_login
    )
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
except ImportError as ex:
    print(f"[SkillTrack-ML ERROR] Failed to import ML modules: {ex}")
    import traceback
    traceback.print_exc()
    get_intern_ml_forecast = None
    ingest_observation = None
    get_observation_summary = None
    get_observations_for_intern = None
    create_pending_forecasting_sample = None
    resolve_pending_forecasting_labels = None
    get_recent_dataset_records = None
    get_active_production_model = None
    check_retraining_eligibility = None
    assess_system_data_drift = None
    get_online_learner = None
    get_online_learning_status = None
    send_chat_message = None
    get_conversation_history = None
    mark_messages_as_read = None
    get_mentor_unread_counts = None
    get_chat_engagement_metadata = None
    get_curated_resources = None
    get_available_filters = None
    get_recommended_resources = None
    diff_prediction_traces = None
    diff_feature_snapshots = None
    get_recent_prediction_traces = None
    get_trace_by_id = None
    get_traces_for_intern = None
    get_annotated_recent_observations = None
    get_observation_lineage_details = None
    get_intern_live_ml_status = None
    get_mentor_live_ml_kpis = None
    run_live_ml_trace_demo = None
    QUESTIONNAIRE_QUESTIONS = []
    save_questionnaire_response = None
    get_questionnaire_response = None
    get_all_questionnaire_responses = None
    build_feature_inspection_table = None
    QuestionnaireValidationError = Exception
    run_historical_benchmark = None
    run_live_data_benchmark = None
    get_dataset_audit = None
    save_benchmark_results = None
    load_saved_benchmark_results = None
    validate_and_save_assessment = None
    get_intern_assessments = None
    get_assessment_summary_stats = None
    AssessmentValidationError = Exception
    compute_hiring_recommendation = None
    save_mentor_hiring_decision = None
    get_latest_hiring_decision = None
    RUBRIC_WEIGHTS = {}
    render_live_dataset_page = None
    render_live_ml_lab_page = None
    render_e2e_demo_page = None
    render_intern_deliverables_page = None
    render_mentor_review_hub = None
    render_pdf_preview = None
    get_orange_theme_css = None
    render_split_screen_login = None
    get_all_milestones = None
    get_milestone_by_id = None
    get_intern_deliverable_history = None
    save_pdf_deliverable = None
    record_mentor_deliverable_review = None
    get_all_submissions_for_mentor = None
    get_submission_by_id = None
    get_cohort_deliverable_metrics = None
    DeliverableValidationError = Exception


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SkillTrack - Internship Management & Evaluation Portal",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


# ============================================================
# TASK-SPECIFIC LEARNING VIDEOS
# You can replace these URLs later.
# ============================================================

TASK_VIDEOS = {
    "Python Basics":
        "https://www.youtube.com/watch?v=rfscVS0vtbw",

    "String Manipulation":
        "https://www.youtube.com/watch?v=rfscVS0vtbw",

    "Data Structures":
        "https://www.youtube.com/watch?v=rfscVS0vtbw",

    "SQL Basics":
        "https://www.youtube.com/watch?v=HXV3zeQKqGY",

    "REST API Task":
        "https://www.youtube.com/watch?v=-MTSQjw5DrM"
}


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def ensure_submission_columns():
    """
    Adds code/language/output columns to the existing
    submissions table if they don't already exist.
    """

    connection = get_connection()
    cursor = connection.cursor()

    columns = cursor.execute(
        "PRAGMA table_info(submissions)"
    ).fetchall()

    column_names = [column[1] for column in columns]

    if "language" not in column_names:
        cursor.execute(
            "ALTER TABLE submissions ADD COLUMN language TEXT"
        )

    if "code" not in column_names:
        cursor.execute(
            "ALTER TABLE submissions ADD COLUMN code TEXT"
        )

    if "output" not in column_names:
        cursor.execute(
            "ALTER TABLE submissions ADD COLUMN output TEXT"
        )

    connection.commit()
    connection.close()


ensure_submission_columns()


def authenticate_user(username, password):

    connection = get_connection()

    user = connection.execute(
        """
        SELECT user_id, username, role, intern_id, mentor_id
        FROM users
        WHERE username = ? AND password = ?
        """,
        (username, password)
    ).fetchone()

    connection.close()

    return user


def get_intern(intern_id):

    connection = get_connection()

    intern = connection.execute(
        """
        SELECT
            intern_id,
            name,
            email,
            role,
            experience_level,
            current_week,
            average_score,
            skill_score,
            final_performance_score,
            skill_gap,
            resource_action,
            progress
        FROM interns
        WHERE intern_id = ?
        """,
        (intern_id,)
    ).fetchone()

    connection.close()

    return intern

def get_coding_tasks():

    connection = get_connection()

    tasks = connection.execute(
        """
        SELECT
            task_id,
            title,
            description,
            max_score,
            skill,
            difficulty,
            video_url,
            guide_url
        FROM coding_tasks
        ORDER BY task_id
        """
    ).fetchall()

    connection.close()

    return tasks


def get_intern_submissions(intern_id):

    connection = get_connection()

    submissions = connection.execute(
        """
        SELECT
            s.submission_id,
            s.task_id,
            t.title,
            s.score,
            s.status,
            s.submitted_at,
            s.language,
            s.code,
            s.output
        FROM submissions s
        LEFT JOIN coding_tasks t
            ON s.task_id = t.task_id
        WHERE s.intern_id = ?
        ORDER BY s.submission_id DESC
        """,
        (intern_id,)
    ).fetchall()

    connection.close()

    return submissions


def save_code_submission(
    intern_id,
    task_id,
    language,
    code,
    output
):

    connection = get_connection()

    connection.execute(
        """
        INSERT INTO submissions
        (
            intern_id,
            task_id,
            score,
            status,
            submitted_at,
            language,
            code,
            output
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            intern_id,
            task_id,
            None,
            "SUBMITTED",
            datetime.now().isoformat(timespec="seconds"),
            language,
            code,
            output
        )
    )

    connection.commit()
    connection.close()

    if ingest_observation:
        try:
            ingest_observation(
                intern_id=intern_id,
                week=10,
                source="coding_submission",
                event_type="task_submitted",
                metric_name="coding_score",
                metric_value=85.0,
                payload={"task_id": task_id, "language": language, "status": "SUBMITTED"},
                update_active_performance=True
            )
        except Exception:
            pass


def get_mcq_attempts(intern_id):

    connection = get_connection()

    attempts = connection.execute(
        """
        SELECT
            attempt_id,
            score,
            total_questions,
            topic,
            attempted_at
        FROM mcq_attempts
        WHERE intern_id = ?
        ORDER BY attempt_id DESC
        """,
        (intern_id,)
    ).fetchall()

    connection.close()

    return attempts


def save_mcq_attempt(intern_id, score, total_questions, topic):
    connection = get_connection()
    connection.execute(
        """
        INSERT INTO mcq_attempts (intern_id, score, total_questions, topic, attempted_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            intern_id,
            score,
            total_questions,
            topic,
            datetime.now().isoformat(timespec="seconds")
        )
    )
    connection.commit()
    connection.close()

    if ingest_observation:
        try:
            ingest_observation(
                intern_id=intern_id,
                week=10,
                source="quiz",
                event_type="mcq_attempted",
                metric_name="mcq_score",
                metric_value=float(score),
                payload={"total_questions": total_questions, "topic": topic},
                update_active_performance=True
            )
        except Exception:
            pass


def get_intern_performance_history(intern_id):
    connection = get_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT * FROM intern_performance
            WHERE intern_id = ?
            ORDER BY week ASC
            """,
            connection,
            params=(intern_id,)
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        connection.close()
    return df


def get_resources():

    connection = get_connection()

    resources = connection.execute(
        """
        SELECT
            resource_id,
            title,
            resource_type,
            url,
            topic,
            difficulty
        FROM resources
        ORDER BY resource_id
        """
    ).fetchall()

    connection.close()

    return resources


def get_all_interns():

    connection = get_connection()

    interns = connection.execute(
        """
        SELECT
            intern_id,
            name,
            email,
            average_score,
            progress
        FROM interns
        ORDER BY intern_id
        """
    ).fetchall()

    connection.close()

    return interns


def get_intern_evaluation(intern_id):

    connection = get_connection()

    evaluation = connection.execute(
        """
        SELECT
            technical_score,
            communication_score,
            discipline_score,
            overall_score,
            decision,
            remarks,
            evaluated_at
        FROM mentor_evaluations
        WHERE intern_id = ?
        ORDER BY evaluation_id DESC
        LIMIT 1
        """,
        (intern_id,)
    ).fetchone()

    connection.close()

    return evaluation


def get_evaluation_counts():

    connection = get_connection()

    results = connection.execute(
        """
        SELECT decision, COUNT(*)
        FROM mentor_evaluations
        GROUP BY decision
        """
    ).fetchall()

    connection.close()

    return dict(results)


def save_evaluation(
    intern_id,
    mentor_id,
    technical_score,
    communication_score,
    discipline_score,
    overall_score,
    decision,
    remarks
):

    connection = get_connection()

    connection.execute(
        """
        INSERT INTO mentor_evaluations
        (
            intern_id,
            mentor_id,
            technical_score,
            communication_score,
            discipline_score,
            overall_score,
            decision,
            remarks,
            evaluated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            intern_id,
            mentor_id,
            technical_score,
            communication_score,
            discipline_score,
            overall_score,
            decision,
            remarks,
            datetime.now().isoformat(timespec="seconds")
        )
    )

    connection.commit()
    connection.close()

    if ingest_observation:
        try:
            ingest_observation(
                intern_id=intern_id,
                week=10,
                source="mentor_evaluation",
                event_type="mentor_evaluated",
                metric_name="mentor_overall_score",
                metric_value=float(overall_score),
                payload={"mentor_id": mentor_id, "decision": decision, "remarks": remarks},
                update_active_performance=False
            )
        except Exception:
            pass


def get_cohort_summary_df():
    connection = get_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT
                intern_id,
                name,
                email,
                role,
                experience_level,
                current_week,
                average_score,
                skill_score,
                final_performance_score,
                skill_gap,
                resource_action,
                progress
            FROM interns
            ORDER BY intern_id ASC
            """,
            connection
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        connection.close()
    return df


def get_all_mentor_evaluations():
    connection = get_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT
                e.evaluation_id,
                e.intern_id,
                i.name as intern_name,
                e.technical_score,
                e.communication_score,
                e.discipline_score,
                e.overall_score,
                e.decision,
                e.remarks,
                e.evaluated_at
            FROM mentor_evaluations e
            LEFT JOIN interns i ON e.intern_id = i.intern_id
            ORDER BY e.evaluation_id DESC
            """,
            connection
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        connection.close()
    return df


def get_cohort_performance_df():
    connection = get_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT
                week,
                skill,
                AVG(final_performance_score) as avg_final_score,
                AVG(skill_score) as avg_skill_score,
                AVG(coding_score) as avg_coding_score,
                AVG(mcq_score) as avg_mcq_score,
                AVG(debugging_score) as avg_debugging_score,
                AVG(compile_error_rate) as avg_compile_err,
                AVG(runtime_error_rate) as avg_runtime_err,
                SUM(test_cases_passed) as total_tests_passed,
                SUM(test_cases_total) as total_tests_count
            FROM intern_performance
            GROUP BY week, skill
            ORDER BY week ASC
            """,
            connection
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        connection.close()
    return df


# ============================================================
# CODE RUNNER
# ============================================================

def command_exists(command):
    return shutil.which(command) is not None


def run_process(command, working_directory, stdin_data=""):

    try:

        result = subprocess.run(
            command,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=6,
            cwd=working_directory
        )

        return (
            result.returncode == 0,
            result.stdout.strip(),
            result.stderr.strip()
        )

    except subprocess.TimeoutExpired:

        return (
            False,
            "",
            "Execution timed out after 6 seconds."
        )

    except Exception as error:

        return (
            False,
            "",
            str(error)
        )


def run_code(code, language, stdin_data=""):

    if not code.strip():

        return (
            False,
            "",
            "Please enter code before running."
        )

    temp_directory = tempfile.mkdtemp()

    try:

        # ----------------------------------------------------
        # PYTHON
        # ----------------------------------------------------

        if language == "Python":

            source_file = Path(temp_directory) / "solution.py"

            source_file.write_text(
                code,
                encoding="utf-8"
            )

            return run_process(
                [sys.executable, str(source_file)],
                temp_directory,
                stdin_data
            )

        # ----------------------------------------------------
        # JAVASCRIPT
        # ----------------------------------------------------

        elif language == "JavaScript":

            if not command_exists("node"):

                return (
                    False,
                    "",
                    "Node.js is not installed. "
                    "Install Node.js to run JavaScript."
                )

            source_file = Path(temp_directory) / "solution.js"

            source_file.write_text(
                code,
                encoding="utf-8"
            )

            return run_process(
                ["node", str(source_file)],
                temp_directory,
                stdin_data
            )

        # ----------------------------------------------------
        # JAVA
        # ----------------------------------------------------

        elif language == "Java":

            if not command_exists("javac"):

                return (
                    False,
                    "",
                    "Java JDK is not installed or javac "
                    "is not available in PATH."
                )

            source_file = Path(temp_directory) / "Main.java"

            source_file.write_text(
                code,
                encoding="utf-8"
            )

            compile_success, compile_output, compile_error = (
                run_process(
                    ["javac", "Main.java"],
                    temp_directory
                )
            )

            if not compile_success:

                return (
                    False,
                    compile_output,
                    compile_error
                )

            return run_process(
                ["java", "Main"],
                temp_directory,
                stdin_data
            )

        # ----------------------------------------------------
        # C++
        # ----------------------------------------------------

        elif language == "C++":

            if not command_exists("g++"):

                return (
                    False,
                    "",
                    "g++ is not installed or available in PATH."
                )

            source_file = Path(temp_directory) / "solution.cpp"

            executable_name = (
                "solution.exe"
                if os.name == "nt"
                else "solution"
            )

            source_file.write_text(
                code,
                encoding="utf-8"
            )

            compile_success, compile_output, compile_error = (
                run_process(
                    [
                        "g++",
                        "solution.cpp",
                        "-o",
                        executable_name
                    ],
                    temp_directory
                )
            )

            if not compile_success:

                return (
                    False,
                    compile_output,
                    compile_error
                )

            executable_path = (
                Path(temp_directory)
                / executable_name
            )

            return run_process(
                [str(executable_path)],
                temp_directory,
                stdin_data
            )

        return (
            False,
            "",
            "Unsupported language."
        )

    finally:

        shutil.rmtree(
            temp_directory,
            ignore_errors=True
        )


# ============================================================
# STARTER CODE
# ============================================================

def starter_code(language):

    if language == "Python":

        return """# Write your solution here

numbers = [10, 5, 20, 8]

print(max(numbers))
"""

    if language == "JavaScript":

        return """// Write your solution here

const numbers = [10, 5, 20, 8];

console.log(Math.max(...numbers));
"""

    if language == "Java":

        return """public class Main {

    public static void main(String[] args) {

        int[] numbers = {10, 5, 20, 8};

        int max = numbers[0];

        for (int number : numbers) {
            if (number > max) {
                max = number;
            }
        }

        System.out.println(max);
    }
}
"""

    if language == "C++":

        return """#include <iostream>
using namespace std;

int main() {

    int numbers[] = {10, 5, 20, 8};

    int maxValue = numbers[0];

    for (int number : numbers) {
        if (number > maxValue) {
            maxValue = number;
        }
    }

    cout << maxValue << endl;

    return 0;
}
"""

    return ""


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None


# ============================================================
# LOGIN PAGE
# ============================================================

if not st.session_state.logged_in:
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


# ============================================================
# CURRENT USER
# ============================================================

current_user = st.session_state.user
if not current_user:
    st.stop()
    sys.exit(0)

user_id = current_user[0]
username = current_user[1]
role = current_user[2]
intern_id = current_user[3]
mentor_id = current_user[4]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; padding: 6px 0 16px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 16px;">
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
    </div>
    """,
    unsafe_allow_html=True
)

if role == "intern":
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
    )
elif role == "mentor":
    nav_page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Deliverables",
            "Internship Lifecycle Demo",
            "Performance Analytics",
            "Assessment Records",
            "Messages",
            "Interns",
            "Performance",
            "Skill Gaps",
            "Reports",
            "Forecast Evaluation",
            "Profile"
        ],
        label_visibility="collapsed"
    )
else:
    nav_page = "Dashboard"

st.sidebar.markdown(
    f"""
    <div style="margin-top: 28px; padding-top: 16px; border-top: 1px solid #E2E8F0;">
        <div style="font-size: 10.5px; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">Account</div>
        <div style="font-size: 13px; font-weight: 600; color: #0F172A;">{username}</div>
        <div style="font-size: 11.5px; color: #EA580C; font-weight: 600; margin-bottom: 12px;">{role.title()} Role</div>
    </div>
    """,
    unsafe_allow_html=True
)

if st.sidebar.button("Sign Out", use_container_width=True, key="sign_out_btn"):
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()


# ============================================================
# INTERN PORTAL
# ============================================================

if role == "intern":

    intern = get_intern(intern_id)

    if intern is None:
        st.error("Intern profile could not be found.")
        st.stop()

    (
        intern_id,
        name,
        email,
        intern_role,
        experience_level,
        current_week,
        average_score,
        skill_score,
        final_performance_score,
        skill_gap,
        resource_action,
        stored_progress
    ) = intern

    submissions = get_intern_submissions(intern_id)
    mcq_attempts = get_mcq_attempts(intern_id)
    tasks = get_coding_tasks()
    resources = get_resources()
    perf_history_df = get_intern_performance_history(intern_id)

    completed_tasks = len(set(submission[1] for submission in submissions))
    total_tasks = len(tasks)
    task_progress = (completed_tasks / total_tasks * 100) if total_tasks else 0

    mcq_scores = [attempt[1] for attempt in mcq_attempts if attempt[1] is not None]
    mcq_average = (sum(mcq_scores) / len(mcq_scores)) if mcq_scores else 0

    if task_progress and mcq_average:
        overall_progress = (task_progress * 0.5 + mcq_average * 0.5)
    elif task_progress:
        overall_progress = task_progress
    elif mcq_average:
        overall_progress = mcq_average
    else:
        overall_progress = stored_progress or 0

    # Orange Theme Injection
    st.markdown(get_orange_theme_css(), unsafe_allow_html=True)
    # --------------------------------------------------------
    # PAGE 1: DASHBOARD
    # --------------------------------------------------------
    if nav_page == "Dashboard":

        st.markdown(
            f"""
            <div class="hero-banner">
                <div class="page-eyebrow">YOUR PROGRESS</div>
                <div class="page-heading">Welcome back, {name}</div>
                <div class="page-subheading">Track your coding, assessments, learning activity, and skill development in one place.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Overview Metrics (4 Compact Cards)
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-label">Overall Progress</div>
                    <div class="metric-card-value">{overall_progress:.1f}%</div>
                    <div class="metric-card-caption">Combined curriculum score</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        deliverable_hist = get_intern_deliverable_history(intern_id) if get_intern_deliverable_history else []
        approved_delivs = len([d for d in deliverable_hist if d['review_status'] == 'APPROVED'])
        total_milestones = 10

        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-label">Project Deliverables</div>
                    <div class="metric-card-value">{len(deliverable_hist)} Submitted</div>
                    <div class="metric-card-caption">{approved_delivs} of {total_milestones} milestones approved</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m3:
            avg_str = f"{mcq_average:.1f}%" if mcq_average else "N/A"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-label">Assessment Score</div>
                    <div class="metric-card-value">{avg_str}</div>
                    <div class="metric-card-caption">{len(mcq_attempts)} quizzes attempted</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-label">Active Module</div>
                    <div class="metric-card-value">Week {current_week or 10}</div>
                    <div class="metric-card-caption">{intern_role or 'Frontend Development'}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        # Background sample bookkeeping (silent backend persistence)
        if create_pending_forecasting_sample:
            try:
                create_pending_forecasting_sample(intern_id, current_week or 10)
                if resolve_pending_forecasting_labels:
                    resolve_pending_forecasting_labels()
            except Exception:
                pass

        # Suggested Development Actions
        if resource_action:
            st.markdown(
                f"""
                <div class="section-card" style="border-left: 4px solid #EA580C; margin-bottom: 1.25rem;">
                    <div class="page-eyebrow" style="color: #EA580C; margin-bottom: 2px;">DEVELOPMENT FOCUS</div>
                    <div class="section-card-title">Suggested Development Action</div>
                    <div class="section-card-subtitle">{resource_action}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        # Two Column Layout: Skill Competency Snapshot & Recent Activity
        col_skills, col_act = st.columns([1.1, 1], gap="medium")

        with col_skills:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-card-title">Skill Competency Snapshot</div>
                    <div class="section-card-subtitle">Real-time status across evaluated frontend modules.</div>
                """,
                unsafe_allow_html=True
            )

            if not perf_history_df.empty:
                recent_perf = perf_history_df.tail(4)
                for _, row in recent_perf.iterrows():
                    sk_name = row.get("skill", "Core Skill")
                    sk_score = float(row.get("skill_score", 0))
                    sk_gap = row.get("skill_gap", "Good")
                    badge_cls = "badge-strong" if sk_gap == "Strong" else ("badge-developing" if sk_gap == "Developing" else "badge-needs-improvement")

                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #F1F5F9;">
                            <div>
                                <div style="font-weight: 600; font-size: 13.5px; color: #0F172A;">{sk_name}</div>
                                <div style="font-size: 12px; color: #64748B;">Week {int(row.get('week', 1))} • Score: {sk_score:.1f}</div>
                            </div>
                            <div class="{badge_cls}">{sk_gap}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No weekly skill records recorded yet.")

            st.markdown("</div>", unsafe_allow_html=True)

        with col_act:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-card-title">Recent Submissions</div>
                    <div class="section-card-subtitle">Your latest submitted technical solutions.</div>
                """,
                unsafe_allow_html=True
            )

            if deliverable_hist:
                for sub in deliverable_hist[:4]:
                    sub_title = sub['title']
                    sub_time = sub['submitted_at']
                    sub_status = sub['review_status']
                    badge_c = "badge-approved" if sub_status == "APPROVED" else ("badge-changes-requested" if sub_status == "CHANGES_REQUESTED" else "badge-submitted")
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #F1F5F9;">
                            <div>
                                <div style="font-weight: 600; font-size: 13.5px; color: #0F172A;">{sub_title}</div>
                                <div style="font-size: 12px; color: #64748B;">Version {sub['version_number']} • {sub_time}</div>
                            </div>
                            <span class="{badge_c}">{sub_status.replace('_', ' ')}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No project deliverables submitted yet.")

            st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PAGE 2: MY SKILLS
    # --------------------------------------------------------
    elif nav_page == "My Skills":

        st.markdown(
            """
            <div>
                <div class="page-eyebrow">TECHNICAL MASTERY</div>
                <div class="page-heading">My Skills</div>
                <div class="page-subheading">Comprehensive evaluation of your frontend competencies across curriculum milestones.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if not perf_history_df.empty:
            skills_grouped = perf_history_df.groupby("skill").last().reset_index()

            for _, row in skills_grouped.iterrows():
                sk_name = row["skill"]
                sk_score = float(row.get("skill_score", 0))
                sk_gap = row.get("skill_gap", "Good")
                code_score = float(row.get("coding_score", 0))
                debug_score = float(row.get("debugging_score", 0))
                test_passed = int(row.get("test_cases_passed", 0))
                test_total = int(row.get("test_cases_total", 0))
                pass_rate = (test_passed / test_total * 100) if test_total > 0 else 0
                badge_cls = "badge-strong" if sk_gap == "Strong" else ("badge-developing" if sk_gap == "Developing" else "badge-needs-improvement")

                st.markdown(
                    f"""
                    <div class="section-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <div>
                                <span style="font-size: 18px; font-weight: 700; color: #0F172A;">{sk_name}</span>
                                <span style="font-size: 13px; color: #64748B; margin-left: 8px;">Week {int(row.get('week', 1))}</span>
                            </div>
                            <div class="{badge_cls}">{sk_gap}</div>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-top: 10px;">
                            <div class="metric-card">
                                <div class="metric-card-label">Skill Score</div>
                                <div class="metric-card-value" style="font-size: 18px;">{sk_score:.1f}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-card-label">Coding Score</div>
                                <div class="metric-card-value" style="font-size: 18px;">{code_score:.1f}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-card-label">Debugging</div>
                                <div class="metric-card-value" style="font-size: 18px;">{debug_score:.1f}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-card-label">Test Pass Rate</div>
                                <div class="metric-card-value" style="font-size: 18px;">{pass_rate:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No skill evaluations recorded yet.")

    # --------------------------------------------------------
    # PAGE 3: PROJECT DELIVERABLES WORKSPACE
    # --------------------------------------------------------
    elif nav_page == "Deliverables":
        render_intern_deliverables_page(intern_id)

    # --------------------------------------------------------
    # PAGE 4: ASSESSMENTS
    # --------------------------------------------------------
    elif nav_page == "Assessments":

        st.markdown(
            """
            <div>
                <div class="page-eyebrow">EVALUATION</div>
                <div class="page-heading">Assessments & Quizzes</div>
                <div class="page-subheading">Test your technical comprehension and review performance history across topics.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        q_topics = {
            "JavaScript Core": [
                {
                    "q": "Which method serializes a JavaScript object into a standard JSON formatted string?",
                    "options": ["JSON.stringify()", "JSON.parse()", "Object.toJSON()", "String.serialize()"],
                    "answer": 0
                },
                {
                    "q": "What is the evaluated output of typeof NaN in JavaScript?",
                    "options": ["'number'", "'NaN'", "'undefined'", "'object'"],
                    "answer": 0
                }
            ],
            "React & State Management": [
                {
                    "q": "Which React hook is designed for managing component lifecycle side effects?",
                    "options": ["useEffect", "useState", "useContext", "useReducer"],
                    "answer": 0
                },
                {
                    "q": "What is the best practice rule when updating state arrays or objects in React?",
                    "options": ["Create an immutable copy with new reference", "Directly push or mutate the state variable", "Mutate state synchronously via document object", "State cannot hold objects"],
                    "answer": 0
                }
            ],
            "CSS & Responsive Design": [
                {
                    "q": "Which CSS flexbox property determines how much an item grows relative to siblings?",
                    "options": ["flex-grow", "flex-basis", "flex-shrink", "align-items"],
                    "answer": 0
                }
            ]
        }

        selected_topic = st.selectbox("Select Assessment Topic", list(q_topics.keys()))
        questions = q_topics[selected_topic]

        with st.form("quiz_form"):
            st.markdown(f"### {selected_topic} Assessment")
            answers = []
            for idx, q_data in enumerate(questions):
                st.write(f"**Question {idx + 1}:** {q_data['q']}")
                ans = st.radio(f"Select option for Q{idx+1}:", q_data["options"], key=f"q_{selected_topic}_{idx}")
                answers.append(ans)
                st.divider()

            submit_quiz = st.form_submit_button("Submit Assessment", use_container_width=True)

            if submit_quiz:
                correct = sum(1 for i, q_data in enumerate(questions) if answers[i] == q_data["options"][q_data["answer"]])
                score_pct = (correct / len(questions)) * 100
                save_mcq_attempt(intern_id, score_pct, len(questions), selected_topic)
                st.success(f"Assessment completed! Your Score: {score_pct:.1f}% ({correct}/{len(questions)} correct)")
                st.rerun()

        # Attempt History Table
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-card">
                <div class="section-card-title">Assessment History</div>
                <div class="section-card-subtitle">Your previous quiz and assessment attempts.</div>
            """,
            unsafe_allow_html=True
        )

        if mcq_attempts:
            for attempt in mcq_attempts[:6]:
                topic_name = attempt[3] or "Technical Quiz"
                att_score = attempt[1]
                att_time = attempt[4]
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #F1F5F9;">
                        <div>
                            <div style="font-weight: 600; font-size: 14px; color: #0F172A;">{topic_name}</div>
                            <div style="font-size: 12px; color: #64748B;">Attempted: {att_time}</div>
                        </div>
                        <div class="badge-developing" style="font-size: 13px;">{att_score}% Score</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.caption("No assessment attempts recorded yet.")

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PAGE 5: LEARNING (CURRICULUM & ML-GUIDED RECOMMENDATIONS)
    # --------------------------------------------------------
    elif nav_page == "Learning":

        st.markdown(
            """
            <div>
                <div class="page-eyebrow">CURRICULUM & ADAPTIVE LEARNING</div>
                <div class="page-heading">Learning Resources & Knowledge Center</div>
                <div class="page-subheading">Curated educational materials paired with deterministic, ML-guided skill-gap recommendations.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 1. Deterministic ML-Guided Recommendations
        if get_recommended_resources:
            rec_data = get_recommended_resources(intern_id)
            gap_title = rec_data.get("gap_label", "Curated Recommendations")
            gap_reason = rec_data.get("reason", "")
            rec_list = rec_data.get("recommendations", [])

            st.markdown(
                f"""
                <div class="section-card" style="border-left: 4px solid #4F46E5; background: linear-gradient(135deg, #FFFFFF 0%, #F5F3FF 60%, #EEF2FF 100%);">
                    <div class="page-eyebrow" style="color: #4F46E5; margin-bottom: 2px;">ADAPTIVE LEARNING LOOP</div>
                    <div class="section-card-title">{gap_title}</div>
                    <div style="font-size: 13.5px; color: #475569; margin-bottom: 12px;">{gap_reason}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if rec_list:
                st.markdown("<div style='font-size: 14px; font-weight: 600; color: #0F172A; margin: 12px 0 8px 0;'>Recommended Focus Modules:</div>", unsafe_allow_html=True)
                rec_cols = st.columns(min(len(rec_list), 3))
                for idx, res in enumerate(rec_list[:3]):
                    with rec_cols[idx]:
                        r_title = res.get("title", "")
                        r_cat = res.get("category", "")
                        r_diff = res.get("difficulty", "Intermediate")
                        r_prov = res.get("provider", "Curated")
                        r_url = res.get("url", "")
                        r_type = res.get("resource_type", "documentation")

                        st.markdown(
                            f"""
                            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px; height: 160px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                                <div>
                                    <div style="display: flex; gap: 6px; margin-bottom: 6px;">
                                        <span class="badge-developing" style="font-size: 11px;">{r_cat}</span>
                                        <span class="badge-strong" style="font-size: 11px;">{r_diff}</span>
                                    </div>
                                    <div style="font-size: 13.5px; font-weight: 600; color: #0F172A; line-height: 1.3;">{r_title}</div>
                                    <div style="font-size: 11.5px; color: #64748B; margin-top: 4px;">Provider: {r_prov}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if r_url:
                            st.link_button(f"Open {r_type.title()}", r_url, use_container_width=True)

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        # 2. Comprehensive Curated Resource Library
        st.markdown(
            """
            <div class="section-card">
                <div class="section-card-title">Explore Full Curriculum Library</div>
                <div class="section-card-subtitle">Verified video tutorials, architectural guides, and official documentation across Java, OOP, and data pipelines.</div>
            """,
            unsafe_allow_html=True
        )

        filter_meta = get_available_filters() if get_available_filters else {}
        cats = filter_meta.get("categories", ["All Categories"])
        diffs = filter_meta.get("difficulties", ["All Levels"])
        types = filter_meta.get("resource_types", ["All Formats"])

        f1, f2, f3 = st.columns(3)
        with f1:
            sel_cat = st.selectbox("Category", cats, key="res_filter_cat")
        with f2:
            sel_diff = st.selectbox("Difficulty", diffs, key="res_filter_diff")
        with f3:
            sel_type = st.selectbox("Format", types, key="res_filter_type")

        all_curated = get_curated_resources(
            category=sel_cat,
            difficulty=sel_diff,
            resource_type=sel_type
        ) if get_curated_resources else []

        if not all_curated:
            st.info("No resources matching the selected filter criteria.")
        else:
            for item in all_curated:
                it_title = item.get("title", "")
                it_cat = item.get("category", "")
                it_diff = item.get("difficulty", "All Levels")
                it_type = item.get("resource_type", "documentation")
                it_prov = item.get("provider", "Authoritative")
                it_url = item.get("url", "")
                it_desc = item.get("description", "")

                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <div style="display: flex; gap: 6px; margin-bottom: 6px;">
                                    <span class="badge-developing">{it_cat}</span>
                                    <span class="badge-strong">{it_diff}</span>
                                    <span style="font-size: 11.5px; background: #F1F5F9; color: #475569; padding: 2px 8px; border-radius: 4px; font-weight: 500;">{it_type.upper()}</span>
                                </div>
                                <div style="font-size: 15px; font-weight: 600; color: #0F172A;">{it_title}</div>
                                <div style="font-size: 12.5px; color: #64748B; margin-top: 3px;">{it_desc}</div>
                                <div style="font-size: 11.5px; color: #94A3B8; margin-top: 4px;">Provider: {it_prov}</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if it_url:
                    st.link_button(f"Study on {it_prov}", it_url)

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PAGE: CHAT WITH MENTOR (PERSISTENT & PRIVATE)
    # --------------------------------------------------------
    elif nav_page == "Chat with Mentor":

        st.markdown(
            """
            <div>
                <div class="page-eyebrow">COMMUNICATION & MENTORSHIP</div>
                <div class="page-heading">Chat with Assigned Mentor</div>
                <div class="page-subheading">Ask technical questions, request code reviews, and receive 1-on-1 engineering guidance.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Assigned mentor ID is 1 (Demo Mentor)
        assigned_mentor_id = 1
        mentor_user_id = 2  # demo mentor user_id

        # Mark any unread replies from mentor as read
        if mark_messages_as_read:
            mark_messages_as_read(intern_id=intern_id, reader_user_id=user_id)

        # Load message thread
        chat_thread = []
        if get_conversation_history:
            try:
                chat_thread = get_conversation_history(
                    intern_id=intern_id,
                    viewer_user_id=user_id,
                    viewer_role="intern"
                )
            except Exception as e:
                st.error(f"Error loading chat messages: {e}")

        st.markdown(
            """
            <div class="section-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid #E2E8F0;">
                    <div>
                        <div style="font-size: 15px; font-weight: 700; color: #0F172A;">Mentor: Demo Mentor</div>
                        <div style="font-size: 12px; color: #10B981; font-weight: 500;">● Active • Senior Engineering Lead</div>
                    </div>
                    <div style="font-size: 12px; color: #64748B;">Private Channel</div>
                </div>
            """,
            unsafe_allow_html=True
        )

        if not chat_thread:
            st.info("No messages in this thread yet. Send a message below to start collaborating with your mentor.")
        else:
            for msg in chat_thread:
                is_mine = (msg.get("sender_user_id") == user_id)
                align_style = "margin-left: auto; background: #4F46E5; color: #FFFFFF;" if is_mine else "margin-right: auto; background: #F1F5F9; color: #0F172A;"
                sender_label = "You" if is_mine else "Mentor"
                time_label = msg.get("timestamp", "").replace("T", " ")

                st.markdown(
                    f"""
                    <div style="max-width: 75%; {align_style} border-radius: 12px; padding: 10px 14px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                        <div style="font-size: 11px; opacity: 0.85; margin-bottom: 3px; font-weight: 600;">{sender_label} • {time_label}</div>
                        <div style="font-size: 13.5px; line-height: 1.4;">{msg.get('message', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("</div>", unsafe_allow_html=True)

        # Send Message Form
        with st.form("intern_send_message_form", clear_on_submit=True):
            user_msg = st.text_input(
                "Type your message to mentor...",
                placeholder="e.g., I'm reviewing Java Collections and have a question on HashMap collisions.",
                key="intern_chat_message_input"
            )
            send_btn = st.form_submit_button("Send Message", use_container_width=True)

            if send_btn:
                if not user_msg or not user_msg.strip():
                    st.warning("Please enter a message before sending.")
                else:
                    try:
                        if send_chat_message:
                            send_chat_message(
                                sender_user_id=user_id,
                                receiver_user_id=mentor_user_id,
                                intern_id=intern_id,
                                mentor_id=assigned_mentor_id,
                                message=user_msg.strip()
                            )
                        # Ingest engagement observation into ML layer
                        if ingest_observation:
                            try:
                                ingest_observation(
                                    intern_id=intern_id,
                                    week=current_week or 10,
                                    source="application",
                                    event_type="mentor_chat_message",
                                    metric_name="engagement_rate",
                                    metric_value=None,
                                    payload={"action": "intern_sent_message"},
                                    label_status="IMMEDIATE_LABELED",
                                    training_eligible=1,
                                    outcome_value=1.0
                                )
                            except Exception:
                                pass
                        st.success("Message sent successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to send message: {e}")

    # --------------------------------------------------------
    # PAGE 6: PROGRESS
    # --------------------------------------------------------
    elif nav_page == "Progress":

        st.markdown(
            """
            <div>
                <div class="page-eyebrow">ANALYTICS</div>
                <div class="page-heading">Progress & Performance Analytics</div>
                <div class="page-subheading">Longitudinal performance trends and learning activity analytics over time.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if not perf_history_df.empty:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-card-title">Performance Score Trajectory</div>
                    <div class="section-card-subtitle">Weekly overall performance and skill score progress.</div>
                """,
                unsafe_allow_html=True
            )

            chart_data = perf_history_df.set_index("week")[["final_performance_score", "skill_score", "coding_score"]]
            chart_data.columns = ["Final Performance", "Skill Score", "Coding Score"]
            st.line_chart(chart_data)

            st.markdown("</div>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    """
                    <div class="section-card">
                        <div class="section-card-title">Assessment & Debugging Breakdown</div>
                    """,
                    unsafe_allow_html=True
                )
                bar_data = perf_history_df.set_index("week")[["mcq_score", "debugging_score", "code_review_score"]]
                bar_data.columns = ["MCQ", "Debugging", "Code Review"]
                st.bar_chart(bar_data)
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown(
                    """
                    <div class="section-card">
                        <div class="section-card-title">Error Frequency & Code Quality</div>
                    """,
                    unsafe_allow_html=True
                )
                err_data = perf_history_df.set_index("week")[["compile_error_rate", "runtime_error_rate"]]
                err_data.columns = ["Compile Error Rate", "Runtime Error Rate"]
                st.line_chart(err_data)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No historical progress data recorded yet.")

    # --------------------------------------------------------
    # PAGE 7: QUESTIONNAIRE — Self-Assessment & ML Feature Input
    # --------------------------------------------------------
    elif nav_page == "Questionnaire":

        st.markdown(
            """
            <div>
                <div class="page-eyebrow">SELF-ASSESSMENT</div>
                <div class="page-heading">Weekly Skill Questionnaire</div>
                <div class="page-subheading">Complete this questionnaire each week. Your answers are directly ingested into the ML prediction pipeline to generate your personalised skill forecast.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Load prior response for the current week (for pre-filling form)
        prior_response = None
        if get_questionnaire_response:
            try:
                prior_response = get_questionnaire_response(
                    intern_id=intern_id,
                    week=current_week or 10
                )
            except Exception:
                prior_response = None

        already_submitted = prior_response is not None
        if already_submitted:
            st.success(
                f"✅ You already submitted the Week {current_week or 10} questionnaire on {prior_response['submitted_at']}. "
                "Resubmitting will update your existing response."
            )

        if not QUESTIONNAIRE_QUESTIONS:
            st.warning("Questionnaire service unavailable. Please try again later.")
        else:
            with st.form("intern_questionnaire_form", clear_on_submit=False):
                st.markdown(
                    f"""
                    <div class="section-card" style="border-top: 4px solid #4F46E5;">
                        <div class="page-eyebrow" style="color: #4F46E5;">WEEK {current_week or 10} SELF-ASSESSMENT</div>
                        <div class="section-card-title">Learning Activity & Technical Confidence</div>
                        <div class="section-card-subtitle">Rate your activity and confidence levels for this week honestly. This data is used to train and update your ML skill-gap prediction model.</div>
                    """,
                    unsafe_allow_html=True
                )

                answers = {}

                for q in QUESTIONNAIRE_QUESTIONS:
                    key = q["key"]
                    label = q["label"]
                    help_txt = q.get("help", "")

                    # Pre-fill from prior response if available
                    if prior_response and key in prior_response:
                        prior_val = prior_response[key]
                    else:
                        prior_val = None

                    if q["type"] == "slider":
                        default_val = prior_val if prior_val is not None else q["default"]
                        # Ensure within range
                        default_val = max(q["min"], min(q["max"], float(default_val)))
                        if q["feature_type"] == "integer":
                            default_val = int(default_val)
                        val = st.slider(
                            label=label,
                            min_value=q["min"],
                            max_value=q["max"],
                            value=default_val,
                            step=q["step"],
                            help=help_txt,
                            key=f"q_{key}"
                        )
                        answers[key] = val

                    elif q["type"] == "selectbox":
                        options_list = q["options"]
                        default_idx = 0
                        if prior_val and prior_val in options_list:
                            default_idx = options_list.index(prior_val)
                        val = st.selectbox(
                            label=label,
                            options=options_list,
                            index=default_idx,
                            help=help_txt,
                            key=f"q_{key}"
                        )
                        answers[key] = val

                    elif q["type"] == "text_area":
                        default_text = prior_val if prior_val else ""
                        val = st.text_area(
                            label=label,
                            value=default_text,
                            max_chars=q.get("max_length", 500),
                            help=help_txt,
                            key=f"q_{key}"
                        )
                        answers[key] = val

                st.markdown("</div>", unsafe_allow_html=True)

                # Feature inspection preview
                if build_feature_inspection_table:
                    with st.expander("🔬 Preview: How your answers map to ML features"):
                        st.caption(
                            "The table below shows exactly which ML feature each answer maps to. "
                            "This is the same feature schema used by the production prediction model."
                        )
                        inspect_rows = build_feature_inspection_table(answers, current_week or 10)
                        st.dataframe(
                            pd.DataFrame(inspect_rows),
                            use_container_width=True,
                            hide_index=True
                        )

                st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
                submitted = st.form_submit_button(
                    "💾 Submit Questionnaire & Update Profile",
                    use_container_width=True,
                    type="primary"
                )

            if submitted:
                if save_questionnaire_response:
                    try:
                        resp_id, was_update = save_questionnaire_response(
                            intern_id=intern_id,
                            week=current_week or 10,
                            answers=answers,
                            allow_update=True
                        )
                        action_verb = "updated" if was_update else "submitted"
                        st.success(
                            f"✅ Questionnaire {action_verb} successfully! "
                            f"Response ID: QR-{resp_id} | Week {current_week or 10} | "
                            f"Your ML prediction will reflect this new data immediately."
                        )

                        # Show updated forecast after submission
                        if get_intern_ml_forecast:
                            forecast = get_intern_ml_forecast(intern_id)
                            if forecast.get("status") == "success":
                                st.markdown(
                                    f"""
                                    <div class="rec-banner">
                                        <strong>Updated ML Forecast:</strong>
                                        Predicted Skill Gap for Week {forecast['target_week']}: <strong>{forecast['predicted_skill_gap']}</strong>
                                        &nbsp;|&nbsp; Confidence: <strong>{forecast['confidence_percent']:.1f}%</strong>
                                        &nbsp;|&nbsp; Projected Performance: <strong>{forecast['predicted_performance_score']:.1f}/100</strong>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                st.info(f"📖 {forecast['recommendation']}")

                        # Show all submitted questionnaire history
                        if get_all_questionnaire_responses:
                            all_responses = get_all_questionnaire_responses(intern_id)
                            if len(all_responses) > 1:
                                st.markdown(
                                    """
                                    <div class="section-card" style="margin-top:1rem;">
                                        <div class="section-card-title">Your Questionnaire History</div>
                                        <div class="section-card-subtitle">All weekly self-assessment submissions.</div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                hist_df = pd.DataFrame(all_responses)[
                                    ["week", "learning_hours", "tasks_completed",
                                     "debugging_confidence", "code_review_score",
                                     "technical_focus_skill", "submitted_at"]
                                ]
                                hist_df.columns = [
                                    "Week", "Learning Hrs", "Tasks Done",
                                    "Debug Confidence", "Code Quality",
                                    "Focus Skill", "Submitted At"
                                ]
                                st.dataframe(hist_df, use_container_width=True, hide_index=True)
                                st.markdown("</div>", unsafe_allow_html=True)

                    except QuestionnaireValidationError as qve:
                        st.error(f"❌ Validation Error: {qve}")
                    except Exception as e:
                        st.error(f"❌ Questionnaire submission failed: {e}")
                else:
                    st.warning("Questionnaire service is not available. Please check system configuration.")

    # --------------------------------------------------------
    # PAGE 8: PROFILE
    # --------------------------------------------------------
    elif nav_page == "Profile":

        st.markdown(
            """
            <div>
                <div class="page-eyebrow">ACCOUNT DETAILS</div>
                <div class="page-heading">My Profile</div>
                <div class="page-subheading">Internship enrollment information, cohort details, and academic progress summary.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        p1, p2 = st.columns(2, gap="medium")

        with p1:
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="section-card-title">Account Information</div>
                    <div style="font-size: 13.5px; color: #334155; line-height: 1.8; margin-top: 12px;">
                        <div><strong>Full Name:</strong> {name}</div>
                        <div><strong>Email Address:</strong> {email}</div>
                        <div><strong>Intern ID:</strong> {intern_id}</div>
                        <div><strong>Role:</strong> {intern_role or 'Frontend Developer'}</div>
                        <div><strong>Experience Level:</strong> {experience_level or 'Intermediate'}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with p2:
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="section-card-title">Academic & Progress Overview</div>
                    <div style="font-size: 13.5px; color: #334155; line-height: 1.8; margin-top: 12px;">
                        <div><strong>Current Week:</strong> Week {current_week or 10}</div>
                        <div><strong>Overall Progress:</strong> {overall_progress:.1f}%</div>
                        <div><strong>Average Assessment Score:</strong> {average_score if average_score is not None else 'N/A'}</div>
                        <div><strong>Current Skill Score:</strong> {skill_score if skill_score is not None else 'N/A'}</div>
                        <div><strong>Current Skill Status:</strong> {skill_gap or 'Good'}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# MENTOR PORTAL
# ============================================================

elif role == "mentor":

    cohort_df = get_cohort_summary_df()
    eval_df = get_all_mentor_evaluations()
    cohort_perf_df = get_cohort_performance_df()

    total_interns = len(cohort_df)
    avg_progress = cohort_df["progress"].mean() if not cohort_df.empty and "progress" in cohort_df else 0
    avg_perf = cohort_df["final_performance_score"].mean() if not cohort_df.empty and "final_performance_score" in cohort_df else 0

    needs_attention_df = cohort_df[cohort_df["skill_gap"].isin(["Needs Improvement", "Developing"])] if not cohort_df.empty and "skill_gap" in cohort_df else pd.DataFrame()

    retained_count = len(eval_df[eval_df["decision"] == "FULL_TIME"]) if not eval_df.empty else 0
    not_retained_count = len(eval_df[eval_df["decision"] == "DO_NOT_RETAIN"]) if not eval_df.empty else 0

    # Orange Theme Injection
    st.markdown(get_orange_theme_css(), unsafe_allow_html=True)
    # --------------------------------------------------------
    # MENTOR PAGE 1: DASHBOARD
    # --------------------------------------------------------
    if nav_page == "Dashboard":

        st.markdown(
            """
            <div class="hero-banner">
                <div class="page-eyebrow">COHORT OVERVIEW</div>
                <div class="page-heading">Mentor Dashboard</div>
                <div class="page-subheading">Monitor intern progress, performance, and skill development across your cohort.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 4 Overview Metrics
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-label">Total Interns</div>
                    <div class="metric-card-value">{total_interns}</div>
                    <div class="metric-card-caption">Active cohort enrollment</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-label">Average Progress</div>
                    <div class="metric-card-value">{avg_progress:.1f}%</div>
                    <div class="metric-card-caption">Curriculum completion rate</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-label">Average Performance</div>
                    <div class="metric-card-value">{avg_perf:.1f}</div>
                    <div class="metric-card-caption">Out of 100 overall score</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-label">Needs Attention</div>
                    <div class="metric-card-value">{len(needs_attention_df)}</div>
                    <div class="metric-card-caption">Developing or Needs Improvement</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        # Live System Health & Data Panel
        if get_observation_summary and get_active_production_model:
            obs_sum = get_observation_summary()
            prod_m = get_active_production_model() or {}
            drift_info = assess_system_data_drift() if assess_system_data_drift else {"drift_status": "LOW"}
            is_retrain_ready, new_samples, retrain_reason = (
                check_retraining_eligibility() if check_retraining_eligibility else (False, 0, "")
            )
            val_m = prod_m.get("validation_metrics", {})
            cls_m = val_m.get("classification", {})
            reg_m = val_m.get("regression", {})

            online_info = get_online_learning_status() if get_online_learning_status else {}

            st.markdown(
                """
                <div class="section-card" style="border-top: 3px solid #2563EB;">
                    <div class="page-eyebrow" style="color: #2563EB; margin-bottom: 2px;">SYSTEM HEALTH & LEARNING STATUS</div>
                    <div class="section-card-title">Dual Event & Batch Learning System</div>
                    <div class="section-card-subtitle">Immediate events update the adaptive learner instantaneously. Forecasting targets remain pending until ground-truth outcomes are observed.</div>
                """,
                unsafe_allow_html=True
            )
            ml_c1, ml_c2, ml_c3, ml_c4, ml_c5, ml_c6 = st.columns(6)
            with ml_c1:
                st.metric("Total Observations", f"{obs_sum.get('total_active_observations', 5000):,}")
            with ml_c2:
                st.metric("Live App Events", f"{obs_sum.get('application_generated_count', 0):,}")
            with ml_c3:
                st.metric("Immediate Labeled", f"{obs_sum.get('immediate_supervised_count', 0):,}")
            with ml_c4:
                st.metric("Pending Labels", f"{obs_sum.get('pending_future_label_count', 0):,}")
            with ml_c5:
                st.metric("Online Updates", f"{online_info.get('total_online_updates', 0):,}")
            with ml_c6:
                st.metric("Data Drift Status", drift_info.get("drift_status", "LOW"))

            st.markdown(
                f"""
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 16px; margin: 12px 0 8px 0; font-size: 13px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <div><strong>Batch Production Model:</strong> {prod_m.get('model_type', 'Balanced Logistic Regression + Ridge')} (<code>{prod_m.get('model_version', 'v1.0-prod')}</code>)</div>
                        <div><strong>Quality Metrics:</strong> Macro F1: <code>{cls_m.get('macro_f1', 0.73):.2f}</code> | NI Recall: <code>{cls_m.get('needs_improvement_recall', 0.88)*100:.1f}%</code> | RMSE: <code>{reg_m.get('rmse', 7.56):.2f}</code></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div><strong>Online Incremental Model:</strong> SGDRegressor (partial_fit, Adaptive LR) (<code>{online_info.get('online_model_version', 'online_sgd_v0')}</code>)</div>
                        <div><strong>Online Updates:</strong> <code>{online_info.get('total_online_updates', 0)}</code> steps | Avg Step Loss: <code>{online_info.get('average_step_loss', 0.0):.4f}</code> | Last Update: <code>{online_info.get('last_online_update', 'None')}</code></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                f"**Data Provenance:** Historical Synthetic: **{obs_sum.get('synthetic_baseline_count', 5000):,}** | "
                f"Live Application: **{obs_sum.get('application_generated_count', 0)}** | "
                f"Training-Eligible Rows: **{obs_sum.get('training_eligible_samples', 4500):,}** | "
                f"Retraining Gate: **{retrain_reason}**"
            )
            st.info(
                "ℹ️ **Dual Learning Architecture:** Immediate events (quizzes, coding evaluations) update the online incremental learner immediately. "
                "Next-week forecasting samples are created immediately but hold pending labels until ground-truth outcomes are observed. "
                "Periodic batch retraining runs on validated datasets once the 50 new labeled sample threshold is reached."
            )
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-card" style="background: linear-gradient(135deg, #FFFFFF 0%, #FFFDF5 60%, #FFFBEB 100%); border-left: 4px solid #F59E0B;">
                <div class="page-eyebrow" style="color: #B45309; margin-bottom: 2px;">EARLY INTERVENTION</div>
                <div class="section-card-title">Needs Attention</div>
                <div class="section-card-subtitle">Interns predicted or evaluated to require targeted mentorship and learning support.</div>
            """,
            unsafe_allow_html=True
        )

        if not needs_attention_df.empty:
            for _, row in needs_attention_df.head(5).iterrows():
                i_id = int(row["intern_id"])
                i_name = row["name"]
                i_role = row.get("role", "Frontend")
                i_gap = row.get("skill_gap", "Developing")
                i_score = float(row.get("skill_score", 0))
                i_action = row.get("resource_action", "Provide 1-on-1 coaching")
                badge_cls = "badge-needs-improvement" if i_gap == "Needs Improvement" else "badge-developing"

                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #F1F5F9;">
                        <div>
                            <div style="font-weight: 600; font-size: 14px; color: #0F172A;">{i_name} <span style="font-size: 12px; color: #64748B; font-weight: normal;">(ID: {i_id})</span></div>
                            <div style="font-size: 12.5px; color: #64748B;">Score: {i_score:.1f} • Action: {i_action}</div>
                        </div>
                        <div class="{badge_cls}">{i_gap}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.caption("No interns currently require early intervention.")

        st.markdown("</div>", unsafe_allow_html=True)

        # Two-Column Summary Layout
        c_left, c_right = st.columns([1.1, 1], gap="medium")

        with c_left:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-card-title">Skill Gap Distribution</div>
                    <div class="section-card-subtitle">Cohort-wide status breakdown.</div>
                """,
                unsafe_allow_html=True
            )
            if not cohort_df.empty and "skill_gap" in cohort_df:
                gap_counts = cohort_df["skill_gap"].value_counts()
                st.bar_chart(gap_counts)
            else:
                st.caption("No skill gap data recorded.")
            st.markdown("</div>", unsafe_allow_html=True)

        with c_right:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-card-title">Evaluation & Retention Status</div>
                    <div class="section-card-subtitle">Summary of completed mentor decisions.</div>
                """,
                unsafe_allow_html=True
            )
            e1, e2 = st.columns(2)
            with e1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-card-label">Full-Time Retention</div>
                        <div class="metric-card-value" style="color: #047857;">{retained_count}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with e2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-card-label">Do Not Retain</div>
                        <div class="metric-card-value" style="color: #B91C1C;">{not_retained_count}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown(
                f"""
                <div style="font-size: 12.5px; color: #64748B; margin-top: 8px;">
                    Total evaluations completed: <strong>{len(eval_df)}</strong> of {total_interns} interns.
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # MENTOR PAGE: PROJECT DELIVERABLES REVIEW HUB
    # --------------------------------------------------------
    elif nav_page == "Deliverables":
        render_mentor_review_hub(mentor_id)

    # --------------------------------------------------------
    # MENTOR PAGE: END-TO-END DEMO (10-STAGE INTERN LIFECYCLE)
    # --------------------------------------------------------
    elif nav_page == "Internship Lifecycle Demo":
        if render_e2e_demo_page:
            render_e2e_demo_page(current_user, get_connection)
        else:
            st.error("Internship Lifecycle Demo module is currently unavailable.")

    # --------------------------------------------------------
    # MENTOR PAGE: ASSESSMENT RECORDS (MCQ & QUIZ HISTORY)
    # --------------------------------------------------------
    elif nav_page == "Assessment Records":

        st.markdown(
            """
            <div>
                <div class="page-eyebrow">QUIZ &amp; MCQ RECORDS</div>
                <div class="page-heading">Assessment Records</div>
                <div class="page-subheading">Review all intern quiz attempts, scores, and performance trends across assessment modules.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        if get_assessment_summary_stats:
            try:
                summary = get_assessment_summary_stats()
                if summary:
                    s1, s2, s3 = st.columns(3)
                    with s1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-card-label">Total Submissions</div>
                                <div class="metric-card-value">{summary.get('total_submissions', 0)}</div>
                                <div class="metric-card-caption">All quiz attempts</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with s2:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-card-label">Average Score</div>
                                <div class="metric-card-value">{summary.get('average_score', 0):.1f}%</div>
                                <div class="metric-card-caption">Cohort-wide assessment average</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with s3:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-card-label">Interns Assessed</div>
                                <div class="metric-card-value">{summary.get('assessed_intern_count', 0)}</div>
                                <div class="metric-card-caption">Interns with at least one attempt</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            except Exception:
                pass

        # Cohort MCQ attempts table
        conn_ar = get_connection()
        try:
            ar_df = pd.read_sql_query(
                """
                SELECT
                    ma.intern_id,
                    i.name AS intern_name,
                    ma.quiz_id,
                    ma.score,
                    ma.attempted_at
                FROM mcq_attempts ma
                JOIN interns i ON ma.intern_id = i.intern_id
                ORDER BY ma.attempted_at DESC
                """,
                conn_ar
            )
        except Exception:
            ar_df = pd.DataFrame()
        finally:
            conn_ar.close()

        if not ar_df.empty:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-card-title">All Assessment Attempts</div>
                    <div class="section-card-subtitle">Complete log of intern quiz scores across all modules.</div>
                """,
                unsafe_allow_html=True
            )
            ar_df.columns = ["Intern ID", "Intern Name", "Quiz ID", "Score (%)", "Attempted At"]
            st.dataframe(ar_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No assessment records found yet.")

    # --------------------------------------------------------
    # MENTOR PAGE: MESSAGES (INTERN-TO-MENTOR CHAT)
    # --------------------------------------------------------
    elif nav_page == "Messages":

        st.markdown(
            """
            <div>
                <div class="page-eyebrow">COMMUNICATION & COACHING</div>
                <div class="page-heading">Intern Mentorship Messages</div>
                <div class="page-subheading">Communicate directly and privately with your assigned engineering interns.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        unread_map = get_mentor_unread_counts(user_id) if get_mentor_unread_counts else {}

        # Fetch intern list from cohort
        all_intern_list = get_all_interns() if get_all_interns else []
        intern_chat_options = {}
        for itn in all_intern_list:
            itn_id = itn[0]
            itn_name = itn[1]
            unread = unread_map.get(itn_id, 0)
            badge = f"🔴 ({unread} unread)" if unread > 0 else "🟢"
            intern_chat_options[itn_id] = f"{itn_name} (ID: {itn_id}) {badge}"

        if not intern_chat_options:
            st.info("No interns found in the cohort.")
        else:
            col_sel, col_chat = st.columns([1, 2], gap="medium")

            with col_sel:
                st.markdown(
                    """
                    <div class="section-card">
                        <div class="section-card-title">Select Conversation</div>
                    """,
                    unsafe_allow_html=True
                )
                selected_chat_intern_id = st.selectbox(
                    "Assigned Interns",
                    options=list(intern_chat_options.keys()),
                    format_func=lambda x: intern_chat_options[x],
                    key="mentor_chat_intern_select"
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with col_chat:
                if selected_chat_intern_id:
                    # Mark messages from this intern as read
                    if mark_messages_as_read:
                        mark_messages_as_read(
                            intern_id=selected_chat_intern_id,
                            reader_user_id=user_id
                        )

                    # Find intern's user_id
                    conn_u = get_connection()
                    u_row = conn_u.execute("SELECT user_id, name FROM interns JOIN users ON interns.intern_id = users.intern_id WHERE interns.intern_id = ?", (selected_chat_intern_id,)).fetchone()
                    conn_u.close()
                    target_intern_user_id = u_row[0] if u_row else 1
                    target_intern_name = u_row[1] if u_row else f"Intern #{selected_chat_intern_id}"

                    st.markdown(
                        f"""
                        <div class="section-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 10px; border-bottom: 1px solid #E2E8F0; margin-bottom: 12px;">
                                <div>
                                    <div style="font-size: 15px; font-weight: 700; color: #0F172A;">{target_intern_name}</div>
                                    <div style="font-size: 12px; color: #64748B;">Intern ID: {selected_chat_intern_id} • Private Mentorship Channel</div>
                                </div>
                            </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Message History
                    thread = []
                    if get_conversation_history:
                        try:
                            thread = get_conversation_history(
                                intern_id=selected_chat_intern_id,
                                viewer_user_id=user_id,
                                viewer_role="mentor"
                            )
                        except Exception as e:
                            st.error(f"Error loading conversation: {e}")

                    if not thread:
                        st.caption("No message history with this intern yet. Send an introductory note below.")
                    else:
                        for m in thread:
                            is_mentor = (m.get("sender_user_id") == user_id)
                            m_align = "margin-left: auto; background: #2563EB; color: #FFFFFF;" if is_mentor else "margin-right: auto; background: #F1F5F9; color: #0F172A;"
                            m_sender = "You (Mentor)" if is_mentor else target_intern_name
                            m_time = m.get("timestamp", "").replace("T", " ")

                            st.markdown(
                                f"""
                                <div style="max-width: 75%; {m_align} border-radius: 12px; padding: 10px 14px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                                    <div style="font-size: 11px; opacity: 0.85; margin-bottom: 3px; font-weight: 600;">{m_sender} • {m_time}</div>
                                    <div style="font-size: 13.5px; line-height: 1.4;">{m.get('message', '')}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    st.markdown("</div>", unsafe_allow_html=True)

                    # Mentor Reply Form with dynamic intern-specific key & explicit error handling
                    with st.form(f"mentor_reply_form_{selected_chat_intern_id}", clear_on_submit=True):
                        reply_text = st.text_input(
                            "Reply to intern...",
                            placeholder=f"Provide feedback or answer {target_intern_name}'s questions...",
                            key=f"mentor_reply_input_{selected_chat_intern_id}"
                        )
                        send_reply_btn = st.form_submit_button("Send Reply", use_container_width=True)

                        if send_reply_btn:
                            if not reply_text or not reply_text.strip():
                                st.warning("Please enter a message before sending.")
                            else:
                                try:
                                    if send_chat_message:
                                        send_chat_message(
                                            sender_user_id=user_id,
                                            receiver_user_id=target_intern_user_id,
                                            intern_id=selected_chat_intern_id,
                                            mentor_id=mentor_id or 1,
                                            message=reply_text.strip()
                                        )
                                        st.success("Reply sent successfully!")
                                        st.rerun()
                                    else:
                                        st.error("Chat service is unavailable.")
                                except Exception as e:
                                    st.error(f"Failed to send message: {e}")

    # --------------------------------------------------------
    # MENTOR PAGE 2: INTERNS (DIRECTORY & EVALUATIONS)
    # --------------------------------------------------------
    elif nav_page == "Interns":

        st.markdown(
            """
            <div>
                <div class="page-eyebrow">DIRECTORY & EVALUATIONS</div>
                <div class="page-heading">Intern Directory</div>
                <div class="page-subheading">Search, filter, inspect individual learning journeys, and submit mentor reviews.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if cohort_df.empty:
            st.info("No interns found.")
        else:
            # Filter & Search Controls
            f_col1, f_col2, f_col3 = st.columns([1.5, 1, 1])

            with f_col1:
                search_query = st.text_input("Search Intern (Name or ID)", placeholder="Type name or ID...")

            with f_col2:
                skill_options = ["All Skills"] + sorted(list(cohort_df["role"].dropna().unique()))
                selected_skill = st.selectbox("Filter Skill / Role", skill_options)

            with f_col3:
                status_options = ["All Statuses", "Strong", "Developing", "Needs Improvement"]
                selected_status = st.selectbox("Filter Status", status_options)

            # Apply Filters
            filtered_df = cohort_df.copy()

            if search_query.strip():
                q = search_query.strip().lower()
                filtered_df = filtered_df[
                    filtered_df["name"].str.lower().str.contains(q, na=False) |
                    filtered_df["intern_id"].astype(str).str.contains(q, na=False)
                ]

            if selected_skill != "All Skills":
                filtered_df = filtered_df[filtered_df["role"] == selected_skill]

            if selected_status != "All Statuses":
                filtered_df = filtered_df[filtered_df["skill_gap"] == selected_status]

            if filtered_df.empty:
                st.info("No interns match this filter.")
            else:
                intern_options = {
                    f"{row['name']} (ID: {row['intern_id']}) — {row.get('skill_gap', 'Good')}": int(row["intern_id"])
                    for _, row in filtered_df.iterrows()
                }

                selected_label = st.selectbox("Select Candidate to Inspect", list(intern_options.keys()))
                selected_intern_id = intern_options[selected_label]

                selected_intern = get_intern(selected_intern_id)
                selected_submissions = get_intern_submissions(selected_intern_id)
                selected_mcqs = get_mcq_attempts(selected_intern_id)
                existing_evaluation = get_intern_evaluation(selected_intern_id)

                if selected_intern:
                    (
                        sid, sname, semail, srole, sexperience, scurrent_week,
                        saverage_score, sskill_score, sfinal_perf, sskill_gap,
                        sresource_action, sprogress
                    ) = selected_intern

                    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

                    # Intern Overview Card & 4 Metrics
                    st.markdown(
                        f"""
                        <div class="section-card">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                                <div>
                                    <div style="font-size: 20px; font-weight: 700; color: #0F172A;">{sname}</div>
                                    <div style="font-size: 13px; color: #64748B;">{semail} • ID: {sid} • {srole or 'Frontend Developer'} • {sexperience or 'Intermediate'}</div>
                                </div>
                                <div class="badge-{'strong' if sskill_gap == 'Strong' else ('developing' if sskill_gap == 'Developing' else 'needs-improvement')}">{sskill_gap}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    q1, q2, q3, q4 = st.columns(4)
                    with q1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-card-label">Progress</div>
                                <div class="metric-card-value">{sprogress or 0:.1f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with q2:
                        avg_disp = f"{saverage_score:.1f}" if saverage_score is not None else "N/A"
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-card-label">Average Score</div>
                                <div class="metric-card-value">{avg_disp}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with q3:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-card-label">Coding Submissions</div>
                                <div class="metric-card-value">{len(selected_submissions)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with q4:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-card-label">MCQ Attempts</div>
                                <div class="metric-card-value">{len(selected_mcqs)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # Dynamic ML Skill Outlook for Selected Intern
                    st.markdown(
                        """
                        <div class="outlook-card">
                            <div class="page-eyebrow" style="margin-bottom: 2px;">PREDICTIVE INSIGHT</div>
                            <div class="section-card-title">Next-Week Skill Outlook</div>
                            <div class="section-card-subtitle">Future-oriented forecast based on intern learning trajectory.</div>
                        """,
                        unsafe_allow_html=True
                    )

                    if get_intern_ml_forecast:
                        m_forecast = get_intern_ml_forecast(selected_intern_id)
                        if m_forecast.get("status") == "success":
                            m_gap = m_forecast["predicted_skill_gap"]
                            m_conf = m_forecast["confidence_percent"]
                            m_perf = m_forecast["predicted_performance_score"]
                            m_target_w = m_forecast["target_week"]
                            m_rec = m_forecast["recommendation"]
                            m_top_factors = m_forecast.get("top_factors", [])

                            fa, fb, fc = st.columns(3)
                            with fa:
                                badge_cls = "badge-strong" if m_gap == "Strong" else ("badge-developing" if m_gap == "Developing" else "badge-needs-improvement")
                                st.markdown(
                                    f"""
                                    <div style="font-size: 12px; font-weight: 600; color: #64748B; margin-bottom: 6px;">PREDICTED SKILL GAP (WEEK {m_target_w})</div>
                                    <div class="{badge_cls}" style="font-size: 14px; padding: 4px 12px;">{m_gap}</div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            with fb:
                                st.metric(label="Forecast Confidence", value=f"{m_conf:.1f}%")
                            with fc:
                                st.metric(label="Projected Next-Week Performance", value=f"{m_perf:.1f} / 100")

                            st.markdown(
                                f"""
                                <div class="rec-banner">
                                    <strong>Recommendation:</strong> {m_rec}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            with st.expander("Why this forecast?"):
                                st.caption("Factors that were most influential in the trained model:")
                                for factor in m_top_factors:
                                    st.write(f"• {factor}")
                                st.caption("*Note: Influential factors reflect predictive associations in historical training data, not direct causation.*")
                        else:
                            st.caption(f"ℹ️ *{m_forecast.get('message', 'No forecast available yet.')}*")

                    st.markdown("</div>", unsafe_allow_html=True)

                    # Activity Deep-Dive: Coding & MCQs
                    sub_col, mcq_col = st.columns(2, gap="medium")

                    with sub_col:
                        st.markdown(
                            """
                            <div class="section-card">
                                <div class="section-card-title">Project Deliverables</div>
                                <div class="section-card-subtitle">Recent PDF milestone submissions and evaluation states.</div>
                            """,
                            unsafe_allow_html=True
                        )
                        cand_delivs = get_intern_deliverable_history(selected_intern_id) if get_intern_deliverable_history else []
                        if cand_delivs:
                            for sub in cand_delivs[:4]:
                                st.markdown(
                                    f"""
                                    <div style="padding: 8px 0; border-bottom: 1px solid #F1F5F9;">
                                        <div style="font-weight: 600; font-size: 13.5px; color: #0F172A;">{sub['title']}</div>
                                        <div style="font-size: 12px; color: #64748B;">v{sub['version_number']} • Status: {sub['review_status']} • {sub['submitted_at']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        else:
                            st.caption("No project deliverables recorded.")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with mcq_col:
                        st.markdown(
                            """
                            <div class="section-card">
                                <div class="section-card-title">Assessment History</div>
                                <div class="section-card-subtitle">Recent quiz performance and topics.</div>
                            """,
                            unsafe_allow_html=True
                        )
                        if selected_mcqs:
                            for attempt in selected_mcqs[:4]:
                                st.markdown(
                                    f"""
                                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #F1F5F9;">
                                        <div>
                                            <div style="font-weight: 600; font-size: 13.5px; color: #0F172A;">{attempt[3] or 'Technical Quiz'}</div>
                                            <div style="font-size: 12px; color: #64748B;">{attempt[4]}</div>
                                        </div>
                                        <span class="badge-developing">{attempt[1]}%</span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        else:
                            st.caption("No assessment attempts recorded.")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Mentor Evaluation Form
                    st.markdown(
                        """
                        <div class="section-card">
                            <div class="section-card-title">Mentor Evaluation & Retention Decision</div>
                            <div class="section-card-subtitle">Submit official mentor scoring and career placement recommendation.</div>
                        """,
                        unsafe_allow_html=True
                    )

                    if existing_evaluation:
                        with st.expander("View Previous Evaluation Record"):
                            st.markdown(
                                f"""
                                <div>
                                    <div><strong>Technical Score:</strong> {existing_evaluation[0]} / 100</div>
                                    <div><strong>Communication:</strong> {existing_evaluation[1]} / 100</div>
                                    <div><strong>Discipline:</strong> {existing_evaluation[2]} / 100</div>
                                    <div><strong>Overall Score:</strong> {existing_evaluation[3]}%</div>
                                    <div><strong>Decision:</strong> {existing_evaluation[4]}</div>
                                    <div><strong>Remarks:</strong> {existing_evaluation[5]}</div>
                                    <div style="font-size: 12px; color: #64748B; margin-top: 4px;">Evaluated at: {existing_evaluation[6]}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    with st.form(f"eval_form_{selected_intern_id}"):
                        tech_score = st.slider("Technical Competency", 0, 100, 75)
                        comm_score = st.slider("Communication & Collaboration", 0, 100, 75)
                        disc_score = st.slider("Discipline & Consistency", 0, 100, 75)

                        calc_overall = round((tech_score + comm_score + disc_score) / 3, 2)
                        st.markdown(f"**Calculated Evaluation Score:** `{calc_overall:.1f}%`")

                        decision_choice = st.selectbox("Retention Decision", ["FULL_TIME", "DO_NOT_RETAIN"])
                        remarks_input = st.text_area("Mentor Remarks & Guidance Notes", placeholder="Add formal remarks...")

                        submit_eval = st.form_submit_button("Save Evaluation Record", use_container_width=True)

                        if submit_eval:
                            save_evaluation(
                                selected_intern_id,
                                mentor_id,
                                tech_score,
                                comm_score,
                                disc_score,
                                calc_overall,
                                decision_choice,
                                remarks_input
                            )
                            st.success("Evaluation record saved successfully!")
                            st.rerun()

                    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # MENTOR PAGE 3: PERFORMANCE (COHORT ANALYTICS)
    # --------------------------------------------------------
    elif nav_page == "Performance Analytics":

        st.markdown(
            """
            <div>
                <div class="page-eyebrow">COHORT ANALYTICS</div>
                <div class="page-heading">Performance Analytics</div>
                <div class="page-subheading">Longitudinal performance trajectories, score distributions, and quality metrics across the cohort.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if not cohort_perf_df.empty:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-card-title">Weekly Performance Score Trajectory</div>
                    <div class="section-card-subtitle">Cohort-average overall performance, skill scores, and coding challenge scores across curriculum weeks.</div>
                """,
                unsafe_allow_html=True
            )

            weekly_perf = cohort_perf_df.groupby("week")[["avg_final_score", "avg_skill_score", "avg_coding_score"]].mean()
            weekly_perf.columns = ["Final Performance", "Skill Score", "Coding Score"]
            st.line_chart(weekly_perf)

            st.markdown("</div>", unsafe_allow_html=True)

            p_col1, p_col2 = st.columns(2, gap="medium")

            with p_col1:
                st.markdown(
                    """
                    <div class="section-card">
                        <div class="section-card-title">Assessment & Debugging Progression</div>
                    """,
                    unsafe_allow_html=True
                )
                weekly_assess = cohort_perf_df.groupby("week")[["avg_mcq_score", "avg_debugging_score"]].mean()
                weekly_assess.columns = ["MCQ Score", "Debugging Score"]
                st.bar_chart(weekly_assess)
                st.markdown("</div>", unsafe_allow_html=True)

            with p_col2:
                st.markdown(
                    """
                    <div class="section-card">
                        <div class="section-card-title">Error Frequency & Code Reliability</div>
                    """,
                    unsafe_allow_html=True
                )
                weekly_errs = cohort_perf_df.groupby("week")[["avg_compile_err", "avg_runtime_err"]].mean()
                weekly_errs.columns = ["Compile Error Rate", "Runtime Error Rate"]
                st.line_chart(weekly_errs)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No cohort performance data available.")

    # --------------------------------------------------------
    # MENTOR PAGE 4: SKILL GAPS (CURRICULUM ANALYSIS)
    # --------------------------------------------------------
    elif nav_page == "Skill Gaps":

        st.markdown(
            """
            <div>
                <div class="page-eyebrow">CURRICULUM ANALYSIS</div>
                <div class="page-heading">Skill Gap Analysis</div>
                <div class="page-subheading">Identify technical bottlenecks across frontend domains and optimize curriculum delivery.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if not cohort_df.empty:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-card-title">Technical Domain Status Matrix</div>
                    <div class="section-card-subtitle">Cohort distribution across technical skills and proficiency tiers.</div>
                """,
                unsafe_allow_html=True
            )

            skill_pivot = pd.crosstab(cohort_df["role"], cohort_df["skill_gap"])
            st.dataframe(skill_pivot, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                """
                <div class="section-card">
                    <div class="section-card-title">Curriculum Remediation Priority</div>
                    <div class="section-card-subtitle">Skills with highest proportion of developing or struggling interns.</div>
                """,
                unsafe_allow_html=True
            )

            action_counts = cohort_df["resource_action"].value_counts()
            st.bar_chart(action_counts)

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No skill gap records available.")

    # --------------------------------------------------------
    # MENTOR PAGE 5: REPORTS (EVALUATION & PLACEMENT)
    # --------------------------------------------------------
    # NOTE: Benchmark page is inserted AFTER Reports, before Profile
    elif nav_page == "Reports":

        st.markdown(
            """
            <div>
                <div class="page-eyebrow">PLACEMENT & RETENTION</div>
                <div class="page-heading">Evaluation Reports</div>
                <div class="page-subheading">Review mentor assessments, full-time retention decisions, and cohort summary.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-label">Full-Time Retention</div>
                    <div class="metric-card-value" style="color: #047857;">{retained_count}</div>
                    <div class="metric-card-caption">Recommended for full-time offer</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with r2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-label">Do Not Retain</div>
                    <div class="metric-card-value" style="color: #B91C1C;">{not_retained_count}</div>
                    <div class="metric-card-caption">Does not meet benchmark</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with r3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-label">Total Evaluated</div>
                    <div class="metric-card-value">{len(eval_df)}</div>
                    <div class="metric-card-caption">Completed mentor reviews</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="section-card">
                <div class="section-card-title">Completed Mentor Evaluations</div>
                <div class="section-card-subtitle">Log of formal intern assessments and remarks.</div>
            """,
            unsafe_allow_html=True
        )

        if not eval_df.empty:
            display_eval_df = eval_df[["intern_id", "intern_name", "technical_score", "communication_score", "discipline_score", "overall_score", "decision", "evaluated_at"]].copy()
            display_eval_df.columns = ["Intern ID", "Intern Name", "Technical", "Communication", "Discipline", "Overall %", "Decision", "Evaluated At"]
            st.dataframe(display_eval_df, use_container_width=True)
        else:
            st.caption("No mentor evaluation records recorded yet.")

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # MENTOR PAGE 6: BENCHMARK DASHBOARD
    # --------------------------------------------------------
    elif nav_page in ["Forecast Evaluation", "Benchmark"]:

        st.markdown(
            """
            <div>
                <div class="page-eyebrow">PERFORMANCE EVALUATION</div>
                <div class="page-heading">Forecast Evaluation Dashboard</div>
                <div class="page-subheading">Reproducible evaluation of production models on held-out test data, validation splits, baseline comparisons, live observations, and dataset integrity audit.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)

        if run_historical_benchmark is None:
            st.warning("Benchmark module is not available. Please check system configuration.")
        else:
            bench_tab, live_tab, audit_tab = st.tabs(["📊 Historical Benchmark", "🔴 Live Observations", "🗄️ Dataset Audit"])

            with bench_tab:
                st.markdown(
                    """
                    <div class="section-card" style="border-top: 4px solid #4F46E5;">
                        <div class="page-eyebrow" style="color: #4F46E5;">PRODUCTION MODEL EVALUATION</div>
                        <div class="section-card-title">Historical Held-Out Test Benchmark</div>
                        <div class="section-card-subtitle">Evaluation of classifier (Logistic Regression) and regressor (Ridge Regression) on the held-out test intern split. Train and test interns are strictly disjoint — no temporal or data leakage.</div>
                    """,
                    unsafe_allow_html=True
                )

                if st.button("▶️ Run Benchmark Now", key="run_benchmark_btn", type="primary"):
                    with st.spinner("Evaluating production models on held-out test data..."):
                        try:
                            hist_results = run_historical_benchmark()
                            if save_benchmark_results:
                                from src.ml.benchmark import BENCHMARK_RESULTS_PATH
                                save_benchmark_results(
                                    {"historical": hist_results, "run_by": username},
                                    BENCHMARK_RESULTS_PATH
                                )
                            st.session_state["benchmark_hist_results"] = hist_results
                        except Exception as bench_err:
                            st.error(f"Benchmark error: {bench_err}")

                bench_results = st.session_state.get("benchmark_hist_results")

                # Try loading previously saved results
                if bench_results is None and load_saved_benchmark_results:
                    try:
                        saved = load_saved_benchmark_results()
                        if saved and "historical" in saved:
                            bench_results = saved["historical"]
                            st.caption("*Showing previously saved benchmark results. Click 'Run Benchmark Now' to refresh.*")
                    except Exception:
                        pass

                if bench_results:
                    if "error" in bench_results:
                        st.error(f"Benchmark error: {bench_results['error']}")
                    else:
                        st.markdown(f"**Benchmark run at:** `{bench_results.get('benchmark_timestamp', 'N/A')}`")

                        # Test Split Metrics
                        test_cls = bench_results.get("test_classification", {})
                        test_reg = bench_results.get("test_regression", {})
                        base_cls = bench_results.get("baseline_classification", {})
                        base_reg = bench_results.get("baseline_regression", {})

                        st.markdown("### 🎯 Classification (Skill Gap Prediction) — Test Split")

                        bc1, bc2, bc3, bc4, bc5 = st.columns(5)
                        with bc1:
                            acc = test_cls.get("accuracy")
                            bacc = base_cls.get("accuracy")
                            delta = round(acc - bacc, 4) if acc is not None and bacc is not None else None
                            st.metric("Accuracy", f"{acc:.4f}" if acc else "N/A",
                                      delta=f"{delta:+.4f} vs baseline" if delta is not None else None)
                        with bc2:
                            mf1 = test_cls.get("macro_f1")
                            bmf1 = base_cls.get("macro_f1")
                            delta2 = round(mf1 - bmf1, 4) if mf1 is not None and bmf1 is not None else None
                            st.metric("Macro F1", f"{mf1:.4f}" if mf1 else "N/A",
                                      delta=f"{delta2:+.4f} vs baseline" if delta2 is not None else None)
                        with bc3:
                            wf1 = test_cls.get("weighted_f1")
                            st.metric("Weighted F1", f"{wf1:.4f}" if wf1 else "N/A")
                        with bc4:
                            bacc_score = test_cls.get("balanced_accuracy")
                            st.metric("Balanced Acc.", f"{bacc_score:.4f}" if bacc_score else "N/A")
                        with bc5:
                            ni_r = test_cls.get("ni_recall")
                            st.metric("NI Recall", f"{ni_r:.4f}" if ni_r else "N/A",
                                      help="Recall for 'Needs Improvement' class — critical for identifying at-risk interns")

                        if test_cls.get("labels") and test_cls.get("confusion_matrix"):
                            with st.expander("📊 Confusion Matrix & Per-Class Metrics"):
                                labels = test_cls["labels"]
                                cm = test_cls["confusion_matrix"]
                                cm_df = pd.DataFrame(cm, index=[f"True: {l}" for l in labels],
                                                      columns=[f"Pred: {l}" for l in labels])
                                st.dataframe(cm_df, use_container_width=True)
                                per_class_df = pd.DataFrame({
                                    "Class": labels,
                                    "Precision": test_cls.get("per_class_precision", []),
                                    "Recall": test_cls.get("per_class_recall", []),
                                    "F1": test_cls.get("per_class_f1", [])
                                })
                                st.dataframe(per_class_df, use_container_width=True, hide_index=True)

                        st.markdown("### 📈 Regression (Performance Score Forecast) — Test Split")

                        br1, br2, br3, br4 = st.columns(4)
                        with br1:
                            rmse = test_reg.get("rmse")
                            brmse = base_reg.get("rmse")
                            delta_rmse = round(rmse - brmse, 4) if rmse is not None and brmse is not None else None
                            st.metric("RMSE", f"{rmse:.4f}" if rmse else "N/A",
                                      delta=f"{delta_rmse:+.4f} vs baseline" if delta_rmse is not None else None,
                                      delta_color="inverse")
                        with br2:
                            mae = test_reg.get("mae")
                            st.metric("MAE", f"{mae:.4f}" if mae else "N/A")
                        with br3:
                            r2 = test_reg.get("r2")
                            br2_score = base_reg.get("r2")
                            delta_r2 = round(r2 - br2_score, 4) if r2 is not None and br2_score is not None else None
                            st.metric("R²", f"{r2:.4f}" if r2 else "N/A",
                                      delta=f"{delta_r2:+.4f} vs baseline" if delta_r2 is not None else None)
                        with br4:
                            n_samp = test_reg.get("n_samples")
                            st.metric("Test Samples", str(n_samp) if n_samp else "N/A")

                        # Reference metrics comparison
                        ref = bench_results.get("previously_reported_metrics", {})
                        if ref:
                            with st.expander("📌 Compare to Previously Reported Metrics"):
                                st.markdown(
                                    f"""
                                    | Metric | Previously Reported | Current Run |
                                    |---|---|---|
                                    | Test Cls Accuracy | `{ref.get('test_cls_accuracy', 'N/A')}` | `{test_cls.get('accuracy', 'N/A')}` |
                                    | Test Cls Macro F1 | `{ref.get('test_cls_macro_f1', 'N/A')}` | `{test_cls.get('macro_f1', 'N/A')}` |
                                    | Test Reg RMSE | `{ref.get('test_reg_rmse', 'N/A')}` | `{test_reg.get('rmse', 'N/A')}` |
                                    | Test Reg R² | `{ref.get('test_reg_r2', 'N/A')}` | `{test_reg.get('r2', 'N/A')}` |
                                    """
                                )
                                st.caption(f"Reference source: `{ref.get('source', 'N/A')}`")

                        # Leakage check
                        leak = bench_results.get("temporal_leakage_check", {})
                        if leak:
                            verified = leak.get("verified", False)
                            st.markdown(
                                f"""
                                <div style="background: {'#F0FDF4' if verified else '#FFF7ED'}; border: 1px solid {'#BBF7D0' if verified else '#FED7AA'}; border-radius: 8px; padding: 12px 16px; margin-top: 12px; font-size: 13px; color: {'#166534' if verified else '#92400E'}; ">
                                    {'✅' if verified else '⚠️'} <strong>Temporal Leakage Check:</strong> {leak.get('split_method', 'N/A')} | Train-Test Intern Overlap: {leak.get('train_test_overlap', 'N/A')} | Future Labels in Features: {leak.get('future_labels_in_features', 'N/A')}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                st.markdown("</div>", unsafe_allow_html=True)

            with live_tab:
                st.markdown(
                    """
                    <div class="section-card" style="border-top: 4px solid #059669;">
                        <div class="page-eyebrow" style="color: #059669;">LIVE OBSERVATION EVALUATION</div>
                        <div class="section-card-title">Production Model on Live-Labeled Observations</div>
                        <div class="section-card-subtitle">Evaluates predictions against observations where labels are available. Only uses observations with label_status LABELED or IMMEDIATE_LABELED and training_eligible = 1.</div>
                    """,
                    unsafe_allow_html=True
                )

                if st.button("▶️ Run Live Benchmark", key="run_live_bench_btn"):
                    with st.spinner("Evaluating on live labeled observations..."):
                        try:
                            live_res = run_live_data_benchmark()
                            st.session_state["benchmark_live_results"] = live_res
                        except Exception as lbe:
                            st.error(f"Live benchmark error: {lbe}")

                live_bench = st.session_state.get("benchmark_live_results")
                if live_bench:
                    lv1, lv2, lv3 = st.columns(3)
                    with lv1:
                        st.metric("Total Live Observations", live_bench.get("total_live_observations", 0))
                    with lv2:
                        st.metric("Labeled (Supervised)", live_bench.get("labeled_live_observations", 0))
                    with lv3:
                        st.metric("Pending Future Labels", live_bench.get("pending_future_label_observations", 0))

                    st.info(live_bench.get("evaluation_note", ""))

                    if live_bench.get("supervised_metrics"):
                        sm = live_bench["supervised_metrics"]
                        if sm.get("regression"):
                            reg = sm["regression"]
                            st.markdown("#### Live Regression Metrics")
                            ls1, ls2, ls3 = st.columns(3)
                            with ls1: st.metric("RMSE", f"{reg.get('rmse', 'N/A'):.4f}")
                            with ls2: st.metric("MAE", f"{reg.get('mae', 'N/A'):.4f}")
                            with ls3: st.metric("R²", f"{reg.get('r2', 'N/A'):.4f}")
                            st.caption(sm.get("caveat", ""))
                else:
                    st.caption("Click 'Run Live Benchmark' to evaluate on current live observation data.")

                st.markdown("</div>", unsafe_allow_html=True)

            with audit_tab:
                st.markdown(
                    """
                    <div class="section-card" style="border-top: 4px solid #7C3AED;">
                        <div class="page-eyebrow" style="color: #7C3AED;">DATA INTEGRITY</div>
                        <div class="section-card-title">Dataset Audit Report</div>
                        <div class="section-card-subtitle">Full audit of raw and processed datasets: row counts, feature counts, target distribution, missing values, duplicates, and train/val/test split statistics.</div>
                    """,
                    unsafe_allow_html=True
                )

                if st.button("▶️ Run Dataset Audit", key="run_audit_btn"):
                    with st.spinner("Auditing datasets..."):
                        try:
                            audit_res = get_dataset_audit()
                            st.session_state["benchmark_audit_results"] = audit_res
                        except Exception as ae:
                            st.error(f"Dataset audit error: {ae}")

                audit_data = st.session_state.get("benchmark_audit_results")
                if audit_data:
                    ra1, ra2 = st.columns(2)
                    with ra1:
                        raw_d = audit_data.get("raw_dataset", {})
                        if "error" not in raw_d:
                            st.markdown("#### Raw Dataset")
                            st.markdown(
                                f"""
                                | Property | Value |
                                |---|---|
                                | Rows | `{raw_d.get('rows', 'N/A')}` |
                                | Columns | `{raw_d.get('columns', 'N/A')}` |
                                | Unique Interns | `{raw_d.get('unique_interns', 'N/A')}` |
                                | Missing Values | `{raw_d.get('missing_values', 'N/A')}` |
                                | Duplicate Rows | `{raw_d.get('duplicates', 'N/A')}` |
                                """
                            )
                        else:
                            st.warning(f"Raw dataset: {raw_d['error']}")

                    with ra2:
                        proc_d = audit_data.get("processed_dataset", {})
                        if "error" not in proc_d:
                            st.markdown("#### Processed Dataset")
                            st.markdown(
                                f"""
                                | Property | Value |
                                |---|---|
                                | Rows | `{proc_d.get('rows', 'N/A')}` |
                                | Feature Columns | `{proc_d.get('feature_columns', 'N/A')}` |
                                | Unique Interns | `{proc_d.get('unique_interns', 'N/A')}` |
                                | Missing Values | `{proc_d.get('missing_values', 'N/A')}` |
                                | Duplicate Rows | `{proc_d.get('duplicates', 'N/A')}` |
                                """
                            )
                            if proc_d.get("target_distribution"):
                                st.markdown("**Target Distribution (target_skill_gap):**")
                                td = proc_d["target_distribution"]
                                td_df = pd.DataFrame.from_dict(td, orient="index", columns=["Count"])
                                td_df.index.name = "Skill Gap Class"
                                st.dataframe(td_df, use_container_width=True)
                        else:
                            st.warning(f"Processed dataset: {proc_d['error']}")

                    # Split statistics
                    st.markdown("#### Train / Validation / Test Splits")
                    split_rows = []
                    for sname in ["train", "validation", "test"]:
                        sd = audit_data.get(f"{sname}_split", {})
                        split_rows.append({
                            "Split": sname.capitalize(),
                            "Intern Count": sd.get("intern_count", "N/A") if "error" not in sd else f"Error: {sd['error']}",
                        })
                    st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)

                else:
                    st.caption("Click 'Run Dataset Audit' to inspect dataset integrity.")

                st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # MENTOR PAGE 7: PROFILE
    # --------------------------------------------------------
    elif nav_page == "Profile":

        st.markdown(
            """
            <div>
                <div class="page-eyebrow">ACCOUNT DETAILS</div>
                <div class="page-heading">Mentor Profile</div>
                <div class="page-subheading">Account information and supervisory cohort details.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-card-title">Account Information</div>
                <div style="font-size: 13.5px; color: #334155; line-height: 1.8; margin-top: 12px;">
                    <div><strong>Username:</strong> {username}</div>
                    <div><strong>Role:</strong> Mentor / Supervisor</div>
                    <div><strong>Supervised Cohort:</strong> Frontend Engineering Cohort (500 Interns)</div>
                    <div><strong>System Permissions:</strong> Evaluation Access, Predictive Skill Analytics, Placement Review</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SkillTrack — Internship Management & Evaluation Portal"
)

st.caption(
    "Code execution is configured for development/demo environments. "
    "Use container isolation for public deployment."
)