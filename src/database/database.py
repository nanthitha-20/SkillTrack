import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATABASE_DIR = BASE_DIR / "data" / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATABASE_DIR / "skilltrack.db"


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('intern', 'mentor')),
            intern_id INTEGER,
            mentor_id INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interns (
            intern_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            role TEXT,
            experience_level TEXT,
            current_week INTEGER DEFAULT 10,
            average_score REAL,
            skill_score REAL,
            final_performance_score REAL,
            skill_gap TEXT,
            resource_action TEXT,
            progress REAL DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS intern_performance (
            performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            intern_id INTEGER NOT NULL,
            week INTEGER NOT NULL,
            days_since_start INTEGER,
            skill TEXT,
            learning_hours REAL,
            videos_completed INTEGER,
            guides_completed INTEGER,
            tasks_completed INTEGER,
            engagement_rate REAL,
            coding_attempts INTEGER,
            execution_count INTEGER,
            average_time_minutes REAL,
            compile_error_rate REAL,
            runtime_error_rate REAL,
            test_cases_total INTEGER,
            test_cases_passed INTEGER,
            coding_score REAL,
            mcq_attempts INTEGER,
            mcq_score REAL,
            debugging_score REAL,
            code_review_score REAL,
            previous_skill_score REAL,
            skill_score REAL,
            score_improvement REAL,
            final_performance_score REAL,
            skill_gap TEXT,
            resource_action TEXT,
            FOREIGN KEY (intern_id) REFERENCES interns(intern_id),
            UNIQUE(intern_id, week)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mentors (
            mentor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS coding_tasks (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            max_score REAL DEFAULT 100
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
            intern_id INTEGER,
            task_id INTEGER,
            score REAL,
            status TEXT,
            submitted_at TEXT,
            FOREIGN KEY (intern_id) REFERENCES interns(intern_id),
            FOREIGN KEY (task_id) REFERENCES coding_tasks(task_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mcq_questions (
            question_id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            option_a TEXT,
            option_b TEXT,
            option_c TEXT,
            option_d TEXT,
            correct_option TEXT,
            topic TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mcq_attempts (
            attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            intern_id INTEGER,
            score REAL,
            total_questions INTEGER,
            topic TEXT,
            attempted_at TEXT,
            FOREIGN KEY (intern_id) REFERENCES interns(intern_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resources (
            resource_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            resource_type TEXT,
            url TEXT,
            topic TEXT,
            difficulty TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mentor_evaluations (
            evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            intern_id INTEGER,
            mentor_id INTEGER,
            technical_score REAL,
            communication_score REAL,
            discipline_score REAL,
            overall_score REAL,
            decision TEXT,
            remarks TEXT,
            evaluated_at TEXT,
            FOREIGN KEY (intern_id) REFERENCES interns(intern_id),
            FOREIGN KEY (mentor_id) REFERENCES mentors(mentor_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_observations (
            observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            intern_id INTEGER NOT NULL,
            week INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL CHECK(source IN ('synthetic', 'application', 'quiz', 'coding_submission', 'mentor_evaluation', 'questionnaire', 'external')),
            event_type TEXT NOT NULL,
            metric_name TEXT,
            metric_value REAL,
            payload_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (intern_id) REFERENCES interns(intern_id)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_intern_week ON ml_observations(intern_id, week);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_timestamp ON ml_observations(timestamp);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_source ON ml_observations(source);")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_dataset_versions (
            dataset_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_tag TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL,
            source_composition TEXT NOT NULL,
            total_row_count INTEGER NOT NULL,
            synthetic_row_count INTEGER NOT NULL,
            application_row_count INTEGER NOT NULL,
            external_row_count INTEGER DEFAULT 0,
            feature_count INTEGER NOT NULL,
            training_eligible_rows INTEGER NOT NULL,
            artifact_path TEXT NOT NULL,
            description TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT UNIQUE NOT NULL,
            model_type TEXT NOT NULL,
            algorithm_cls TEXT NOT NULL,
            algorithm_reg TEXT NOT NULL,
            dataset_version TEXT NOT NULL,
            trained_at TEXT NOT NULL,
            validation_metrics TEXT NOT NULL,
            test_metrics TEXT,
            feature_version TEXT NOT NULL,
            artifact_path_cls TEXT NOT NULL,
            artifact_path_reg TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('candidate', 'production', 'archived', 'rejected')),
            promotion_reason TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS questionnaire_responses (
            response_id INTEGER PRIMARY KEY AUTOINCREMENT,
            intern_id INTEGER NOT NULL,
            week INTEGER NOT NULL,
            learning_hours REAL NOT NULL,
            tasks_completed INTEGER NOT NULL,
            videos_completed INTEGER NOT NULL,
            guides_completed INTEGER NOT NULL,
            average_time_minutes REAL NOT NULL,
            debugging_confidence REAL NOT NULL,
            code_review_score REAL NOT NULL,
            technical_focus_skill TEXT NOT NULL,
            challenge_encountered TEXT,
            submitted_at TEXT NOT NULL,
            FOREIGN KEY (intern_id) REFERENCES interns(intern_id),
            UNIQUE(intern_id, week)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_questionnaire_intern_week ON questionnaire_responses(intern_id, week);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_questionnaire_submitted_at ON questionnaire_responses(submitted_at);")

    connection.commit()
    connection.close()


if __name__ == "__main__":
    initialize_database()
    print("SkillTrack database initialized successfully!")
    print(f"Database: {DATABASE_PATH}")