import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DATABASE_PATH = (
    BASE_DIR
    / "data"
    / "database"
    / "skilltrack.db"
)


TASKS = [

    (
        "HTML Portfolio Card",
        "Create a semantic portfolio card containing a profile image, name, role, short description and a contact button.",
        "HTML",
        "Easy",
        "https://www.youtube.com/watch?v=qz0aGYrrlhU",
        "https://developer.mozilla.org/en-US/docs/Learn_web_development/Core/Structuring_content"
    ),

    (
        "Registration Form",
        "Build a registration form containing name, email, password, date of birth and a submit button. Use appropriate HTML form elements.",
        "HTML",
        "Easy",
        "https://www.youtube.com/watch?v=fNcJuPIZ2WE",
        "https://developer.mozilla.org/en-US/docs/Learn_web_development/Extensions/Forms"
    ),

    (
        "Responsive Card Layout",
        "Create a responsive layout containing three cards. The cards should appear in a row on larger screens and stack vertically on smaller screens.",
        "CSS",
        "Easy",
        "https://www.youtube.com/watch?v=phWxA89Dy94",
        "https://developer.mozilla.org/en-US/docs/Learn_web_development/Core/CSS_layout/Flexbox"
    ),

    (
        "Responsive Navigation Bar",
        "Build a responsive navigation bar containing a logo and navigation links. The layout should adapt to smaller screen sizes.",
        "CSS",
        "Medium",
        "https://www.youtube.com/watch?v=oLgtucwjVII",
        "https://developer.mozilla.org/en-US/docs/Learn_web_development/Core/CSS_layout/Responsive_Design"
    ),

    (
        "Form Validation",
        "Create client-side validation for a registration form. Validate required fields, email format and password length and display useful error messages.",
        "JavaScript",
        "Medium",
        "https://www.youtube.com/watch?v=hdI2bqOjy3c",
        "https://developer.mozilla.org/en-US/docs/Learn_web_development/Extensions/Forms/Form_validation"
    ),

    (
        "Live Search Filter",
        "Create a search interface that filters a list of products as the user types. Matching results should remain visible while non-matching results are hidden.",
        "JavaScript",
        "Medium",
        "https://www.youtube.com/watch?v=TlP5WIxVirU",
        "https://developer.mozilla.org/en-US/docs/Web/API/Element"
    ),

    (
        "Todo List",
        "Build an interactive todo list. Users should be able to add tasks, mark tasks as completed and remove tasks.",
        "JavaScript",
        "Medium",
        "https://www.youtube.com/watch?v=3OqZsB9wW4E",
        "https://developer.mozilla.org/en-US/docs/Web/JavaScript"
    ),

    (
        "React Counter",
        "Build a counter interface with increment, decrement and reset controls. Manage the counter value using React state.",
        "React",
        "Easy",
        "https://www.youtube.com/watch?v=SqcY0GlETPk",
        "https://react.dev/learn"
    ),

    (
        "React Todo App",
        "Build a Todo application where users can add tasks, mark tasks as completed and remove tasks. Use React components and state.",
        "React",
        "Hard",
        "https://www.youtube.com/watch?v=SqcY0GlETPk",
        "https://react.dev/learn"
    ),

    (
        "API User Directory",
        "Fetch user information from a REST API and display the users in a clean card-based interface. Include loading and error states.",
        "API Integration",
        "Hard",
        "https://www.youtube.com/watch?v=cuEtnrL9-H0",
        "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API"
    )
]


def ensure_columns(cursor):

    columns = cursor.execute(
        "PRAGMA table_info(coding_tasks)"
    ).fetchall()

    existing_columns = {
        column[1]
        for column in columns
    }

    if "skill" not in existing_columns:

        cursor.execute(
            "ALTER TABLE coding_tasks ADD COLUMN skill TEXT"
        )

    if "difficulty" not in existing_columns:

        cursor.execute(
            "ALTER TABLE coding_tasks ADD COLUMN difficulty TEXT"
        )

    if "video_url" not in existing_columns:

        cursor.execute(
            "ALTER TABLE coding_tasks ADD COLUMN video_url TEXT"
        )

    if "guide_url" not in existing_columns:

        cursor.execute(
            "ALTER TABLE coding_tasks ADD COLUMN guide_url TEXT"
        )


def seed_tasks():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    ensure_columns(cursor)

    for (
        title,
        description,
        skill,
        difficulty,
        video_url,
        guide_url
    ) in TASKS:

        cursor.execute(
            """
            INSERT INTO coding_tasks
            (
                title,
                description,
                max_score,
                skill,
                difficulty,
                video_url,
                guide_url
            )
            SELECT ?, ?, ?, ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1
                FROM coding_tasks
                WHERE title = ?
            )
            """,
            (
                title,
                description,
                100,
                skill,
                difficulty,
                video_url,
                guide_url,
                title
            )
        )

    connection.commit()

    count = cursor.execute(
        "SELECT COUNT(*) FROM coding_tasks"
    ).fetchone()[0]

    connection.close()

    print(
        f"Frontend coding tasks available: {count}"
    )


if __name__ == "__main__":

    seed_tasks()