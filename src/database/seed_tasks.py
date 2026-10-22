import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DATABASE_PATH = (
    BASE_DIR
    / "data"
    / "database"
    / "skilltrack.db"
)


def seed_tasks():

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    tasks = [
        (
            "Python Basics",
            "Write a Python program to find the largest number in a list.",
            100
        ),
        (
            "String Manipulation",
            "Write a program to check whether a given string is a palindrome.",
            100
        ),
        (
            "Data Structures",
            "Implement a stack using Python and demonstrate push and pop operations.",
            100
        ),
        (
            "SQL Basics",
            "Write an SQL query to retrieve employees whose salary is greater than the average salary.",
            100
        ),
        (
            "REST API Task",
            "Create a simple REST API endpoint that returns a list of users.",
            100
        )
    ]

    for title, description, max_score in tasks:

        cursor.execute(
            """
            INSERT INTO coding_tasks
            (title, description, max_score)
            SELECT ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1
                FROM coding_tasks
                WHERE title = ?
            )
            """,
            (
                title,
                description,
                max_score,
                title
            )
        )

    connection.commit()

    count = cursor.execute(
        "SELECT COUNT(*) FROM coding_tasks"
    ).fetchone()[0]

    connection.close()

    print(
        f"Coding tasks available: {count}"
    )


if __name__ == "__main__":
    seed_tasks()