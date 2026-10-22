import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATABASE_PATH = (
    BASE_DIR
    / "data"
    / "database"
    / "skilltrack.db"
)

OLD_TASKS = [
    "Python Basics",
    "String Manipulation",
    "Data Structures",
    "SQL Basics",
    "REST API Task",
]

connection = sqlite3.connect(DATABASE_PATH)
cursor = connection.cursor()

for title in OLD_TASKS:
    cursor.execute(
        "DELETE FROM coding_tasks WHERE title = ?",
        (title,)
    )

connection.commit()

remaining = cursor.execute(
    "SELECT COUNT(*) FROM coding_tasks"
).fetchone()[0]

print(f"Remaining coding tasks: {remaining}")

print("\nCurrent tasks:")

tasks = cursor.execute(
    "SELECT task_id, title FROM coding_tasks ORDER BY task_id"
).fetchall()

for task in tasks:
    print(task)

connection.close()