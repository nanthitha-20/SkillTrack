import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


def seed_users():
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    # Get one existing intern
    intern = cursor.execute(
        "SELECT intern_id FROM interns LIMIT 1"
    ).fetchone()

    if intern:
        intern_id = intern[0]

        cursor.execute("""
            INSERT OR IGNORE INTO users
            (username, password, role, intern_id)
            VALUES (?, ?, ?, ?)
        """, (
            "intern1",
            "intern123",
            "intern",
            intern_id
        ))

    # Create demo mentor
    cursor.execute("""
        INSERT OR IGNORE INTO mentors
        (name, email)
        VALUES (?, ?)
    """, (
        "Demo Mentor",
        "mentor@skilltrack.local"
    ))

    mentor = cursor.execute(
        "SELECT mentor_id FROM mentors WHERE email = ?",
        ("mentor@skilltrack.local",)
    ).fetchone()

    if mentor:
        mentor_id = mentor[0]

        cursor.execute("""
            INSERT OR IGNORE INTO users
            (username, password, role, mentor_id)
            VALUES (?, ?, ?, ?)
        """, (
            "mentor1",
            "mentor123",
            "mentor",
            mentor_id
        ))

    connection.commit()
    connection.close()

    print("Demo users created successfully!")


if __name__ == "__main__":
    seed_users()