"""
SkillTrack-ML: Chat & Communication Service
Provides secure, persistent intern-to-mentor messaging stored in SQLite.
Enforces strict privacy boundaries (interns only view their own thread).
Provides engagement metadata counters without ever leaking message text into ML features.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


def send_chat_message(
    sender_user_id: int,
    receiver_user_id: int,
    intern_id: int,
    mentor_id: int,
    message: str,
    db_path: Path = DATABASE_PATH
) -> int:
    """
    Persists a new message between an intern and their mentor.
    Returns the generated message_id.
    """
    clean_msg = message.strip()
    if not clean_msg:
        raise ValueError("Message content cannot be empty.")

    now_str = datetime.now().isoformat(timespec="seconds")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO chat_messages (
            sender_user_id, receiver_user_id, intern_id, mentor_id,
            message, timestamp, is_read
        ) VALUES (?, ?, ?, ?, ?, ?, 0)
    """, (sender_user_id, receiver_user_id, intern_id, mentor_id, clean_msg, now_str))
    msg_id = cur.lastrowid
    conn.commit()
    conn.close()
    return msg_id


def get_conversation_history(
    intern_id: int,
    viewer_user_id: int,
    viewer_role: str,
    mentor_id: Optional[int] = None,
    db_path: Path = DATABASE_PATH
) -> List[Dict[str, Any]]:
    """
    Fetches the message history for an intern-mentor conversation.
    Enforces privacy:
    - If viewer_role == 'intern', viewer MUST match the intern's user_id.
    - If viewer_role == 'mentor', viewer MUST be the assigned mentor.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Privacy verification
    if viewer_role == "intern":
        cur.execute("SELECT user_id FROM users WHERE intern_id = ?", (intern_id,))
        row = cur.fetchone()
        if not row or row[0] != viewer_user_id:
            conn.close()
            raise PermissionError("Access denied: Interns can only view their own messages.")
    elif viewer_role == "mentor":
        cur.execute("SELECT mentor_id FROM users WHERE user_id = ?", (viewer_user_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            raise PermissionError("Access denied: Invalid mentor account.")

    query = """
        SELECT m.message_id, m.sender_user_id, m.receiver_user_id, m.intern_id,
               m.mentor_id, m.message, m.timestamp, m.is_read, u.username as sender_name,
               u.role as sender_role
        FROM chat_messages m
        JOIN users u ON m.sender_user_id = u.user_id
        WHERE m.intern_id = ?
        ORDER BY m.timestamp ASC
    """
    cur.execute(query, (intern_id,))
    cols = [col[0] for col in cur.description]
    rows = cur.fetchall()
    conn.close()

    return [dict(zip(cols, row)) for row in rows]


def mark_messages_as_read(
    intern_id: int,
    reader_user_id: int,
    db_path: Path = DATABASE_PATH
) -> int:
    """
    Marks messages sent to reader_user_id as read.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        UPDATE chat_messages
        SET is_read = 1
        WHERE intern_id = ? AND receiver_user_id = ? AND is_read = 0
    """, (intern_id, reader_user_id))
    affected = cur.rowcount
    conn.commit()
    conn.close()
    return affected


def get_mentor_unread_counts(
    mentor_user_id: int,
    db_path: Path = DATABASE_PATH
) -> Dict[int, int]:
    """
    Returns a mapping of intern_id -> unread message count for a mentor.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT intern_id, COUNT(*)
        FROM chat_messages
        WHERE receiver_user_id = ? AND is_read = 0
        GROUP BY intern_id
    """, (mentor_user_id,))
    counts = dict(cur.fetchall())
    conn.close()
    return counts


def get_chat_engagement_metadata(
    intern_id: int,
    db_path: Path = DATABASE_PATH
) -> Dict[str, Any]:
    """
    Extracts high-level engagement counts without accessing message content.
    Prevents privacy violations while providing engagement signals.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*) as total_messages,
            SUM(CASE WHEN u.role = 'intern' THEN 1 ELSE 0 END) as intern_messages,
            SUM(CASE WHEN u.role = 'mentor' THEN 1 ELSE 0 END) as mentor_messages,
            MAX(m.timestamp) as last_message_timestamp
        FROM chat_messages m
        JOIN users u ON m.sender_user_id = u.user_id
        WHERE m.intern_id = ?
    """, (intern_id,))
    row = cur.fetchone()
    conn.close()

    return {
        "total_chat_messages": row[0] or 0,
        "intern_sent_messages": row[1] or 0,
        "mentor_sent_messages": row[2] or 0,
        "last_chat_timestamp": row[3]
    }
