"""
SkillTrack-ML: Learning Resource Service
Provides access to curated educational programming materials (Java, OOP, etc.).
Generates deterministic, ML-guided resource recommendations linked to detected skill gaps.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


def get_curated_resources(
    category: Optional[str] = None,
    language: Optional[str] = None,
    difficulty: Optional[str] = None,
    resource_type: Optional[str] = None,
    db_path: Path = DATABASE_PATH
) -> List[Dict[str, Any]]:
    """
    Returns filtered educational resources from SQLite.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    query = """
        SELECT resource_id, title, category, language, provider,
               url, resource_type, difficulty, description
        FROM resources
        WHERE 1=1
    """
    params: List[Any] = []

    if category and category != "All Categories":
        query += " AND category = ?"
        params.append(category)

    if language and language != "All Languages":
        query += " AND language = ?"
        params.append(language)

    if difficulty and difficulty != "All Levels":
        query += " AND difficulty = ?"
        params.append(difficulty)

    if resource_type and resource_type != "All Formats":
        query += " AND resource_type = ?"
        params.append(resource_type.lower())

    query += " ORDER BY resource_id ASC"

    cur.execute(query, params)
    cols = [col[0] for col in cur.description]
    rows = cur.fetchall()
    conn.close()

    return [dict(zip(cols, row)) for row in rows]


def get_available_filters(db_path: Path = DATABASE_PATH) -> Dict[str, List[str]]:
    """
    Returns unique categories, languages, and difficulty levels for UI dropdowns.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT category FROM resources WHERE category IS NOT NULL ORDER BY category")
    categories = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT DISTINCT language FROM resources WHERE language IS NOT NULL ORDER BY language")
    languages = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT DISTINCT difficulty FROM resources WHERE difficulty IS NOT NULL ORDER BY difficulty")
    difficulties = [r[0] for r in cur.fetchall()]

    conn.close()

    return {
        "categories": ["All Categories"] + categories,
        "languages": ["All Languages"] + languages,
        "difficulties": ["All Levels"] + difficulties,
        "resource_types": ["All Formats", "Video", "Documentation"]
    }


def get_recommended_resources(
    intern_id: int,
    db_path: Path = DATABASE_PATH
) -> Dict[str, Any]:
    """
    Determines recommended learning resources based on the intern's active skill gap:
    - Needs Improvement (<55 or Needs Improvement): Foundations (Java Basics, OOP, Exception Handling)
    - Developing (55-72 or Developing): Core Data Structures (Collections, Streams, Data Structures)
    - Strong (>=72 or Strong): Advanced Engineering (Multithreading, JDBC, Algorithms)
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT skill_score, skill_gap, final_performance_score
        FROM interns
        WHERE intern_id = ?
    """, (intern_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        target_categories = ["Java Basics", "OOP"]
        gap_label = "General Foundation"
        reason = "Starting curriculum for all engineering candidates."
    else:
        sk_score, gap, f_perf = row
        sk_score = float(sk_score) if sk_score is not None else 70.0
        gap = gap or "Developing"

        if gap == "Needs Improvement" or sk_score < 55.0:
            target_categories = ["Java Basics", "OOP", "Exception Handling"]
            gap_label = "Foundational Mastery Needed"
            reason = f"Current skill score is {sk_score:.1f}. Focus on core syntax, OOP architecture, and exception control."
        elif gap == "Developing" or sk_score < 72.0:
            target_categories = ["Collections", "Streams", "Data Structures"]
            gap_label = "Intermediate Acceleration"
            reason = f"Current skill score is {sk_score:.1f}. Elevate performance with collection pipelines, functional streams, and algorithmic indexing."
        else:
            target_categories = ["Multithreading", "JDBC", "Algorithms"]
            gap_label = "Advanced Engineering"
            reason = f"High performance trajectory ({sk_score:.1f}). Advance into concurrent systems, persistence drivers, and complexity optimization."

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    placeholders = ",".join(["?"] * len(target_categories))
    cur.execute(f"""
        SELECT resource_id, title, category, language, provider,
               url, resource_type, difficulty, description
        FROM resources
        WHERE category IN ({placeholders})
        ORDER BY resource_id ASC
        LIMIT 4
    """, target_categories)
    cols = [col[0] for col in cur.description]
    rows = cur.fetchall()
    conn.close()

    return {
        "gap_label": gap_label,
        "reason": reason,
        "target_categories": target_categories,
        "recommendations": [dict(zip(cols, row)) for row in rows]
    }
