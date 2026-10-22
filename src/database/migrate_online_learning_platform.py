"""
SkillTrack-ML: Online Learning Platform Migration
Extends the schema with:
1. ml_observations fields: label_status, training_eligible, outcome_value, target_week, source_record_id
2. chat_messages table (persistent intern-to-mentor messaging)
3. online_learning_updates table (tracks incremental partial_fit updates)
4. resources table extension with curated Java and Frontend learning materials
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


def run_migration(db_path: Path = DATABASE_PATH):
    print("=" * 70)
    print("SKILLTRACK-ML: ONLINE LEARNING & COMMUNICATION SCHEMA MIGRATION")
    print("=" * 70)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Extend ml_observations with label_status and online learning flags
    cursor.execute("PRAGMA table_info(ml_observations)")
    obs_cols = [c[1] for c in cursor.fetchall()]

    new_obs_cols = [
        ("label_status", "TEXT DEFAULT 'IMMEDIATE_LABELED'"),
        ("training_eligible", "INTEGER DEFAULT 1"),
        ("outcome_value", "REAL"),
        ("target_week", "INTEGER"),
        ("source_record_id", "INTEGER")
    ]

    for col_name, col_def in new_obs_cols:
        if col_name not in obs_cols:
            cursor.execute(f"ALTER TABLE ml_observations ADD COLUMN {col_name} {col_def}")
            print(f"[OK] Added column '{col_name}' to ml_observations")

    # Update existing ml_observations with outcome_value = metric_value
    cursor.execute("""
        UPDATE ml_observations
        SET outcome_value = metric_value
        WHERE outcome_value IS NULL AND metric_value IS NOT NULL
    """)
    cursor.execute("""
        UPDATE ml_observations
        SET label_status = 'IMMEDIATE_LABELED', training_eligible = 1
        WHERE label_status IS NULL
    """)

    # 2. Create chat_messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_user_id INTEGER NOT NULL,
            receiver_user_id INTEGER NOT NULL,
            intern_id INTEGER NOT NULL,
            mentor_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            is_read INTEGER DEFAULT 0,
            FOREIGN KEY (intern_id) REFERENCES interns(intern_id),
            FOREIGN KEY (mentor_id) REFERENCES mentors(mentor_id)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_intern ON chat_messages(intern_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_mentor ON chat_messages(mentor_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON chat_messages(timestamp);")
    print("[OK] Created table: chat_messages (with indices)")

    # 3. Create online_learning_updates table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS online_learning_updates (
            update_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_name TEXT NOT NULL,
            event_type TEXT NOT NULL,
            samples_processed INTEGER NOT NULL,
            cumulative_samples INTEGER NOT NULL,
            current_loss REAL,
            model_path TEXT NOT NULL
        )
    """)
    print("[OK] Created table: online_learning_updates")

    # 4. Extend resources table schema
    cursor.execute("PRAGMA table_info(resources)")
    res_cols = [c[1] for c in cursor.fetchall()]

    new_res_cols = [
        ("category", "TEXT"),
        ("language", "TEXT DEFAULT 'Java'"),
        ("provider", "TEXT"),
        ("description", "TEXT")
    ]
    for col_name, col_def in new_res_cols:
        if col_name not in res_cols:
            cursor.execute(f"ALTER TABLE resources ADD COLUMN {col_name} {col_def}")
            print(f"[OK] Added column '{col_name}' to resources")

    # 5. Seed Curated Learning Resources (Java + Full Frontend/Fullstack Stack)
    cursor.execute("SELECT count(*) FROM resources")
    existing_res_count = cursor.fetchone()[0]

    if existing_res_count == 0:
        resources_data = [
            # Java Basics
            ("Java Full Course for Beginners", "video", "https://www.youtube.com/watch?v=A74TOX803D0", "Java Basics", "Beginner", "Java Basics", "Java", "freeCodeCamp.org", "Comprehensive 9-hour foundational course covering Java syntax, variables, conditionals, loops, and methods."),
            ("Oracle Official Java SE Documentation", "documentation", "https://docs.oracle.com/en/java/javase/17/docs/api/index.html", "Java Basics", "All Levels", "Java Basics", "Java", "Oracle", "Authoritative API specification and language reference for Java Standard Edition."),

            # Java OOP
            ("Object Oriented Programming in Java", "video", "https://www.youtube.com/watch?v=SiBw7HNVAQk", "OOP", "Beginner", "OOP", "Java", "freeCodeCamp.org", "In-depth guide to classes, encapsulation, inheritance, polymorphism, and abstraction in Java."),
            ("Oracle OOP Concepts Tutorial", "documentation", "https://docs.oracle.com/javase/tutorial/java/concepts/index.html", "OOP", "Beginner", "OOP", "Java", "Oracle", "Official Oracle guide explaining core Object-Oriented design principles."),

            # Java Collections
            ("Java Collections Framework Masterclass", "video", "https://www.youtube.com/watch?v=9sgZ8Z_1T3k", "Collections", "Intermediate", "Collections", "Java", "Telusko", "Complete deep-dive into List, Set, Map, ArrayList, LinkedList, and HashMap in Java."),
            ("Oracle Collections Framework Overview", "documentation", "https://docs.oracle.com/javase/8/docs/technotes/guides/collections/overview.html", "Collections", "Intermediate", "Collections", "Java", "Oracle", "Architectural overview and implementation guide for the Java Collections API."),

            # Exception Handling
            ("Exception Handling in Java Explained", "video", "https://www.youtube.com/watch?v=1XACTLTe1E8", "Exception Handling", "Beginner", "Exception Handling", "Java", "Programming with Mosh", "Learn try-catch-finally blocks, checked vs unchecked exceptions, and custom exception handling."),
            ("Oracle Java Exceptions Tutorial", "documentation", "https://docs.oracle.com/javase/tutorial/essential/exceptions/index.html", "Exception Handling", "Beginner", "Exception Handling", "Java", "Oracle", "Official guide on the exception hierarchy, throwing exceptions, and try-with-resources."),

            # Multithreading & Concurrency
            ("Java Multithreading & Concurrency Tutorial", "video", "https://www.youtube.com/watch?v=r_MbozD32eo", "Multithreading", "Advanced", "Multithreading", "Java", "Bro Code", "Learn Thread class, Runnable interface, synchronization, locks, and thread safety."),
            ("Oracle Java Concurrency Tutorial", "documentation", "https://docs.oracle.com/javase/tutorial/essential/concurrency/index.html", "Multithreading", "Advanced", "Multithreading", "Java", "Oracle", "Comprehensive reference on thread management, executor services, and atomic variables."),

            # File Handling
            ("Java File Handling & I/O Complete Guide", "video", "https://www.youtube.com/watch?v=k3K9KfdECrE", "File Handling", "Intermediate", "File Handling", "Java", "Amigoscode", "Learn reading and writing files using BufferedReader, FileReader, and Java NIO."),
            ("Oracle Java NIO File I/O Tutorial", "documentation", "https://docs.oracle.com/javase/tutorial/essential/io/fileio.html", "File Handling", "Intermediate", "File Handling", "Java", "Oracle", "Official guide to modern Java NIO Path, Files, and channel operations."),

            # JDBC
            ("Java JDBC Database Connectivity Crash Course", "video", "https://www.youtube.com/watch?v=2i4t-SL1kVU", "JDBC", "Intermediate", "JDBC", "Java", "Telusko", "Step-by-step connection to SQL databases using DriverManager, Statement, and PreparedStatement."),
            ("Oracle JDBC Basics Tutorial", "documentation", "https://docs.oracle.com/javase/tutorial/jdbc/basics/index.html", "JDBC", "Intermediate", "JDBC", "Java", "Oracle", "Official tutorial for establishing database connections and executing SQL queries in Java."),

            # Java Streams
            ("Java 8 Streams API in 1 Hour", "video", "https://www.youtube.com/watch?v=q6zF3kf884w", "Java Streams", "Intermediate", "Java Streams", "Java", "Amigoscode", "Hands-on tutorial on filter, map, flatMap, collect, and functional programming in Java."),
            ("Baeldung Complete Guide to Java 8 Streams", "documentation", "https://www.baeldung.com/java-8-streams", "Java Streams", "Intermediate", "Java Streams", "Java", "Baeldung", "Extensive reference on stream operations, parallel streams, and stream collectors."),

            # Data Structures & Algorithms
            ("Data Structures and Algorithms in Java", "video", "https://www.youtube.com/watch?v=8hly31xKli0", "Data Structures", "Intermediate", "Data Structures", "Java", "freeCodeCamp.org", "Complete course on arrays, linked lists, stacks, queues, trees, graphs, and sorting algorithms."),
            ("GeeksforGeeks Data Structures in Java", "documentation", "https://www.geeksforgeeks.org/data-structures-in-java/", "Data Structures", "Intermediate", "Data Structures", "Java", "GeeksforGeeks", "Curated reference with code examples for fundamental data structure implementations in Java."),

            # Frontend & Framework Resources for Skill-Gap Recommendations
            ("React Official Documentation & Interactive Tutorial", "documentation", "https://react.dev/learn", "React", "All Levels", "React", "JavaScript", "React Core Team", "The official interactive documentation for modern React hooks, state, and component architecture."),
            ("MDN Web Docs: Modern JavaScript Handbook", "documentation", "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide", "JavaScript", "All Levels", "JavaScript", "JavaScript", "Mozilla", "Authoritative guide to asynchronous programming, promises, DOM manipulation, and ES6+ features."),
            ("MDN Web Docs: CSS Layouts & Responsive Design", "documentation", "https://developer.mozilla.org/en-US/docs/Learn/CSS/CSS_layout", "CSS", "Beginner", "CSS", "CSS", "Mozilla", "Complete guide to Flexbox, Grid, media queries, and responsive web layouts."),
            ("RESTful API Design & HTTP Status Guide", "documentation", "https://restfulapi.net/", "API Integration", "Intermediate", "API Integration", "API", "RESTful API", "Industry-standard reference for REST endpoint design, JSON handling, and HTTP status codes.")
        ]

        cursor.executemany("""
            INSERT INTO resources (title, resource_type, url, topic, difficulty, category, language, provider, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, resources_data)
        print(f"[OK] Seeded {len(resources_data)} curated learning resources (Java + Web modules).")

    # 6. Seed sample welcome message from mentor to intern1
    cursor.execute("SELECT count(*) FROM chat_messages")
    if cursor.fetchone()[0] == 0:
        now_ts = datetime.now().isoformat(timespec="seconds")
        cursor.execute("""
            INSERT INTO chat_messages (sender_user_id, receiver_user_id, intern_id, mentor_id, message, timestamp, is_read)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            2, # mentor1
            1, # intern1
            10001,
            1,
            "Welcome to SkillTrack-ML! I am your assigned mentor. Feel free to reach out here if you have any questions about coding tasks, quizzes, or learning resources.",
            now_ts,
            0
        ))
        print("[OK] Seeded initial welcome chat message between mentor1 and intern1.")

    conn.commit()
    conn.close()
    print("=" * 70)
    print("ONLINE LEARNING & COMMUNICATION MIGRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_migration()
