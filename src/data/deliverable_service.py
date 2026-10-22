"""
SkillTrack-ML: Project Deliverable Service
Handles secure validation, safe file storage, version control, metadata persistence,
mentor evaluation, and ML observation emission for PDF project deliverables.
"""

import os
import hashlib
import uuid
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"
UPLOAD_DIR = BASE_DIR / "data" / "deliverables"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# File constraints
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB limit
PDF_MAGIC_HEADER = b"%PDF-"


class DeliverableValidationError(Exception):
    pass


def get_db_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def validate_pdf_content(file_bytes: bytes, original_filename: str) -> None:
    """
    Validates file extension, size, and binary magic bytes.
    Does not rely on client-supplied MIME headers.
    """
    if not original_filename or not original_filename.lower().endswith(".pdf"):
        raise DeliverableValidationError("Invalid file extension: Deliverable must be a .pdf file.")

    if len(file_bytes) == 0:
        raise DeliverableValidationError("Uploaded file is empty (0 bytes).")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise DeliverableValidationError(f"File size exceeds maximum limit of {max_mb:.0f} MB.")

    if not file_bytes.startswith(PDF_MAGIC_HEADER):
        raise DeliverableValidationError("Invalid file format: File header is not a valid PDF document.")


def get_all_milestones(db_path: Path = DATABASE_PATH) -> List[Dict[str, Any]]:
    """Returns all curriculum project milestones."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT milestone_id, milestone_number, title, track, week, description,
               deliverable_requirements, rubric_criteria_json, max_score
        FROM project_milestones
        ORDER BY milestone_number ASC
    """)
    rows = cur.fetchall()
    conn.close()

    milestones = []
    for r in rows:
        rubric = {}
        if r[7]:
            try:
                rubric = json.loads(r[7])
            except Exception:
                rubric = {}
        milestones.append({
            "milestone_id": r[0],
            "milestone_number": r[1],
            "title": r[2],
            "track": r[3],
            "week": r[4],
            "description": r[5],
            "deliverable_requirements": r[6],
            "rubric": rubric,
            "max_score": r[8]
        })
    return milestones


def get_milestone_by_id(milestone_id: int, db_path: Path = DATABASE_PATH) -> Optional[Dict[str, Any]]:
    milestones = get_all_milestones(db_path)
    for m in milestones:
        if m["milestone_id"] == milestone_id:
            return m
    return None


def get_intern_deliverable_history(intern_id: int, db_path: Path = DATABASE_PATH) -> List[Dict[str, Any]]:
    """Retrieves all PDF deliverable submissions made by an intern with full version history."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT s.submission_id, s.intern_id, s.milestone_id, s.version_number,
               s.title, s.description, s.intern_notes, s.original_filename,
               s.stored_filename, s.file_path, s.file_size_bytes, s.file_hash,
               s.self_assessment_score, s.self_assessment_notes, s.review_status,
               s.mentor_score, s.mentor_feedback, s.reviewed_by, s.reviewed_at,
               s.submitted_at, m.title as milestone_title, m.milestone_number, m.track, m.week
        FROM project_submissions s
        JOIN project_milestones m ON s.milestone_id = m.milestone_id
        WHERE s.intern_id = ?
        ORDER BY m.milestone_number ASC, s.version_number DESC
    """, (intern_id,))
    rows = cur.fetchall()
    conn.close()

    history = []
    for r in rows:
        history.append({
            "submission_id": r[0],
            "intern_id": r[1],
            "milestone_id": r[2],
            "version_number": r[3],
            "title": r[4],
            "description": r[5],
            "intern_notes": r[6],
            "original_filename": r[7],
            "stored_filename": r[8],
            "file_path": r[9],
            "file_size_bytes": r[10],
            "file_hash": r[11],
            "self_assessment_score": r[12],
            "self_assessment_notes": r[13],
            "review_status": r[14],
            "mentor_score": r[15],
            "mentor_feedback": r[16],
            "reviewed_by": r[17],
            "reviewed_at": r[18],
            "submitted_at": r[19],
            "milestone_title": r[20],
            "milestone_number": r[21],
            "track": r[22],
            "week": r[23]
        })
    return history


def get_next_version_number(intern_id: int, milestone_id: int, db_path: Path = DATABASE_PATH) -> int:
    """Returns the next sequential version number for an intern's milestone submission."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT MAX(version_number) FROM project_submissions
        WHERE intern_id = ? AND milestone_id = ?
    """, (intern_id, milestone_id))
    max_ver = cur.fetchone()[0]
    conn.close()
    return (max_ver or 0) + 1


def save_pdf_deliverable(
    intern_id: int,
    milestone_id: int,
    title: str,
    file_bytes: bytes,
    original_filename: str,
    description: Optional[str] = None,
    intern_notes: Optional[str] = None,
    self_assessment_score: Optional[float] = None,
    self_assessment_notes: Optional[str] = None,
    db_path: Path = DATABASE_PATH,
    storage_dir: Path = UPLOAD_DIR
) -> Dict[str, Any]:
    """
    Validates PDF, generates secure server storage path, creates versioned DB entry,
    and safely logs telemetry to ML observations without fabricating compiler metrics.
    """
    # 1. Validation
    validate_pdf_content(file_bytes, original_filename)

    if not title or not title.strip():
        raise DeliverableValidationError("Submission title cannot be empty.")

    milestone = get_milestone_by_id(milestone_id, db_path)
    if not milestone:
        raise DeliverableValidationError(f"Invalid milestone ID: {milestone_id}")

    # Check for self-assessment score boundaries
    if self_assessment_score is not None:
        if self_assessment_score < 0.0 or self_assessment_score > 100.0:
            raise DeliverableValidationError("Self-assessment score must be between 0 and 100.")

    # 2. Version determination
    version_num = get_next_version_number(intern_id, milestone_id, db_path)

    # 3. Secure file storage (Safe UUID filename to prevent directory traversal & name collision)
    storage_dir.mkdir(parents=True, exist_ok=True)
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    safe_uuid = uuid.uuid4().hex[:12]
    stored_filename = f"intern_{intern_id}_m{milestone['milestone_number']}_v{version_num}_{safe_uuid}.pdf"
    target_path = storage_dir / stored_filename

    # Write file safely
    with open(target_path, "wb") as f:
        f.write(file_bytes)

    now_iso = datetime.now().isoformat(timespec="seconds")

    # 4. Save metadata to project_submissions
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO project_submissions (
            intern_id, milestone_id, version_number, title, description,
            intern_notes, original_filename, stored_filename, file_path,
            file_size_bytes, file_hash, self_assessment_score,
            self_assessment_notes, review_status, submitted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'SUBMITTED', ?)
    """, (
        intern_id, milestone_id, version_num, title.strip(), description,
        intern_notes, original_filename, stored_filename, str(target_path),
        len(file_bytes), file_hash, self_assessment_score,
        self_assessment_notes, now_iso
    ))
    submission_id = cur.lastrowid
    conn.commit()
    conn.close()

    # 5. ML Observation Telemetry (Transparent Fallback: event_type="deliverable_submitted")
    try:
        from src.data.live_data_ingestion import ingest_observation
        payload = {
            "submission_id": submission_id,
            "milestone_id": milestone_id,
            "milestone_number": milestone["milestone_number"],
            "version_number": version_num,
            "title": title.strip(),
            "file_size_bytes": len(file_bytes),
            "self_assessment_score": self_assessment_score,
            "status": "SUBMITTED",
            "submission_type": "pdf_deliverable"
        }
        ingest_observation(
            intern_id=intern_id,
            week=milestone["week"],
            source="application",
            event_type="deliverable_submitted",
            metric_name="tasks_completed",
            metric_value=float(version_num),
            payload=payload,
            db_path=db_path,
            label_status="IMMEDIATE_LABELED",
            training_eligible=1,
            outcome_value=self_assessment_score if self_assessment_score is not None else 80.0
        )
    except Exception as ex:
        # Observability fallback: logging errors shouldn't crash submissions
        print(f"[DeliverableService] Telemetry ingestion notice: {ex}")

    return {
        "submission_id": submission_id,
        "intern_id": intern_id,
        "milestone_id": milestone_id,
        "version_number": version_num,
        "stored_filename": stored_filename,
        "review_status": "SUBMITTED",
        "submitted_at": now_iso
    }


def record_mentor_deliverable_review(
    submission_id: int,
    mentor_id: int,
    review_status: str,
    score: float,
    feedback: str,
    rubric_scores: Optional[Dict[str, float]] = None,
    db_path: Path = DATABASE_PATH
) -> Dict[str, Any]:
    """
    Submits official mentor evaluation on an intern's PDF deliverable.
    Updates submission status, logs to review history, and ingests an authentic mentor_evaluation event.
    """
    valid_statuses = ("UNDER_REVIEW", "CHANGES_REQUESTED", "APPROVED")
    if review_status not in valid_statuses:
        raise DeliverableValidationError(f"Invalid review status: '{review_status}'. Must be one of {valid_statuses}")

    if score < 0.0 or score > 100.0:
        raise DeliverableValidationError("Evaluation score must be between 0 and 100.")

    if not feedback or not feedback.strip():
        raise DeliverableValidationError("Mentor feedback cannot be empty.")

    now_iso = datetime.now().isoformat(timespec="seconds")
    rubric_json = json.dumps(rubric_scores or {})

    conn = get_db_connection(db_path)
    cur = conn.cursor()

    # Get submission details
    cur.execute("""
        SELECT intern_id, milestone_id, version_number FROM project_submissions
        WHERE submission_id = ?
    """, (submission_id,))
    sub_row = cur.fetchone()
    if not sub_row:
        conn.close()
        raise DeliverableValidationError(f"Submission ID {submission_id} not found.")

    intern_id, milestone_id, version_num = sub_row

    # Update project_submissions
    cur.execute("""
        UPDATE project_submissions
        SET review_status = ?,
            mentor_score = ?,
            mentor_feedback = ?,
            reviewed_by = ?,
            reviewed_at = ?
        WHERE submission_id = ?
    """, (review_status, score, feedback.strip(), mentor_id, now_iso, submission_id))

    # Insert into submission_reviews audit log
    cur.execute("""
        INSERT INTO submission_reviews (
            submission_id, mentor_id, review_status, score, rubric_breakdown_json, feedback, reviewed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (submission_id, mentor_id, review_status, score, rubric_json, feedback.strip(), now_iso))

    conn.commit()
    conn.close()

    # Emit ML Observation for mentor evaluation
    try:
        from src.data.live_data_ingestion import ingest_observation
        milestone = get_milestone_by_id(milestone_id, db_path)
        week = milestone["week"] if milestone else 10

        obs_payload = {
            "submission_id": submission_id,
            "milestone_id": milestone_id,
            "version_number": version_num,
            "mentor_id": mentor_id,
            "review_status": review_status,
            "rubric_scores": rubric_scores or {},
            "feedback_snippet": feedback.strip()[:100]
        }

        ingest_observation(
            intern_id=intern_id,
            week=week,
            source="mentor_evaluation",
            event_type="deliverable_reviewed",
            metric_name="mentor_overall_score",
            metric_value=float(score),
            payload=obs_payload,
            db_path=db_path,
            label_status="IMMEDIATE_LABELED",
            training_eligible=1,
            outcome_value=float(score)
        )
    except Exception as ex:
        print(f"[DeliverableService] Mentor review telemetry notice: {ex}")

    return {
        "submission_id": submission_id,
        "intern_id": intern_id,
        "review_status": review_status,
        "score": score,
        "reviewed_at": now_iso
    }


def get_all_submissions_for_mentor(
    status_filter: Optional[str] = None,
    milestone_filter: Optional[int] = None,
    db_path: Path = DATABASE_PATH
) -> List[Dict[str, Any]]:
    """Retrieves all submissions across all interns for cohort mentor inspection."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    query = """
        SELECT s.submission_id, s.intern_id, i.name as intern_name, i.email as intern_email,
               s.milestone_id, m.milestone_number, m.title as milestone_title, m.track,
               s.version_number, s.title, s.description, s.intern_notes,
               s.original_filename, s.stored_filename, s.file_path, s.file_size_bytes,
               s.self_assessment_score, s.review_status, s.mentor_score, s.mentor_feedback,
               s.submitted_at, s.reviewed_at, m.max_score
        FROM project_submissions s
        JOIN interns i ON s.intern_id = i.intern_id
        JOIN project_milestones m ON s.milestone_id = m.milestone_id
        WHERE 1=1
    """
    params = []

    if status_filter and status_filter != "All Statuses":
        query += " AND s.review_status = ?"
        params.append(status_filter)

    if milestone_filter is not None and milestone_filter > 0:
        query += " AND m.milestone_number = ?"
        params.append(milestone_filter)

    query += " ORDER BY s.submitted_at DESC"

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "submission_id": r[0],
            "intern_id": r[1],
            "intern_name": r[2],
            "intern_email": r[3],
            "milestone_id": r[4],
            "milestone_number": r[5],
            "milestone_title": r[6],
            "track": r[7],
            "version_number": r[8],
            "title": r[9],
            "description": r[10],
            "intern_notes": r[11],
            "original_filename": r[12],
            "stored_filename": r[13],
            "file_path": r[14],
            "file_size_bytes": r[15],
            "self_assessment_score": r[16],
            "review_status": r[17],
            "mentor_score": r[18],
            "mentor_feedback": r[19],
            "submitted_at": r[20],
            "reviewed_at": r[21],
            "max_score": r[22]
        })
    return results


def get_submission_by_id(submission_id: int, db_path: Path = DATABASE_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves a single submission with full details and audit review history."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT s.submission_id, s.intern_id, i.name as intern_name, i.email as intern_email,
               s.milestone_id, m.milestone_number, m.title as milestone_title, m.track, m.week,
               m.description as milestone_desc, m.deliverable_requirements, m.rubric_criteria_json,
               s.version_number, s.title, s.description, s.intern_notes,
               s.original_filename, s.stored_filename, s.file_path, s.file_size_bytes,
               s.self_assessment_score, s.self_assessment_notes, s.review_status,
               s.mentor_score, s.mentor_feedback, s.reviewed_by, s.reviewed_at, s.submitted_at
        FROM project_submissions s
        JOIN interns i ON s.intern_id = i.intern_id
        JOIN project_milestones m ON s.milestone_id = m.milestone_id
        WHERE s.submission_id = ?
    """, (submission_id,))
    r = cur.fetchone()

    if not r:
        conn.close()
        return None

    # Fetch audit reviews
    cur.execute("""
        SELECT review_id, mentor_id, review_status, score, rubric_breakdown_json, feedback, reviewed_at
        FROM submission_reviews
        WHERE submission_id = ?
        ORDER BY reviewed_at DESC
    """, (submission_id,))
    review_rows = cur.fetchall()
    conn.close()

    reviews = []
    for rv in review_rows:
        rubric = {}
        if rv[4]:
            try:
                rubric = json.loads(rv[4])
            except Exception:
                rubric = {}
        reviews.append({
            "review_id": rv[0],
            "mentor_id": rv[1],
            "review_status": rv[2],
            "score": rv[3],
            "rubric": rubric,
            "feedback": rv[5],
            "reviewed_at": rv[6]
        })

    milestone_rubric = {}
    if r[11]:
        try:
            milestone_rubric = json.loads(r[11])
        except Exception:
            milestone_rubric = {}

    return {
        "submission_id": r[0],
        "intern_id": r[1],
        "intern_name": r[2],
        "intern_email": r[3],
        "milestone_id": r[4],
        "milestone_number": r[5],
        "milestone_title": r[6],
        "track": r[7],
        "week": r[8],
        "milestone_desc": r[9],
        "deliverable_requirements": r[10],
        "rubric": milestone_rubric,
        "version_number": r[12],
        "title": r[13],
        "description": r[14],
        "intern_notes": r[15],
        "original_filename": r[16],
        "stored_filename": r[17],
        "file_path": r[18],
        "file_size_bytes": r[19],
        "self_assessment_score": r[20],
        "self_assessment_notes": r[21],
        "review_status": r[22],
        "mentor_score": r[23],
        "mentor_feedback": r[24],
        "reviewed_by": r[25],
        "reviewed_at": r[26],
        "submitted_at": r[27],
        "reviews": reviews
    }


def get_cohort_deliverable_metrics(db_path: Path = DATABASE_PATH) -> Dict[str, Any]:
    """Computes summary metrics for mentor dashboard overview."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(DISTINCT intern_id) FROM interns")
    total_interns = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM project_submissions WHERE review_status = 'SUBMITTED'")
    pending_reviews = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM project_submissions WHERE review_status = 'APPROVED'")
    approved_count = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM project_submissions WHERE review_status = 'CHANGES_REQUESTED'")
    changes_requested_count = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM project_submissions")
    total_submissions = cur.fetchone()[0] or 0

    conn.close()

    return {
        "total_interns": total_interns,
        "pending_reviews": pending_reviews,
        "approved_count": approved_count,
        "changes_requested_count": changes_requested_count,
        "total_submissions": total_submissions
    }
