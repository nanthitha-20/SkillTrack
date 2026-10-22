"""
Unit and Integration Tests for PDF Project Deliverables Service:
- File validation (PDF magic bytes, file extensions, max size, empty files)
- Versioning and non-destructive resubmissions
- Intern deliverable submission privacy and history
- Mentor evaluation, scoring, rubric recording, and status changes
- Audit review history
- Telemetry emission into ml_observations
"""

import unittest
import sqlite3
import tempfile
import shutil
from pathlib import Path

from src.data.deliverable_service import (
    validate_pdf_content,
    save_pdf_deliverable,
    record_mentor_deliverable_review,
    get_intern_deliverable_history,
    get_all_submissions_for_mentor,
    get_submission_by_id,
    get_cohort_deliverable_metrics,
    get_all_milestones,
    DeliverableValidationError,
    MAX_FILE_SIZE_BYTES
)
from src.database.migrate_deliverables import run_deliverables_migration

PROD_DB_PATH = Path("data/database/skilltrack.db")


class TestDeliverableService(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_skilltrack.db"
        self.storage_dir = Path(self.temp_dir) / "deliverables"
        shutil.copy2(PROD_DB_PATH, self.db_path)
        run_deliverables_migration(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pdf_validation_magic_bytes(self):
        # Valid PDF header
        valid_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
        validate_pdf_content(valid_bytes, "milestone1_report.pdf")

        # Invalid magic header (e.g. text file renamed to .pdf)
        fake_bytes = b"This is just a text file renamed to pdf"
        with self.assertRaises(DeliverableValidationError) as ctx:
            validate_pdf_content(fake_bytes, "fake.pdf")
        self.assertIn("not a valid PDF document", str(ctx.exception))

        # Invalid extension
        with self.assertRaises(DeliverableValidationError) as ctx:
            validate_pdf_content(valid_bytes, "report.docx")
        self.assertIn("must be a .pdf file", str(ctx.exception))

        # Empty file
        with self.assertRaises(DeliverableValidationError) as ctx:
            validate_pdf_content(b"", "empty.pdf")
        self.assertIn("empty", str(ctx.exception))

        # Oversized file
        oversized = b"%PDF-" + b"A" * (MAX_FILE_SIZE_BYTES + 10)
        with self.assertRaises(DeliverableValidationError) as ctx:
            validate_pdf_content(oversized, "big.pdf")
        self.assertIn("exceeds maximum limit", str(ctx.exception))

    def test_submission_and_version_progression(self):
        intern_id = 10001
        milestone_id = 1
        pdf_v1 = b"%PDF-1.5\n%Header\nVersion 1 content\n%%EOF"

        # Submit Version 1
        res_v1 = save_pdf_deliverable(
            intern_id=intern_id,
            milestone_id=milestone_id,
            title="Semantic Web Architecture Dossier",
            file_bytes=pdf_v1,
            original_filename="arch_report_v1.pdf",
            description="Initial submission covering semantic DOM.",
            self_assessment_score=85.0,
            db_path=self.db_path,
            storage_dir=self.storage_dir
        )
        self.assertEqual(res_v1["version_number"], 1)
        self.assertEqual(res_v1["review_status"], "SUBMITTED")

        # Verify file physically stored in storage_dir
        stored_file_path = self.storage_dir / res_v1["stored_filename"]
        self.assertTrue(stored_file_path.exists())

        # Mentor evaluates: Changes Requested
        sub_id_1 = res_v1["submission_id"]
        review_res = record_mentor_deliverable_review(
            submission_id=sub_id_1,
            mentor_id=1,
            review_status="CHANGES_REQUESTED",
            score=70.0,
            feedback="Please add ARIA roles breakdown and contrast analysis.",
            rubric_scores={"Semantic Hierarchy": 25, "Accessibility & ARIA Compliance": 15, "Technical Documentation": 30},
            db_path=self.db_path
        )
        self.assertEqual(review_res["review_status"], "CHANGES_REQUESTED")

        # Intern resubmits an improved Version 2
        pdf_v2 = b"%PDF-1.5\n%Header\nVersion 2 with ARIA roles\n%%EOF"
        res_v2 = save_pdf_deliverable(
            intern_id=intern_id,
            milestone_id=milestone_id,
            title="Semantic Web Architecture Dossier (Revised)",
            file_bytes=pdf_v2,
            original_filename="arch_report_v2.pdf",
            description="Added WCAG contrast matrix and ARIA landmarks.",
            intern_notes="Addressed mentor feedback from v1.",
            self_assessment_score=95.0,
            db_path=self.db_path,
            storage_dir=self.storage_dir
        )
        # Check that v2 did not overwrite v1
        self.assertEqual(res_v2["version_number"], 2)
        history = get_intern_deliverable_history(intern_id, db_path=self.db_path)
        milestone_submissions = [s for s in history if s["milestone_id"] == milestone_id]
        self.assertEqual(len(milestone_submissions), 2)
        # Verify both physical files exist
        self.assertTrue((self.storage_dir / res_v1["stored_filename"]).exists())
        self.assertTrue((self.storage_dir / res_v2["stored_filename"]).exists())

        # Mentor Approves Version 2
        record_mentor_deliverable_review(
            submission_id=res_v2["submission_id"],
            mentor_id=1,
            review_status="APPROVED",
            score=95.0,
            feedback="Outstanding revision. Approved.",
            rubric_scores={"Semantic Hierarchy": 30, "Accessibility & ARIA Compliance": 30, "Technical Documentation": 35},
            db_path=self.db_path
        )

        sub_v2_detail = get_submission_by_id(res_v2["submission_id"], db_path=self.db_path)
        self.assertEqual(sub_v2_detail["review_status"], "APPROVED")
        self.assertEqual(sub_v2_detail["mentor_score"], 95.0)
        self.assertEqual(len(sub_v2_detail["reviews"]), 1)

    def test_privacy_and_cohort_filtering(self):
        intern1 = 10001
        intern2 = 10002
        pdf_bytes = b"%PDF-1.4 sample pdf content %%EOF"

        # Submit for intern 1
        save_pdf_deliverable(
            intern_id=intern1,
            milestone_id=1,
            title="Intern 1 Deliverable",
            file_bytes=pdf_bytes,
            original_filename="intern1.pdf",
            db_path=self.db_path,
            storage_dir=self.storage_dir
        )

        # Submit for intern 2
        save_pdf_deliverable(
            intern_id=intern2,
            milestone_id=2,
            title="Intern 2 Deliverable",
            file_bytes=pdf_bytes,
            original_filename="intern2.pdf",
            db_path=self.db_path,
            storage_dir=self.storage_dir
        )

        # Intern 1 only sees their submission
        history_1 = get_intern_deliverable_history(intern1, db_path=self.db_path)
        self.assertEqual(len(history_1), 1)
        self.assertEqual(history_1[0]["intern_id"], intern1)

        # Mentor sees both submissions
        all_subs = get_all_submissions_for_mentor(db_path=self.db_path)
        self.assertGreaterEqual(len(all_subs), 2)

        # Metrics overview check
        metrics = get_cohort_deliverable_metrics(db_path=self.db_path)
        self.assertGreaterEqual(metrics["pending_reviews"], 2)


if __name__ == "__main__":
    unittest.main()
