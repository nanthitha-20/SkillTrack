"""
SkillTrack-ML: Deliverables and Project Milestones Schema Migration
Adds support for versioned PDF project submissions, milestone rubrics,
and mentor deliverable evaluations while preserving existing database structures.
"""

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"


def run_deliverables_migration(db_path: Path = DATABASE_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Project Milestones Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_milestones (
            milestone_id INTEGER PRIMARY KEY AUTOINCREMENT,
            milestone_number INTEGER UNIQUE NOT NULL,
            title TEXT NOT NULL,
            track TEXT NOT NULL,
            week INTEGER NOT NULL,
            description TEXT NOT NULL,
            deliverable_requirements TEXT,
            rubric_criteria_json TEXT,
            max_score REAL DEFAULT 100.0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Versioned Project Submissions Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_submissions (
            submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
            intern_id INTEGER NOT NULL,
            milestone_id INTEGER NOT NULL,
            version_number INTEGER NOT NULL DEFAULT 1,
            title TEXT NOT NULL,
            description TEXT,
            intern_notes TEXT,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            file_hash TEXT,
            self_assessment_score REAL,
            self_assessment_notes TEXT,
            review_status TEXT NOT NULL DEFAULT 'SUBMITTED'
                CHECK(review_status IN ('SUBMITTED', 'UNDER_REVIEW', 'CHANGES_REQUESTED', 'APPROVED')),
            mentor_score REAL,
            mentor_feedback TEXT,
            reviewed_by INTEGER,
            reviewed_at TEXT,
            submitted_at TEXT NOT NULL,
            FOREIGN KEY (intern_id) REFERENCES interns(intern_id),
            FOREIGN KEY (milestone_id) REFERENCES project_milestones(milestone_id),
            FOREIGN KEY (reviewed_by) REFERENCES mentors(mentor_id),
            UNIQUE(intern_id, milestone_id, version_number)
        )
    """)

    # 3. Submission Review History Table (audit trail of all reviews)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS submission_reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            submission_id INTEGER NOT NULL,
            mentor_id INTEGER NOT NULL,
            review_status TEXT NOT NULL CHECK(review_status IN ('UNDER_REVIEW', 'CHANGES_REQUESTED', 'APPROVED')),
            score REAL,
            rubric_breakdown_json TEXT,
            feedback TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            FOREIGN KEY (submission_id) REFERENCES project_submissions(submission_id),
            FOREIGN KEY (mentor_id) REFERENCES mentors(mentor_id)
        )
    """)

    # 4. Seed Standard 10 Milestones (Matching 10-week curriculum)
    milestones_data = [
        (1, "Milestone 1: Semantic Web Architecture & Accessibility", "Frontend Core", 1,
         "Architecture diagram and accessibility (WCAG 2.1 AA) evaluation report of the core component library.",
         "Submit a compiled PDF technical report (max 10 pages) containing semantic DOM hierarchy, ARIA roles, and contrast audits.",
         '{"Semantic Hierarchy": 30, "Accessibility & ARIA Compliance": 30, "Technical Documentation": 40}', 100.0),
        
        (2, "Milestone 2: Modular CSS Design System & Theme Engine", "Styling Systems", 2,
         "Specification document detailing custom CSS tokens, typography scales, light/dark orange color palettes, and responsive breakpoints.",
         "PDF deliverable documenting token architecture, responsive grid specifications, and component visual proof.",
         '{"Token Architecture": 30, "Responsive Grid Strategy": 35, "Code Organization": 35}', 100.0),
        
        (3, "Milestone 3: Reactive State Management & Data Flow", "JavaScript & State", 3,
         "State machine specification, unidirectional data flow architecture, and immutable state updates report.",
         "PDF deliverable covering state tree design, reducer/action dispatch flows, and side-effect handling diagrams.",
         '{"State Architecture": 35, "Predictability & Immutability": 35, "Error Resilience": 30}', 100.0),
        
        (4, "Milestone 4: RESTful API Integration & Network Resilience", "API Integration", 4,
         "API contract design, cache-first network strategies, retry backoff algorithms, and error-handling specification.",
         "PDF report detailing endpoint schemas, optimistic UI updates, payload validation, and HTTP failure recoveries.",
         '{"Contract Design": 30, "Resilience & Fallbacks": 40, "Security & Auth Headers": 30}', 100.0),
        
        (5, "Milestone 5: Client-Side Performance & Bundle Optimization", "Performance Engineering", 5,
         "Lighthouse performance audit, Web Vitals optimization plan (LCP, FID, CLS), and tree-shaking analysis.",
         "PDF audit report presenting before-and-after performance metrics, chunk-splitting graphs, and critical rendering path audit.",
         '{"Core Web Vitals": 40, "Bundle Splitting Strategy": 30, "Asset Optimization": 30}', 100.0),
        
        (6, "Milestone 6: Component Library & Reusability Audit", "UI Architecture", 6,
         "Design tokens, atomic design breakdown, props contracts, and cross-browser visual regression testing plan.",
         "PDF deliverable including component inventory, props interface documentation, and test matrix across screen sizes.",
         '{"Component Modularization": 35, "Reusability & Extensibility": 35, "Visual Consistency": 30}', 100.0),
        
        (7, "Milestone 7: Test Engineering & Quality Assurance Plan", "Testing & QA", 7,
         "Comprehensive test pyramid strategy: unit testing, integration tests, mock service workers, and end-to-end user journeys.",
         "PDF test documentation outlining coverage targets, edge-case matrices, mocking strategies, and regression test results.",
         '{"Test Pyramid Strategy": 35, "Edge-Case Coverage": 35, "Quality Reporting": 30}', 100.0),
        
        (8, "Milestone 8: Client-Side Security & Threat Modeling", "Application Security", 8,
         "OWASP Top 10 client-side risk analysis: XSS mitigation, Content Security Policies (CSP), secure storage, and CORS handling.",
         "PDF security evaluation covering sanitize algorithms, token storage vulnerabilities, and CSP header policies.",
         '{"Threat Identification": 40, "Mitigation Specifications": 35, "Compliance & Best Practices": 25}', 100.0),
        
        (9, "Milestone 9: Real-time Telemetry & ML Observability Integration", "Full-Stack Integration", 9,
         "User engagement telemetry tracking architecture, observation pipelines, and client analytics instrumentation.",
         "PDF system architecture report documenting event schema, latency profiling, and logging pipeline design.",
         '{"Telemetry Pipeline Architecture": 40, "Data Integrity & Schema": 30, "Observability": 30}', 100.0),
        
        (10, "Milestone 10: Capstone Engineering Deliverable & Production Readiness", "Capstone Project", 10,
         "Comprehensive capstone release package, deployment architecture, rollback protocols, and end-to-end audit report.",
         "Complete production-readiness dossier: architecture blueprints, disaster recovery plans, mentor sign-off rubrics.",
         '{"Production Architecture": 35, "Deployment & Rollback Strategy": 30, "Executive Presentation": 35}', 100.0),
    ]

    for item in milestones_data:
        cur.execute("""
            INSERT INTO project_milestones (
                milestone_number, title, track, week, description, deliverable_requirements, rubric_criteria_json, max_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(milestone_number) DO UPDATE SET
                title = excluded.title,
                track = excluded.track,
                week = excluded.week,
                description = excluded.description,
                deliverable_requirements = excluded.deliverable_requirements,
                rubric_criteria_json = excluded.rubric_criteria_json,
                max_score = excluded.max_score
        """, item)

    conn.commit()
    conn.close()
    print("[SkillTrack-ML] Deliverables & Milestones schema migration executed successfully.")


if __name__ == "__main__":
    run_deliverables_migration()
