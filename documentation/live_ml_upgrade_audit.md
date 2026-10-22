# SkillTrack-ML: Live ML Architecture & System Audit Report

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-07  
**Scope:** Pre-upgrade architectural and code inspection across all components (Streamlit, SQLite, ML Pipeline, Data Processing, and Feature Engineering).

---

## 1. Executive Summary

SkillTrack-ML is an operational, web-based engineering internship portal and machine learning platform designed to monitor frontend engineering interns, evaluate coding tasks and assessments, and deliver predictive forecasts of skill gaps and performance scores for upcoming weeks.

The primary objective of this upgrade is to transform the existing static synthetic dataset pipeline into a **continuously updated, live ML system**:
* The original 500-intern synthetic cohort serves as the **historical baseline layer**.
* Live application interactions (coding submissions, test completions, quiz attempts, and mentor reviews) are recorded in SQLite as persistent real-world observations with strict data provenance.
* Live observations update the intern's current-week feature vector and predictions on-the-fly.
* When sufficient labeled transitions accumulate, the observations become part of the unified dataset eligible for controlled model retraining and promotion.

This audit establishes the baseline system state prior to any schema modifications or code extensions.

---

## 2. Current Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                            │
│  Streamlit Application (app/app.py)                                    │
│   ├── Intern Portal (Dashboard, Tasks, Quizzes, AI Skill Outlook Card) │
│   └── Mentor Portal (Cohort Table, Risk Flags, Analytics, Evaluations) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                        Reads & Writes Directly
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       PERSISTENCE LAYER (SQLite)                       │
│  data/database/skilltrack.db                                           │
│   ├── users (Authentication: intern1, mentor1)                         │
│   ├── interns (500 profiles: 10001 to 10500)                           │
│   ├── intern_performance (5,000 historical rows: 10 weeks per intern)  │
│   ├── coding_tasks (10 frontend coding challenges)                     │
│   ├── submissions (Intern code submissions & timestamps)               │
│   ├── mcq_questions & mcq_attempts (Quizzes & scores)                  │
│   └── mentor_evaluations (Mentor reviews & retention decisions)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                          Read-Only Queries
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     INFERENCE LAYER (src/ml/predict.py)                │
│   1. build_feature_vector(intern_id, prediction_week)                  │
│      - Strict cutoff: SELECT * FROM intern_performance WHERE week <= W │
│      - Extracts 43 features (numeric, categorical, expanding averages) │
│   2. Dual Model Inference:                                             │
│      - Balanced Logistic Regression (models/skill_gap_model.pkl)       │
│      - Ridge Regression (models/performance_model.pkl)                 │
│   3. Returns Forecast: Predicted Gap, Confidence %, Score, Recs        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Current Data Flow

1. **Initial Seeding:**
   * Raw benchmark file: `data/skilltrack_frontend_interns_500x10_29cols.csv` (5,000 rows × 29 columns).
   * Seeded into SQLite via `src/database/seed_frontend_intern_data.py`:
     * 500 rows inserted into `interns`.
     * 5,000 rows inserted into `intern_performance` (Weeks 1 to 10).
   * Tasks and credentials seeded via `seed_frontend_tasks.py` and `seed_users.py`.

2. **Supervised Dataset Preparation (`src/ml/prepare_ml_dataset.py`):**
   * Reads raw CSV file.
   * Calculates historical metrics (`test_case_pass_rate`, cumulative sums, expanding averages).
   * Shifts target forward by 1 week (`target_skill_gap = shift(-1)`).
   * Drops Week 10 (no Week 11 target), resulting in **4,500 supervised training instances × 50 columns** saved to `data/processed/intern_ml_dataset.csv`.

3. **Offline Training (`src/ml/train_models.py`):**
   * Splits dataset by `intern_id` (70% Train = 350 interns, 15% Val = 75 interns, 15% Test = 75 interns).
   * Trains Balanced Logistic Regression (selected for Macro F1: 0.7277, Needs Improvement Recall: 87.80%).
   * Trains Ridge Regression (selected for RMSE: 7.5577, $R^2$: 0.6320).
   * Exports models to `models/skill_gap_model.pkl` and `models/performance_model.pkl`.

4. **Runtime User Interaction (`app/app.py`):**
   * Intern logs in $\rightarrow$ `get_intern_ml_forecast(intern_id)` runs `build_feature_vector(intern_id)`.
   * Queries `intern_performance` up to max recorded week (Week 10).
   * Returns AI forecast to dashboard.
   * When an intern runs or submits code, or completes an MCQ:
     * Row is inserted into `submissions` or `mcq_attempts`.
     * **Critical architectural disconnect:** `submissions` and `mcq_attempts` are disconnected from `intern_performance` and `build_feature_vector`. As a result, new runtime activities do not modify the feature vector, do not alter predictions, and do not feed back into the ML dataset.

---

## 4. Current Database Schema

The active database is SQLite3 at `data/database/skilltrack.db`.

| Table | Primary Key | Key Columns | Current Count | Role / Status |
| :--- | :--- | :--- | :--- | :--- |
| `users` | `user_id` | `username`, `password`, `role`, `intern_id`, `mentor_id` | 2 | Authentication (`intern1`, `mentor1`) |
| `interns` | `intern_id` | `name`, `email`, `role`, `experience_level`, `current_week`, `average_score`, `skill_score`, `final_performance_score`, `skill_gap`, `resource_action`, `progress` | 500 | Cohort intern profiles |
| `intern_performance` | `performance_id` | `intern_id`, `week`, `days_since_start`, `skill`, `learning_hours`, `videos_completed`, `guides_completed`, `tasks_completed`, `engagement_rate`, `coding_attempts`, `execution_count`, `average_time_minutes`, `compile_error_rate`, `runtime_error_rate`, `test_cases_total`, `test_cases_passed`, `coding_score`, `mcq_attempts`, `mcq_score`, `debugging_score`, `code_review_score`, `previous_skill_score`, `skill_score`, `score_improvement`, `final_performance_score`, `skill_gap`, `resource_action` | 5,000 | Static 10-week synthetic performance records (UNIQUE on `intern_id, week`) |
| `coding_tasks` | `task_id` | `title`, `description`, `max_score` | 10 | Frontend coding problems |
| `submissions` | `submission_id` | `intern_id`, `task_id`, `score`, `status`, `submitted_at`, `language`, `code`, `output` | 1 | Live code submission logs |
| `mcq_questions` | `question_id` | `question`, options A-D, `correct_option`, `topic` | 0 | (MCQs embedded directly in app.py) |
| `mcq_attempts` | `attempt_id` | `intern_id`, `score`, `total_questions`, `topic`, `attempted_at` | 5 | Live quiz results |
| `resources` | `resource_id` | `title`, `resource_type`, `url`, `topic`, `difficulty` | 0 | Curated links |
| `mentor_evaluations` | `evaluation_id` | `intern_id`, `mentor_id`, scores, `decision`, `remarks`, `evaluated_at` | 2 | Mentor review records |

Integrity Check: `PRAGMA foreign_key_check;` returns **0 violations**.

---

## 5. Current ML Pipeline

* **Input Data:** `data/processed/intern_ml_dataset.csv` (4,500 samples).
* **Target 1 (Classification):** `target_skill_gap` (`Strong`, `Developing`, `Needs Improvement`).
* **Target 2 (Regression):** `target_performance_score` (continuous 0–100 score).
* **Pipeline Structure:**
  * Preprocessor: `ColumnTransformer` with `StandardScaler` on numerical features and `OneHotEncoder(handle_unknown='ignore')` on categorical features (`role`, `experience_level`, `skill`).
  * Classifier: `LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)`.
  * Regressor: `Ridge(alpha=1.0)`.
* **Splitting Strategy:** Grouped by `intern_id` using fixed random seed (42):
  * Train: 350 interns (3,150 samples)
  * Validation: 75 interns (675 samples)
  * Test: 75 interns (675 samples)
* **Current Evaluation Metrics:**
  * Classification:
    * Validation Accuracy: 80.89% | Validation Macro F1: 0.7277 | Needs Improvement Recall: 87.80%
    * Test Accuracy: 80.30% | Test Macro F1: 0.7188 | Test Needs Improvement Recall: 80.77%
  * Regression:
    * Validation MAE: 6.0552 | Validation RMSE: 7.5577 | Validation $R^2$: 0.6320
    * Test MAE: 6.0712 | Test RMSE: 7.6347 | Test $R^2$: 0.6525

---

## 6. Current Feature Pipeline

Extracted on-the-fly in `src/ml/predict.py:build_feature_vector()`:
1. **Categorical (3):** `role`, `experience_level`, `skill`
2. **Current-Week Snapshot (24):** `week`, `days_since_start`, `learning_hours`, `videos_completed`, `guides_completed`, `tasks_completed`, `engagement_rate`, `coding_attempts`, `execution_count`, `average_time_minutes`, `compile_error_rate`, `runtime_error_rate`, `test_cases_total`, `test_cases_passed`, `coding_score`, `mcq_attempts`, `mcq_score`, `debugging_score`, `code_review_score`, `previous_skill_score`, `skill_score`, `score_improvement`, `final_performance_score`, `test_case_pass_rate`
3. **Cumulative Totals (6):** `cumulative_learning_hours`, `cumulative_tasks_completed`, `cumulative_videos_completed`, `cumulative_guides_completed`, `cumulative_coding_attempts`, `cumulative_execution_count`
4. **Historical Expanding Averages (10):** `historical_coding_score_avg`, `historical_mcq_score_avg`, `historical_debugging_score_avg`, `historical_code_review_score_avg`, `historical_skill_score_avg`, `historical_learning_hours_avg`, `historical_engagement_rate_avg`, `historical_compile_error_rate_avg`, `historical_runtime_error_rate_avg`, `historical_average_time_minutes_avg`

**Total Feature Count:** 43 features.

---

## 7. Current Prediction Pipeline

* Function: `get_intern_ml_forecast(intern_id, prediction_week=None)`
* Steps:
  1. Loads cached model pipelines from `models/skill_gap_model.pkl` and `models/performance_model.pkl`.
  2. Resolves `prediction_week` (defaults to latest recorded week for intern, i.e., Week 10).
  3. Builds 43-column feature DataFrame from SQLite query constrained to `week <= prediction_week`.
  4. Generates predictions:
     * Classification: class label + class probabilities via `predict_proba()`.
     * Regression: bounded between 0 and 100.
  5. Maps `predicted_skill_gap` and `current_skill` to domain-specific recommendations and action tag.
  6. Attaches top 5 influential factors for model explainability.

---

## 8. Existing Leakage Protections

1. **Temporal Cutoff:** The SQL query in `build_feature_vector` explicitly enforces `WHERE intern_id = ? AND week <= ?`. No records from future weeks can enter the feature vector.
2. **Shifted Target Alignment:** In `prepare_ml_dataset.py`, the target is shifted forward by 1 week (`df.groupby('intern_id')['skill_gap'].shift(-1)`). Features at week $W$ are strictly mapped to outcomes at week $W+1$.
3. **Exclusion of Target Proxies:** Ground-truth target columns (`target_skill_gap`, `target_performance_score`, `target_resource_action`, `target_skill_score`) and current-week `skill_gap` / `resource_action` are excluded from the training feature set.
4. **Grouped Intern Splitting:** Interns are partitioned strictly by `intern_id`. All 10 weekly records for any single intern reside entirely in Train, Validation, or Test. No intern's data leaks across splits.

---

## 9. Existing Strengths

* Complete, beautiful end-to-end working UI for both intern and mentor personas.
* Clean separation of database seeding, feature preparation, model training, and inference modules.
* Statistically sound baseline models (balanced class weights for high minority class recall on "Needs Improvement").
* Transparent explainability without causal exaggeration.
* Zero foreign key violations and strict schema consistency.

---

## 10. Existing Weaknesses & Gaps Identified

1. **Static Dataset Isolation:** Live user events (code submissions, quiz attempts, mentor evaluations) are inserted into isolated tables (`submissions`, `mcq_attempts`) and **never enter `intern_performance` or the ML dataset**.
2. **Missing Observation Layer & Data Provenance:** The database lacks an event-driven observation store (`ml_observations`) tracking event timestamps, sources (`synthetic`, `application`, `mentor_evaluation`, `external`), and feature deltas.
3. **Pickle Environment Deserialization Mismatch:** The current `.pkl` files were serialized under a newer scikit-learn version (1.9.0), causing an `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'` when unpickled in the project virtual environment (`scikit-learn==1.7.2`). Retraining under the `.venv` environment resolves this immediately.
4. **No Continuous Feature Aggregator:** When a user submits a task or attempts a quiz in Week 10, the current-week performance record is not dynamically recalculated from event aggregations.
5. **No Model Registry or Dataset Versioning:** Retrained models overwrite `skill_gap_model.pkl` in-place without versioning, promotion gates, or evaluation lineage.
6. **No Data Drift Monitoring:** As new application events enter the system, there is no mechanism tracking distribution divergence between synthetic baselines and live observations.
7. **No Automated Temporal Leakage Unit Test:** While temporal logic is present in SQL, there is no formal test suite guaranteeing zero forward-looking leakage.

---

## 11. What Must Remain Unchanged

* Existing Streamlit UI structure, routes, layouts, and role-based views.
* Demo accounts: `intern1 / intern123` and `mentor1 / mentor123`.
* Existing SQLite tables: `users`, `interns`, `intern_performance`, `coding_tasks`, `submissions`, `mcq_attempts`, `mentors`, `mentor_evaluations`.
* The 500 synthetic interns and their 5,000 historical weekly records.
* The 43-feature mathematical definition and W $\rightarrow$ W+1 temporal horizon.
* Baseline models: Balanced Logistic Regression and Ridge Regression.
* Existing backup files and notebooks.

---

## 12. What Will Be Added

1. **Database Migration:**
   * `ml_observations`: Event store recording every user action with `source` (`synthetic`, `application`, `quiz`, `coding_submission`, `mentor_evaluation`), `event_type`, payload, and ISO timestamp.
   * `ml_dataset_versions`: Tracks unified dataset builds, row counts, synthetic vs application counts, and training eligibility.
   * `model_registry`: Tracks model versions, training timestamps, validation metrics, test metrics, and promotion status.
2. **Live Ingestion Layer (`src/data/live_data_ingestion.py`):**
   * Validates and ingests user events (code submissions, quiz attempts, mentor evaluations) into `ml_observations`.
   * Real-time aggregator: updates/enriches the intern's active week metrics in `intern_performance`.
3. **Unified Dataset Builder (`src/ml/build_unified_dataset.py`):**
   * Merges synthetic historical records with validated application observations.
   * Computes shifted targets for completed intervals.
   * Produces reproducible versioned datasets with explicit provenance metadata.
4. **Enhanced Real-Time Feature Generation (`src/ml/predict.py`):**
   * Extends `build_feature_vector` to incorporate real-time observation aggregates from `ml_observations` while preserving strict cutoff $t \le T$ (temporal safety).
5. **Model Registry & Quality Gates (`src/ml/model_registry.py` & `src/ml/train_models.py`):**
   * Supports candidate model training, evaluation on validation set, quality comparison against active production model, and gated promotion.
   * Controlled retraining: only triggered when threshold of new labeled samples is reached.
6. **Data Drift Monitoring (`src/ml/drift_monitor.py`):**
   * Computes Wasserstein distance / PSI / mean shift between synthetic baseline and live application data.
7. **Automated Leakage & Integrity Tests (`tests/test_temporal_leakage.py` & `tests/test_live_ml_pipeline.py`):**
   * Rigorous test suite validating strict temporal cutoff, duplicate prevention, and feature consistency.
8. **UI Live ML Status Components:**
   * Mentor Dashboard: Live "ML System & Data Health" card showing dataset size, synthetic vs live counts, active model version, drift status, and validation metrics.
   * Intern Dashboard: "Prediction updated from latest activity" badge when real-time events have adjusted current features.
9. **Professor Demonstration Script & Guide:**
   * Deterministic demonstration verifying live event ingestion $\rightarrow$ feature delta $\rightarrow$ prediction update $\rightarrow$ dataset expansion.

---

## 13. Migration Risks & Mitigations

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| Database Corruption during Migration | High | Create full backup `data/database/skilltrack_backup_pre_live_ml.db` before applying migration. Use non-destructive `CREATE TABLE IF NOT EXISTS` and `ALTER TABLE`. |
| Breaking UI / Streamlit Runtime | High | Preserve all existing functions in `app/app.py`. Only hook ingestion into existing `save_code_submission`, `save_mcq_attempt`, and `save_evaluation` handlers. |
| Pickling Incompatibilities | Medium | Retrain and serialize models in the active `.venv` environment (`scikit-learn==1.7.2`). |
| Uncontrolled Retraining Slowdowns | Medium | Implement strict retraining guardrails (`MIN_NEW_LABELED_SAMPLES_FOR_RETRAINING`); never retrain on UI button clicks. |
| Future Target Leakage | Critical | Enforce strict timestamp and week filtering ($t \le T$) in all feature construction functions. Validate with automated leakage tests. |

---

*Audit completed and certified. Ready to proceed to Phase 2 (Database Migration & Ingestion Layer).*
