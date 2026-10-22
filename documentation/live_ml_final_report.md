# SkillTrack-ML: Live ML System Upgrade & Architectural Report

**Platform:** SkillTrack-ML (Frontend Engineering Internship Monitoring & Predictive Forecasting)  
**Version:** 2.0.0 (Live ML & Continuous Ingestion Upgrade)  
**Authors/Supervision:** Engineering Internship Evaluation Platform  
**Date:** 2026-10-07  

---

## 1. Original Architecture

In its initial version, SkillTrack-ML was a static, offline machine learning demonstration:
* **Static Synthetic Cohort:** An unvarying CSV file containing 500 synthetic interns across 10 weeks (5,000 records).
* **Isolated Database Tables:** SQLite stored intern profiles, tasks, and historical performance, but user activities (`submissions`, `mcq_attempts`) were completely isolated from the performance tracking table (`intern_performance`).
* **Static Feature Extraction:** `predict.py` queried historical performance records up to Week 10, but user actions in the application never updated the feature vector.
* **No Provenance or Ingestion Layer:** The system had no mechanism to distinguish synthetic baseline records from real-world user activity, no dataset versioning, and no model registry.
* **No Retraining or Drift Monitoring:** Once models were trained offline, they could not evolve with newly submitted data.

---

## 2. New Target Architecture

The upgraded platform implements a **hybrid, continuously updated machine learning system**:

```text
                    ┌─────────────────────────────┐
                    │ Existing Synthetic Dataset  │
                    │ 500 interns × 10 weeks      │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ Historical Baseline Layer   │
                    └──────────────┬──────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
       ▼                           ▼                           ▼
Application Events          Current Intern Data       External Context
Tasks / Quizzes / Code       submissions / scores      optional live signals
Activity frequency           current week metrics      e.g. calendar/context
       │                           │                           │
       └───────────────────────────┼───────────────────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │ SQLite Event / Observation  │
                    │ Store (ml_observations)     │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │ Unified ML Dataset          │
                    │ Historical + Current Data   │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │ Feature Engineering         │
                    │ Strict temporal cutoff t<=T │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │ ML Training / Validation    │
                    │ Grouped + Temporal Split    │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │ Model Registry & Gates      │
                    │ Promoted Production Models  │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │ Real-Time Prediction        │
                    │ Dynamic Feature Update      │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │ Intern / Mentor Dashboard   │
                    │ Live ML Status & Feedback   │
                    └─────────────────────────────┘
```

---

## 3. Data Sources & Provenance

The system enforces rigorous, academic data provenance tracking:

1. **Synthetic Historical Baseline:**
   * Source identifier: `source = 'synthetic'`
   * File: `data/skilltrack_frontend_interns_500x10_29cols.csv`
   * Size: 500 interns $\times$ 10 weekly records = 5,000 rows.
   * Role: Provides initial bootstrap distributions for cold-start model training and baseline feature extraction.
2. **Current Application-Generated Data:**
   * Source identifiers: `coding_submission`, `quiz`, `mentor_evaluation`, `application`
   * Ingestion: Collected on-the-fly when users submit coding challenges, attempt MCQs, or receive mentor reviews.
   * Persistence: Saved in `ml_observations` with exact ISO timestamps, intern ID, week, metric value, and payload.
   * Dynamic Impact: Synchronizes into the intern's active week in `intern_performance`, updating features and predictions immediately.
3. **External Contextual Signals (Extensible):**
   * Source identifier: `source = 'external'`
   * Architecture: `ExternalDataProvider` abstract interface and `AcademicCalendarContextProvider`.
   * Role: Provides calendar sprint phase and semester milestones without fabricated internet APIs.

> [!NOTE]
> **Academic Transparency Policy:**  
> The system explicitly declares that historical baseline data is synthetic and used to bootstrap the models. Real-world user interactions continuously accumulate into the unified dataset, allowing the platform to progressively transition toward empirical, application-derived data.

---

## 4. Database Changes

All database extensions were implemented non-destructively via safe schema migrations (`src/database/migrate_live_ml.py` and `database.py`):

1. **`ml_observations`**:
   * Stores real-time application events: `observation_id`, `intern_id`, `week`, `timestamp`, `source`, `event_type`, `metric_name`, `metric_value`, `payload_json`, `created_at`.
   * Indexed by `(intern_id, week)`, `timestamp`, and `source` for sub-millisecond temporal queries.
2. **`ml_dataset_versions`**:
   * Tracks dataset build lineage: `dataset_version_id`, `version_tag`, `created_at`, `source_composition` (JSON), `total_row_count`, `synthetic_row_count`, `application_row_count`, `feature_count`, `training_eligible_rows`, `artifact_path`.
3. **`model_registry`**:
   * Governs model lifecycle: `model_id`, `model_version`, `model_type`, `algorithm_cls`, `algorithm_reg`, `dataset_version`, `trained_at`, `validation_metrics` (JSON), `test_metrics` (JSON), `feature_version`, `artifact_path_cls`, `artifact_path_reg`, `status`, `promotion_reason`.

Integrity Check: `PRAGMA foreign_key_check;` returned **0 violations**.

---

## 5. Feature Engineering

* **Total Feature Vector:** Exactly 43 engineered features matching between offline training and real-time prediction:
  * **Categorical (3):** `role`, `experience_level`, `skill`
  * **Current-Week Snapshot (24):** `week`, `days_since_start`, `learning_hours`, `videos_completed`, `guides_completed`, `tasks_completed`, `engagement_rate`, `coding_attempts`, `execution_count`, `average_time_minutes`, `compile_error_rate`, `runtime_error_rate`, `test_cases_total`, `test_cases_passed`, `coding_score`, `mcq_attempts`, `mcq_score`, `debugging_score`, `code_review_score`, `previous_skill_score`, `skill_score`, `score_improvement`, `final_performance_score`, `test_case_pass_rate`
  * **Cumulative Metrics (6):** `cumulative_learning_hours`, `cumulative_tasks_completed`, `cumulative_videos_completed`, `cumulative_guides_completed`, `cumulative_coding_attempts`, `cumulative_execution_count`
  * **Historical Expanding Averages (10):** Expanding means of scores, engagement, errors, and time across all prior weeks ($w \le W$).

---

## 6. Temporal Leakage Prevention

Temporal integrity is enforced through four strict architectural safeguards:
1. **SQL Timestamp & Week Cutoff:** All feature queries strictly execute `WHERE intern_id = ? AND week <= origin_week` (and timestamp $\le T$). Information from future weeks ($w > W$) cannot enter the query result.
2. **Shifted Target Alignment:** Feature predictors at Week $W$ are aligned with outcome targets at Week $W+1$ (`shift(-1)`). The target itself is never present in the input feature matrix.
3. **Exclusion of Ground Truth Proxies:** Target columns and concurrent evaluation labels are strictly excluded from the feature matrix.
4. **Automated Unit Testing:** `tests/test_temporal_leakage.py` runs 4 formal tests verifying that introducing future activity at Week 10 causes zero change in earlier Week 3/4 forecasts. **Status: PASS**.

---

## 7. Machine Learning Models

The pipeline evaluates candidate algorithms using standardized preprocessing:
* **Classification Candidates:** Logistic Regression (Balanced), Random Forest Classifier, Gradient Boosting Classifier.
* **Regression Candidates:** Ridge Linear Regression, Random Forest Regressor, Gradient Boosting Regressor.
* **Model Selection:** Based strictly on validation set metrics (never the test set).
* **Selected Models:**
  * **Classifier:** Balanced Multinomial Logistic Regression (`models/skill_gap_model.pkl`).
  * **Regressor:** Ridge Linear Regression ($\alpha=1.0$) (`models/performance_model.pkl`).

---

## 8. Evaluation Metrics

### Split Strategy
* **Grouped Split by `intern_id`:** 350 Train (70%), 75 Validation (15%), 75 Test (15%).
* **Zero Group Leakage:** Verified 0 intern overlap across all partitions.

### Quantitative Results

| Task | Model | Validation Metric | Test Set Evaluation (Untouched) |
| :--- | :--- | :--- | :--- |
| **Skill Gap Classification** | Balanced Logistic Regression | Accuracy: 81.19%<br>Macro F1: **0.7297**<br>NI Recall: **87.80%** | Accuracy: **80.15%**<br>Macro F1: **0.6850**<br>NI Recall: **80.77%** |
| **Performance Regression** | Ridge Regression ($\alpha=1.0$) | MAE: 6.0552<br>RMSE: **7.5577**<br>$R^2$: **0.6320** | MAE: **5.9743**<br>RMSE: **7.6347**<br>$R^2$: **0.6525** |

---

## 9. Live Data Ingestion Layer

Implemented in `src/data/live_data_ingestion.py`:
* **Validation:** Verifies intern existence in SQLite and checks numeric ranges (scores $\in [0, 100]$, hours $\in [0, 24]$, error rates $\in [0, 1]$).
* **Timestamps:** Attaches ISO 8601 timestamps (`YYYY-MM-DDTHH:MM:SS`).
* **Deduplication:** Suppresses duplicate event submissions within a 2-second window.
* **Dynamic Performance Synchronization:** Synchronizes user events to `intern_performance`, updating current-week attempts, scores, and composite skill metrics in real time.

---

## 10. Dataset Growth Mechanism

Implemented in `src/ml/build_unified_dataset.py`:
* Reads historical baseline records joined with active application observations.
* Computes cumulative features and expanding averages across the unified time series.
* Creates shifted prediction targets ($W \rightarrow W+1$).
* Generates versioned artifacts in `data/processed/` and records version metadata in `ml_dataset_versions`.
* Reproducible on-demand with a single command:
  ```bash
  python src/ml/build_unified_dataset.py
  ```

---

## 11. Retraining Mechanism & Guardrails

* **Controlled Retraining Rule:** Live application events do **NOT** trigger immediate model retraining. Immediate retraining on every user click causes compute thrashing, instability, and overfitting.
* **Threshold Enforcement:** Retraining eligibility requires at least `MIN_NEW_LABELED_SAMPLES_FOR_RETRAINING = 50` newly accumulated labeled samples.
* **Verification:** `check_retraining_eligibility()` dynamically confirms whether retraining is permitted.

---

## 12. Model Versioning & Registry

* **Governance Table:** `model_registry` records every trained model artifact.
* **Versioned Storage:** Artifacts are archived under `models/registry/<version>/` alongside production pointers `models/skill_gap_model.pkl` and `models/performance_model.pkl`.
* **Metadata Tracked:** Algorithm types, training timestamp, dataset version, validation metrics, test metrics, feature version, artifact path, and promotion status.

---

## 13. Model Quality Gates

Implemented in `src/ml/model_registry.py`:
* Candidate models are never automatically promoted.
* **Gate Criteria:**
  1. Classification Macro F1 $\ge 0.65$
  2. Needs Improvement Recall $\ge 0.70$ (mandatory safety floor for early intervention)
  3. Regression RMSE $\le 10.0$
  4. Regression $R^2 \ge 0.50$
* **Outcome:** If criteria are satisfied, model status is set to `production` and previous models are set to `archived`. Otherwise, marked `rejected`.

---

## 14. Data Drift Monitoring

Implemented in `src/ml/drift_monitor.py`:
* Compares historical baseline records against incoming cohort observations.
* Computes mean shift, standard deviation shift, Z-scores, and categorical class distribution shift for 7 key performance metrics.
* Assigns an overall drift rating: `LOW`, `MODERATE`, or `HIGH`.
* Real-time metrics are displayed in the Mentor Portal.

---

## 15. Explainability & Transparent Guidance

* **Explainability:** Displays top influential model factors (skill trajectory, test pass rate, debugging score, learning hours, error frequency).
* **Transparent Terminology:** Uses "Forecast Confidence" (derived from class probabilities via `predict_proba()`), avoiding misleading claims of causal certainty.
* **Domain Recommendations:** Dynamically contextualizes advice to the intern's active technology stack (React, JavaScript, CSS, HTML, API Integration).

---

---

## 16. Online Learning & Platform Upgrades

* **Dual-Path Dataset Ingestion:** Distinguishes immediate event-level supervised rows (`IMMEDIATE_LABELED`, `training_eligible = 1`) from multi-week forecasting samples (`PENDING_FUTURE_LABEL`, `training_eligible = 0`).
* **Online Incremental Learner:** Operates `SGDRegressor` via `partial_fit()`, adapting model parameters in milliseconds per immediately labeled event without touching the batch production baseline.
* **Curated Learning Resources:** 22 verified Java, OOP, Collections, Concurrency, and persistence resources in SQLite, deterministically recommended based on detected skill gaps.
* **Intern-to-Mentor Chat:** Secure, persistent messaging in SQLite (`chat_messages`) with strict access control and unread tracking. Private message text is quarantined from ML features.

---

## 17. Comprehensive Test Suite

| Test Suite | File | Tests Run | Result |
| :--- | :--- | :--- | :--- |
| Temporal Leakage | `tests/test_temporal_leakage.py` | 4 | **PASS** |
| Live ML Pipeline Integration | `tests/test_live_ml_pipeline.py` | 6 | **PASS** |
| Online Learning, Resources & Chat | `tests/test_online_learning_and_chat.py` | 7 | **PASS** |
| End-to-End User Journey | `scratch/verify_e2e_journey.py` | 10 stages | **PASS** |

---

## 18. Limitations & Academic Transparency

* **Synthetic Bootstrap:** The historical 5,000 records are synthetic and serve as the initial training bootstrap. Live observations continuously accumulate into the unified layer.
* **Forecasting Delay:** Next-week forecasting records ($W \to W+1$) inherently require waiting for Week $W+1$ outcomes to mature before becoming training-eligible. Ground-truth labels are never fabricated.
* **Online Learner Scope:** The online learner currently models event-level continuous performance using lightweight linear stochastic approximation; full batch model promotion requires the 50-sample threshold and validation quality gate.
* **Chat Feature Scope:** Chat message text is not analyzed with NLP for ML predictions to respect intern privacy. Only non-sensitive engagement metadata (message counts, interaction intervals) is utilized.
