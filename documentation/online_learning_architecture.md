# SkillTrack-ML: Online Learning & Platform Architecture

## Executive Summary

SkillTrack-ML has been enhanced from a batch prediction dashboard into a closed-loop **Online-Learning Internship Intelligence Platform**.

The architecture combines:
1. A **Synthetic Historical Baseline** (5,000 bootstrap records across 500 interns over 10 weeks) for model cold-start and baseline benchmarking.
2. A **Dual-Path Online Learning Dataset** distinguishing immediate event-level supervised learning from multi-week forecasting.
3. An **Incremental Online Learner** (`SGDRegressor` via `partial_fit()`) running concurrently alongside the validated **Batch Production Forecasting Baseline** (`Balanced Logistic Regression + Ridge Regression`).
4. An **Adaptive Learning Feedback Loop** connecting ML-predicted skill gaps to curated educational resources (Java, OOP, Collections, Concurrency, etc.).
5. A **Private, Persistent Intern-to-Mentor Communication System** that isolates conversation streams and logs non-content engagement metadata.

---

## 1. Dual Online Learning Data Architecture

The platform strictly avoids the fatal flaw of fabricating future ground-truth labels. Forecasting future weeks ($W \to W+1$) inherently involves future uncertainty, whereas immediate assessment activities yield immediate deterministic outcomes.

```
                    ┌───────────────────────────────────┐
                    │      Intern Application Activity  │
                    │   (Quizzes, Code, Tasks, Chat)    │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                    ┌───────────────────────────────────┐
                    │   Persistent ML Ingestion Layer   │
                    │         (ml_observations)         │
                    └─────────┬───────────────────────┬─┘
                              │                       │
           Immediate Outcome  │                       │ Next-Week Forecast
           Known Immediately  │                       │ (W+1 Unknown)
                              ▼                       ▼
            ┌───────────────────────────┐   ┌───────────────────────────┐
            │   Immediate Supervised    │   │    Pending Forecasting    │
            │          Dataset          │   │          Dataset          │
            │  label_status:            │   │  label_status:            │
            │    IMMEDIATE_LABELED      │   │    PENDING_FUTURE_LABEL   │
            │  training_eligible: 1     │   │  training_eligible: 0     │
            │  outcome_value: Observed  │   │  outcome_value: NULL      │
            └─────────────┬─────────────┘   └─────────────┬─────────────┘
                          │                               │
                          ▼                               ▼
            ┌───────────────────────────┐   ┌───────────────────────────┐
            │  Online Learner (SGD)     │   │   Wait for Week W+1       │
            │       partial_fit()       │   │      Actual Arrival       │
            └───────────────────────────┘   └─────────────┬─────────────┘
                                                          │ When W+1 Arrives
                                                          ▼
                                            ┌───────────────────────────┐
                                            │    Automated Resolution   │
                                            │  label_status: LABELED    │
                                            │  training_eligible: 1     │
                                            └─────────────┬─────────────┘
                                                          │
                                                          ▼
                                            ┌───────────────────────────┐
                                            │   Periodic Batch Retrain  │
                                            │     (Threshold >= 50)     │
                                            └───────────────────────────┘
```

### A. Immediate Event-Level Supervised Path
When an intern completes an activity whose outcome is directly measurable at execution time:
- **Quiz Completed**: Score (0-100%) is known immediately $\to$ `outcome_value = score`, `label_status = IMMEDIATE_LABELED`, `training_eligible = 1`.
- **Code Submission Evaluated**: Test cases passed, compilation/runtime score is evaluated $\to$ `outcome_value = coding_score`, `label_status = IMMEDIATE_LABELED`, `training_eligible = 1`.
- **Mentor Evaluation**: Evaluated rubrics scored $\to$ `outcome_value = mentor_score`, `label_status = IMMEDIATE_LABELED`, `training_eligible = 1`.

These rows immediately trigger `partial_fit()` on the online incremental learner, adapting model parameters in milliseconds.

### B. Next-Week Performance Forecasting Path ($W \to W+1$)
- When an intern accesses the portal during Week $W$, the ML engine predicts Week $W+1$ performance and skill gap.
- An observation is recorded in `ml_observations` with `event_type = 'weekly_forecast'`.
- The target $W+1$ outcome is **strictly unknown**. The row is saved with `outcome_value = NULL`, `label_status = PENDING_FUTURE_LABEL`, and `training_eligible = 0`.
- **No ground-truth labels are fabricated.** Predictions are never copied into the target column.
- When Week $W+1$ performance data is later recorded, the automated resolution service attaches the true observed performance score, promotes the status to `LABELED`, and sets `training_eligible = 1`.

---

## 2. Machine Learning Model Architecture: Dual Operational Modes

| Dimension | Batch Production Baseline | Online Incremental Learner |
|---|---|---|
| **Role** | Primary production forecast | Real-time adaptive learner |
| **Estimator Types** | Balanced Logistic Regression & Ridge Regression | `SGDRegressor` (Adaptive LR, L2 Penalty) |
| **Training Paradigm** | Full batch retraining with stratified validation | Incremental streaming updates via `partial_fit()` |
| **Update Cadence** | Controlled threshold gate ($\ge 50$ new labeled samples) | Per immediate event outcome |
| **Validation Gate** | Mandatory quality gate (F1 $\ge$ 0.70, Recall $\ge$ 0.85) | Step loss monitoring & offline benchmark calibration |
| **Artifact Path** | `models/production_model.pkl` | `models/online_event_model.pkl` |
| **Model Registry** | Versioned in `model_registry` table | Versioned in `online_learning_updates` table |

---

## 3. Closed-Loop Adaptive Learning Cycle

```
Assessment / Activity
         ↓
ML Predicts Skill Gap (e.g., "Developing", Collections/Streams weakness)
         ↓
Deterministic Recommendation of Curated Resources (Java Collections, Functional Streams)
         ↓
Intern Studies Curated Videos / Authoritative Documentation
         ↓
Intern Attempts Hands-on Task / Quiz
         ↓
New Live Observation Ingested (Immediate Supervised Example)
         ↓
Online Learner Updates (partial_fit) + Batch Features React
         ↓
Recalculated Forecast Shows Performance Gain
         ↓
Mentor Observes Trajectory & Provides 1-on-1 Chat Guidance
```

---

## 4. Curated Learning Resources

The system maintains 22+ verified, reputable educational resources in the `resources` table covering:
- **Java Basics**: Syntax, variables, control flow, loops.
- **Object-Oriented Programming (OOP)**: Encapsulation, inheritance, polymorphism, abstraction, interface design.
- **Java Collections Framework**: `List`, `Set`, `Map`, `Queue`, collision handling in `HashMap`.
- **Java Streams & Functional Programming**: Functional interfaces, lambda expressions, stream pipelines.
- **Multithreading & Concurrency**: Thread life cycle, synchronization, thread pools, `ExecutorService`.
- **Exception Handling & Robustness**: Checked vs unchecked exceptions, `try-with-resources`.
- **JDBC & Persistence**: Database connectivity, prepared statements, transaction management.
- **Data Structures & Algorithms**: Arrays, linked lists, binary trees, sorting, searching.

Every resource link has been verified for educational authority (Oracle Official Documentation, freeCodeCamp, etc.) and is categorized by language, difficulty, and format (video vs documentation).

---

## 5. Intern-to-Mentor Communication & Privacy

- **Database Table**: `chat_messages` (`message_id`, `sender_user_id`, `receiver_user_id`, `intern_id`, `mentor_id`, `message`, `timestamp`, `is_read`).
- **Persistence**: Messages are stored in SQLite and survive application restarts.
- **Read State**: Unread counts are dynamically tracked and cleared upon opening conversation.
- **Strict Privacy**:
  - Interns can only view conversations matching their authenticated `intern_id`.
  - Mentors can only view conversations with their assigned cohort interns.
  - Cross-intern inspection is blocked by security assertions (`PermissionError`).
- **ML Privacy Boundary**:
  - Message text is strictly private and **never used as an ML feature**.
  - Only legitimate metadata (message counts, timestamp intervals) can contribute to non-sensitive engagement metrics.

---

## 6. Verification and Proof Table

| Requirement | Implementation Mechanism | Status |
|---|---|---|
| Live Data Ingestion | `ingest_observation()` in `src/data/live_data_ingestion.py` | Verified (15+ live rows ingested) |
| Immediate Supervised Labels | `label_status = 'IMMEDIATE_LABELED'`, `training_eligible = 1` | Verified (13 immediate rows) |
| Zero-Leakage Pending Forecasts | `label_status = 'PENDING_FUTURE_LABEL'`, `training_eligible = 0` | Verified (0 future labels fabricated) |
| Incremental Online Learning | `SGDRegressor.partial_fit()` in `src/ml/online_learner.py` | Verified (adaptive learning active) |
| Batch Baseline Protection | 50-sample threshold quality gate in `train_models.py` | Verified (production baseline intact) |
| Temporal Leakage Tests | Strict cutoff time comparisons in `test_temporal_leakage.py` | Passed (4/4 tests) |
| Chat Messaging & Privacy | `chat_service.py` with role and user permission checks | Passed (3/3 tests) |
| Curated Educational Resources | `resource_service.py` with 22 curated Java assets | Passed |
