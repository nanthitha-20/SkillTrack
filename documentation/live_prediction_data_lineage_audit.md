# SkillTrack-ML: Live Prediction Data Lineage & Code Path Audit

**Audit Date:** 2026-10-07  
**Scope:** Complete Python execution path verification across live events, feature engineering, prediction, online learning, and model governance.

---

## 1. Concrete Execution Trace Audit (A through H)

We inspected the actual Python implementation files (`app/app.py`, `src/data/live_data_ingestion.py`, `src/ml/predict.py`, `src/ml/online_learner.py`, `src/ml/build_unified_dataset.py`, `src/ml/train_models.py`, `src/ml/model_registry.py`). Below is the verified reality of each step:

### A. Where a live application event enters the database
* **Code Location:** [`app/app.py`](file:///C:/Users/raghu/Downloads/SkillTrack-ML/SkillTrack-ML/app/app.py)
* **Concrete Functions:**
  * **Quiz Submission:** `save_mcq_attempt(intern_id, score, num_questions, topic)` at line 273.
    * Executes: `INSERT INTO mcq_attempts (intern_id, score, total_questions, topic, completed_at) VALUES (?, ?, ?, ?, ?)`
  * **Coding Task Submission:** `save_code_submission(intern_id, task_id, score, passed, code, language, output, tc_passed, tc_total)` at line 232.
    * Executes: `INSERT INTO submissions (intern_id, task_id, score, passed, submitted_at, language, code, output) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
  * **Mentor Evaluation:** `save_evaluation(intern_id, mentor_id, score, notes, decision)` at line 342.
    * Executes: `INSERT INTO mentor_evaluations (...) VALUES (...)`
  * **Intern-Mentor Chat:** `send_chat_message(...)` in [`src/data/chat_service.py`](file:///C:/Users/raghu/Downloads/SkillTrack-ML/SkillTrack-ML/src/data/chat_service.py).
    * Executes: `INSERT INTO chat_messages (...) VALUES (...)`

### B. Where it becomes an ML observation
* **Code Location:** [`src/data/live_data_ingestion.py`](file:///C:/Users/raghu/Downloads/SkillTrack-ML/SkillTrack-ML/src/data/live_data_ingestion.py)
* **Concrete Function:** `ingest_observation(...)`
* **Trigger Point:**
  * In `app/app.py`, immediately inside `save_mcq_attempt()` (line 297), `save_code_submission()` (line 257), `save_evaluation()` (line 353), and `app/app.py` chat submission form (line 2408), `ingest_observation(...)` is invoked.
* **Database Action:**
  * Validates metric ranges (`VALID_NUMERIC_RANGES`).
  * Enforces intern foreign key validation.
  * Executes:
    ```sql
    INSERT INTO ml_observations (
        intern_id, week, timestamp, source, event_type,
        metric_name, metric_value, payload_json, created_at,
        label_status, training_eligible, outcome_value, target_week, source_record_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ```
  * Returns `obs_id`.
* **State Synchronization:**
  * Calls `_sync_observation_to_performance(...)` which updates the active week's row in `intern_performance` (coding score, mcq score, attempts, test cases passed, composite skill score).

### C. Where that observation becomes a feature vector
* **Code Location:** [`src/ml/predict.py`](file:///C:/Users/raghu/Downloads/SkillTrack-ML/SkillTrack-ML/src/ml/predict.py)
* **Concrete Function:** `build_feature_vector(intern_id, prediction_week=None, db_path=DATABASE_PATH)`
* **Call Sequence:**
  1. Queries static intern demographics from `interns` (`role`, `experience_level`).
  2. Queries historical performance from `intern_performance WHERE intern_id = ? AND week <= origin_week`.
  3. Computes:
     * Current week snapshot metrics (21 numeric variables).
     * Expanding cumulative sums (6 variables: hours, completed tasks, attempts, etc.).
     * Expanding historical averages (10 variables: `historical_coding_score_avg`, `historical_mcq_score_avg`, error rates, etc.).
     * Test case pass rate ($passed / total$).
  4. Returns a single-row `DataFrame` with exactly **43 features**.
* **Audit Finding on Gap:**
  * Previously, `build_feature_vector()` did NOT record the specific `observation_ids` contributing to that state, did NOT compute a cryptographic feature hash, and did NOT accept an explicit ISO timestamp cutoff $t \le T_{cutoff}$. We will add this traceability.

### D. Where that feature vector is passed to prediction
* **Code Location:** [`src/ml/predict.py`](file:///C:/Users/raghu/Downloads/SkillTrack-ML/SkillTrack-ML/src/ml/predict.py)
* **Concrete Function:** `get_intern_ml_forecast(intern_id, prediction_week=None, db_path=DATABASE_PATH)`
* **Call Sequence:**
  * Calls `build_feature_vector(...)` to obtain `features_df`.
  * Loads production models:
    * `cls_model = joblib.load("models/skill_gap_model.pkl")` (Balanced Logistic Regression)
    * `reg_model = joblib.load("models/performance_model.pkl")` (Ridge Regression)
  * Executes:
    * `predicted_gap = cls_model.predict(features_df)[0]`
    * `probabilities = cls_model.predict_proba(features_df)[0]`
    * `predicted_perf = reg_model.predict(features_df)[0]`

### E. Whether the resulting prediction is stored
* **Audit Finding:**
  * **NO (Prior to this upgrade).** In the earlier implementation, `get_intern_ml_forecast()` calculated predictions in memory and returned a Python dictionary to Streamlit without inserting a row into a prediction history table.
  * **Upgrade Requirement:** Create `prediction_traces` table and persist every prediction along with data cutoff timestamp, model version, observation IDs used, and feature hash.

### F. Whether the observation is actually passed to the online learner
* **Code Location:** [`src/data/live_data_ingestion.py`](file:///C:/Users/raghu/Downloads/SkillTrack-ML/SkillTrack-ML/src/data/live_data_ingestion.py)
* **Concrete Function:** `ingest_observation(...)` lines 162–173.
* **Audit Verification:**
  * When `label_status == 'IMMEDIATE_LABELED'` and `outcome_value is not None`:
    ```python
    from src.ml.online_learner import get_online_learner
    get_online_learner().update_with_event(
        intern_id=intern_id,
        outcome_value=float(outcome_value),
        event_type=event_type,
        db_path=db_path
    )
    ```
  * In `src/ml/online_learner.py`, `update_with_event()` extracts features, performs `self.model.partial_fit(X_scaled, y)` on `SGDRegressor`, increments `cumulative_samples`, saves `models/online_event_model.pkl`, and logs to `online_learning_updates`.
  * **Result:** **YES**, immediately labeled observations genuinely reach `SGDRegressor.partial_fit()`.

### G. Whether future predictions use the updated learner/model
* **Audit Verification:**
  * In `src/ml/predict.py`, `get_intern_ml_forecast()` loads:
    * `SKILL_GAP_MODEL_PATH = "models/skill_gap_model.pkl"` (Balanced Logistic Regression)
    * `PERFORMANCE_MODEL_PATH = "models/performance_model.pkl"` (Ridge Regression)
  * It does **NOT** use `models/online_event_model.pkl` (`SGDRegressor`) for production forecasts!
  * **Result:** The online learner runs in **SHADOW / EXPERIMENTAL MODE**. Future forecasts in the portal are generated by the production batch model (Ridge + Logistic Regression), which responds to live data via the **updated feature vector**, not via online model weights.

### H. Whether the current production prediction model is actually different from the online learner
* **Audit Verification:**
  * **YES, THEY ARE COMPLETELY DIFFERENT.**
    1. **Production Batch Model:**
       * Classification: Balanced Multinomial Logistic Regression (`Pipeline[StandardScaler, OneHotEncoder, LogisticRegression(class_weight='balanced')]`).
       * Regression: Ridge Linear Regression ($\alpha=1.0$).
       * Trained via full batch optimization on validated historical partitions.
    2. **Online Incremental Learner:**
       * Continuous Regressor: `SGDRegressor(loss='squared_error', penalty='l2', alpha=0.001, learning_rate='adaptive')`.
       * Trained via streaming `partial_fit()` calls on individual immediate event outcomes.
  * **Critical Distinction to Maintain:** The UI must explicitly state that the production forecast model is Ridge/Logistic Regression (updating through live feature vectors), while the online learner is a complementary shadow model operating via `partial_fit()`.

---

## 2. Summary Call Graph

```text
[User Submits Quiz in app/app.py]
  │
  ├─► save_mcq_attempt()
  │     ├─► INSERT INTO mcq_attempts
  │     └─► ingest_observation() [src/data/live_data_ingestion.py]
  │           ├─► INSERT INTO ml_observations (returns obs_id)
  │           ├─► _sync_observation_to_performance() -> UPDATE intern_performance
  │           └─► if IMMEDIATE_LABELED:
  │                 └─► OnlineEventLearner.update_with_event() [src/ml/online_learner.py]
  │                       ├─► SGDRegressor.partial_fit()
  │                       ├─► joblib.dump(online_event_model.pkl)
  │                       └─► INSERT INTO online_learning_updates
  │
[User Views Forecast in app/app.py]
  │
  └─► get_intern_ml_forecast() [src/ml/predict.py]
        ├─► build_feature_vector()
        │     └─► SELECT FROM intern_performance WHERE week <= origin_week
        │           (reflects updated scores from _sync_observation_to_performance!)
        ├─► load_models() -> Ridge & Logistic Regression
        ├─► cls_model.predict(features_df)
        ├─► reg_model.predict(features_df)
        └─► [NEW] record_prediction_trace() [src/ml/prediction_tracer.py]
              └─► INSERT INTO prediction_traces
```
