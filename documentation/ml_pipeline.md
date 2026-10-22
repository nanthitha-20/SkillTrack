# SkillTrack-ML: End-to-End Machine Learning Pipeline

## Pipeline Architecture Overview

```text
                                 INTERN ACTIVITY (Quizzes / Tasks)
                                                 │
                                                 ▼
                                        ml_observations
                                                 │
                     ┌───────────────────────────┴───────────────────────────┐
                     ▼                                                       ▼
         IMMEDIATE SUPERVISED LEARNING                            BATCH FORECASTING PIPELINE
         (Outcome known immediately)                              (W -> W+1 Shifted Target)
                     │                                                       │
                     ▼                                                       ▼
         ONLINE INCREMENTAL LEARNER                               UNIFIED DATASET BUILDER
          src/ml/online_learner.py                              src/ml/build_unified_dataset.py
       • SGDRegressor with partial_fit()                        • 5,000 synthetic baseline rows
       • Adaptive learning rate & L2 reg                        • 4,500 supervised W->W+1 rows
       • Millisecond model update                               • Strict temporal ordering
                     │                                                       │
                     ▼                                                       ▼
            ONLINE MODEL ARTIFACT                                 INTERN-LEVEL PARTITIONING
          models/online_event_model.pkl                         • 350 Train / 75 Val / 75 Test
                     │                                                       │
                     │                                                       ▼
                     │                                            BATCH MODEL TRAINING
                     │                                          • Balanced Logistic Regression
                     │                                          • Ridge Linear Regression
                     │                                                       │
                     │                                                       ▼
                     │                                            QUALITY GATE VALIDATION
                     │                                          • Macro F1 >= 0.65
                     │                                          • NI Recall >= 0.70
                     │                                          • Retrain Threshold >= 50
                     │                                                       │
                     │                                                       ▼
                     │                                            PRODUCTION MODEL ARTIFACT
                     │                                           models/production_model.pkl
                     │                                                       │
                     └───────────────────────────┬───────────────────────────┘
                                                 │
                                                 ▼
                                     LIVE PREDICTION SERVICE
                                        src/ml/predict.py
                                 • 43 features strictly (t <= T)
                                 • Real-time forecast & confidence
                                                 │
                                                 ▼
                                    INTERN & MENTOR PORTALS
```

---

## 1. Why 5,000 Raw Rows Become 4,500 Supervised Forecasting Rows
The raw dataset contains 10 weekly records per intern for 500 interns ($500 \times 10 = 5,000$ rows).
Because the ML task is **forward-looking prediction** (predicting the outcome at Week $W+1$ given data through Week $W$), each record at Week $W$ is paired with the target observed at Week $W+1$.

For Week 10 (the final observation in the bootstrap dataset), there is no recorded Week 11 outcome to serve as ground truth. Consequently, Week 10 records cannot be used for supervised training and remain as pending forecast samples ($5,000 - 500 = 4,500$ rows).

---

## 2. Dual Online Learning Paths (No Label Fabrication)

SkillTrack-ML implements two distinct learning paths:
1. **Immediate Event-Level Supervised Path:**
   - Activities with runtime measurable outcomes (quizzes, evaluated code submissions) have known scores at execution time.
   - Tagged as `label_status = IMMEDIATE_LABELED` and `training_eligible = 1`.
   - Update the online incremental learner (`SGDRegressor`) via `partial_fit()`.
2. **Next-Week Forecasting Path:**
   - When Week $W$ features are evaluated, Week $W+1$ outcome is **unknown**.
   - Recorded as `label_status = PENDING_FUTURE_LABEL`, `outcome_value = NULL`, and `training_eligible = 0`.
   - **Ground-truth labels are never fabricated.**
   - When Week $W+1$ performance is later observed, the automated resolution function updates the label to `LABELED` and activates training eligibility.

---

## 3. Why Grouped Intern-Level Splitting Is Required
Standard random row splitting randomly assigns different weeks of the same intern across train, validation, and test sets. This creates severe temporal and entity data leakage because an intern's baseline capability in Week 3 strongly informs Week 4.

To ensure realistic generalization, data splitting is performed strictly at the **intern entity level**:
- **Train Interns:** 350 interns (3,150 rows)
- **Validation Interns:** 75 interns (675 rows)
- **Test Interns:** 75 interns (675 rows)
- **Zero Overlap:** No `intern_id` appears in more than one partition.

---

## 4. Target Definitions

### Classification Target: `target_skill_gap`
- **Classes:** `Strong`, `Developing`, `Needs Improvement`
- **Definition:** The intern's skill mastery tier observed at the subsequent week ($W+1$).
- **Purpose:** Early risk identification to alert mentors and provide targeted learning recommendations before the intern falls behind.

### Regression Target: `target_performance_score`
- **Range:** $0.0 - 100.0$
- **Definition:** The overall performance score achieved at the subsequent week ($W+1$).
- **Purpose:** Continuous numerical trajectory forecasting.

---

## 5. Leakage Prevention Protocol
1. **No Current-Week Deterministic Shortcuts:** The current week's `skill_gap`, `resource_action`, and `target_*` columns are strictly dropped from input features $X$.
2. **Strictly Backward-Looking Rolling Features:** Expanding window statistics (e.g. `historical_coding_score_avg`, `cumulative_learning_hours`) for prediction origin Week $W$ are computed strictly using historical weeks $\le W$.
3. **Pipeline Encapsulation:** Scaling (`StandardScaler`) and encoding (`OneHotEncoder`) are fit strictly inside Scikit-learn `Pipeline` objects to prevent test-set distribution leakage.
4. **Zero Future Target Fabrication:** Pending forecasting rows hold `NULL` until actual ground truth arrives.
