# SkillTrack-ML: Live Online Learning Platform — Professor Demonstration Guide

**Demonstration Duration:** 3–4 Minutes  
**Target Audience:** Professor / Academic Evaluator  
**Core Demonstration Objective:**  
Demonstrate that SkillTrack-ML is a closed-loop **Online-Learning Internship Intelligence Platform**.
Show the exact 9-step demonstration journey proving that:
1. Every new application interaction enters the persistent ML data layer immediately.
2. Immediately measurable events (quizzes, coding tasks) create immediate supervised training rows and adapt the online incremental learner (`SGDRegressor` via `partial_fit()`).
3. Next-week forecasting records remain pending without fake label fabrication or temporal leakage.
4. Detected skill gaps deterministically recommend curated educational resources (Java, OOP, etc.).
5. The intern-to-mentor private chat facilitates intervention while capturing engagement metadata.

---

## 🚀 Quick Preparation (Before the Demo)

1. Ensure the Streamlit portal is running:
   ```bash
   streamlit run app/app.py
   ```
   Open `http://localhost:8501` in your browser.
2. Credentials:
   * **Intern:** `intern1` / `intern123`
   * **Mentor:** `mentor1` / `mentor123`

---

## ⏱️ Step-by-Step Demonstration Script (Exact 9-Step Sequence)

### Step 1: Show the Mentor Dashboard & Live ML Dataset Baseline (45 Seconds)
1. In the browser, log in as Mentor: `mentor1` / `mentor123`.
2. Navigate to **Live ML Dataset** in the sidebar.
3. Show the observation counters:
   * **Total Observations:** `5,015+` (Dynamic from database)
   * **Synthetic Baseline:** `5,000`
   * **Live Application Events:** `15+`
   * **Immediate Labeled:** `15+`
   * **Pending Future Labels:** `X`
4. Point out the **Recent Ingested Observations** streaming table with columns:
   `Obs ID | Timestamp | Intern ID | Intern Name | Week | Event Type | Source | Outcome | Label Status | Training Eligible`.
5. **What to say:**
   > *"Professor, our initial 5,000 records are synthetic and serve as the bootstrap baseline dataset. We continuously collect actual application-generated observations. Every new observation immediately enters our persistent ML data layer. Notice here in our Live ML Dataset view that every event has clear provenance, an explicit label status, and a training eligibility flag."*

---

### Step 2: Switch to Intern Portal & Inspect AI Forecast (30 Seconds)
1. Sign out from mentor and log in as Intern: `intern1` / `intern123`.
2. On the **Intern Dashboard**, point to the **Next-Week Skill Outlook** card:
   * **Predicted Skill Gap (Week 11):** e.g., `Strong` or `Developing`
   * **Forecast Confidence:** e.g., `99.6%`
   * **Projected Next-Week Performance:** e.g., `89.3 / 100`
3. Point to the indicator banner:
   `🟢 Live ML: Active features evaluated ... Target Week 11 outcome: Pending Future Label (No Fake Data)`.
4. **What to say (Explaining Forecasting Distinction):**
   > *"The platform predicts performance for Week W+1 based on Week W features. Because Week W+1 has not occurred yet, the ground-truth target is strictly unknown. We record the pending sample immediately, but keep its target NULL. We never fabricate labels, never copy current scores into future targets, and never use predictions as ground truth."*

---

### Step 3: Show the Adaptive Learning Loop & Curated Resources (30 Seconds)
1. In the Intern Portal sidebar, click on **Learning**.
2. Point out the **ADAPTIVE LEARNING LOOP** recommendation card:
   * Shows detected skill gap: e.g., `Advanced Engineering` or `Intermediate Acceleration`.
   * Shows tailored recommended modules (e.g., *Java Concurrency*, *Streams*, *JDBC*).
3. Scroll down to show the **Curated Resource Library**:
   * Show topic filters (Java Basics, OOP, Collections, Exception Handling, Concurrency).
   * Demonstrate opening authoritative resources (e.g., Oracle Java documentation or verified video).
4. **What to say:**
   > *"Our ML model identifies specific skill weaknesses, which automatically links to curated educational materials. The intern studies the recommended module before attempting a new assessment."*

---

### Step 4: Perform a Live Assessment Activity (30 Seconds)
1. In the Intern Portal, click **Assessments** in the sidebar.
2. Select an assessment topic (e.g., *Technical Quiz*), select answers, and click **Submit Assessment**.
3. Note the completed score (e.g., `100%`).
4. **What to say (Explaining Immediate Supervised Learning):**
   > *"Now the intern completes a technical assessment. Unlike next-week forecasting, the outcome of this quiz is immediately measurable at runtime. This event instantly becomes a supervised training row and immediately updates our online incremental learner via partial_fit()."*

---

### Step 5: Verify Live Dataset Count Increment in Mentor Portal (30 Seconds)
1. Sign out and log back in as Mentor: `mentor1` / `mentor123`.
2. Navigate to **Live ML Dataset**:
   * Point out that **Total Observations** has dynamically incremented by 1 (e.g., from `5,015` to `5,016`).
   * Show the new row at the very top of the **Recent Ingested Observations** table:
     * `Source: quiz`
     * `Event Type: mcq_attempted`
     * `Outcome: 100.0`
     * `Label Status: IMMEDIATE_LABELED`
     * `Training Eligible: 1`
3. Point out that the **Online Incremental Model** step count increased.
4. **What to say:**
   > *"As you can see, without manually running any offline script, the live observation appeared immediately in SQLite. It has `label_status = IMMEDIATE_LABELED`, is marked training-eligible, and has adapted our online model in real time."*

---

### Step 6: Demonstrate the Intern-Mentor Chat System (30 Seconds)
1. Navigate to **Messages** in the Mentor Portal:
   * Show the list of assigned interns with unread count indicators (e.g., `🔴 Intern 10001 (1 unread)`).
   * Open the conversation thread with Intern 10001.
   * Point out the conversation history and timestamps.
   * Type a mentor reply (e.g., *"Great job on the quiz! Keep working on multithreading."*) and click **Send Reply**.
2. **What to say (Privacy & Governance):**
   > *"We also integrated a persistent, private intern-mentor chat system. Interns only see their assigned mentor; mentors only see their cohort interns. Crucially, private message text is NEVER used as ML training data to protect privacy—only non-sensitive interaction counts can contribute to engagement metrics."*

---

## 🎯 Summary Talking Points

```text
                INTERN ACTIVITY (Quizzes / Coding)
                                │
                                ▼
                       LIVE OBSERVATIONS
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
        Immediate Supervised          Pending Forecasting
        (Outcome Known: 100%)         (W+1 Outcome: UNKNOWN)
                  │                           │
                  ▼                           ▼
         Online Learner (SGD)         Wait for Actual Arrival
            partial_fit()                     │
                  │                           ▼
                  │                  Automated Resolution
                  │                           │
                  └─────────────┬─────────────┘
                                ▼
                        Dynamic Inference
                                │
                                ▼
                       Skill Gap Detected
                                │
                                ▼
                        Curated Resources
                                │
                                ▼
                       Intern Improvement
                                +
                      Mentor Direct Guidance
```

### Direct Quotations for the Evaluation:

1. **On Dataset Authenticity:**
   > *"Our initial 5,000 records are synthetic and serve as the bootstrap dataset. We then continuously collect actual application-generated observations. Every new observation immediately enters our persistent ML data layer."*

2. **On Immediate vs Forecasting Labels:**
   > *"When the outcome is known immediately, such as a quiz or evaluated coding submission, it becomes a supervised training example immediately and can update the online learner. For future forecasting targets, the sample is created immediately but remains pending until the actual future outcome becomes available. This prevents us from fabricating labels or leaking future information."*

3. **On the Educational Feedback Loop:**
   > *"The model identifies a skill gap, recommends relevant learning resources, the intern learns and attempts another assessment, and that new result feeds back into the ML system. The mentor can communicate directly with the intern through the integrated chat system."*
