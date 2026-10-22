# SkillTrack-ML: Frontend Intern Skill Tracking & AI Forecasting Portal

SkillTrack-ML is a web-based portal and machine learning platform designed to monitor frontend engineering interns, evaluate coding progress, and provide predictive forecasts of future skill gaps and performance scores.

---

## 🛠️ Technology Stack
- **Frontend & App Framework:** Streamlit
- **Backend Database:** SQLite3 (`data/database/skilltrack.db`)
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Joblib
- **Language / Environment:** Python 3.12 (`.venv`)

---

## 🚀 Quickstart Guide

### 1. Activate Environment
```bash
# Windows
.venv\Scripts\activate
```

### 2. Launch the Application
```bash
streamlit run app/app.py
```
The application will open in your browser at `http://localhost:8501`.

---

## 🔑 Demo Accounts

| Role | Username | Password | Purpose |
| :--- | :--- | :--- | :--- |
| **Intern** | `intern1` | `intern123` | Complete tasks, take quizzes, view AI skill outlook |
| **Mentor** | `mentor1` | `mentor123` | Review cohort metrics, filter interns needing attention |

---

## 📊 Machine Learning Models
- **Skill Gap Classifier:** `models/skill_gap_model.pkl` (Balanced Logistic Regression predicting `Strong`, `Developing`, `Needs Improvement` for week $W+1$).
- **Performance Regressor:** `models/performance_model.pkl` (Ridge Regression predicting continuous score for week $W+1$).
- **Explainability:** Identifies top influential factors driving predictions without asserting causal claims.

> [!NOTE]
> The models were trained and evaluated on synthetic frontend intern dataset records (`data/skilltrack_frontend_interns_500x10_29cols.csv`). Reported metrics reflect performance on this synthetic dataset.
