# SkillTrack: Frontend Intern Skill Tracking & AI Forecasting Portal

SkillTrack is a web-based internship management and skill development platform designed to track frontend engineering interns, evaluate coding progress, and forecast future skill gaps and performance scores using machine learning.

## 🛠️ Technology Stack

- **Frontend & Application Framework:** Streamlit
- **Backend Database:** SQLite3
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Joblib
- **Programming Language:** Python 3.12
- **Version Control:** Git and GitHub
- **Project Tracking:** Jira
- **Performance Testing:** Apache JMeter

## 🏗️ Software Development Model

**Development Model: Iterative and Incremental Model**

SkillTrack follows an iterative and incremental development approach, where the application is developed and improved in multiple stages. Features such as authentication, intern task tracking, coding challenge submissions, mentor evaluation, machine learning predictions, and performance testing are implemented and refined incrementally through testing and feedback.

## 🚀 Quickstart Guide

### 1. Activate the Python Environment

```powershell
.venv\Scripts\activate
```

### 2. Launch the Application

```powershell
streamlit run app/app.py
```

Open the application in your browser at `http://localhost:8501`.

## 🔑 Demo Accounts

| Role | Username | Password | Purpose |
|---|---|---|---|
| Intern | `intern1` | `intern123` | Complete tasks, take quizzes, and view skill forecasts |
| Mentor | `mentor1` | `mentor123` | Review intern progress and identify interns needing support |

> **Note:** These are demonstration credentials. Change them before any real deployment.

## 📊 Machine Learning Models

- **Skill Gap Classifier — Logistic Regression:** Predicts the next week's skill category: `Strong`, `Developing`, or `Needs Improvement`.
- **Performance Regressor — Ridge Regression:** Predicts the continuous performance score for the following week.
- **Explainability:** Highlights influential factors behind predictions without claiming that they cause the outcomes.

The models are trained and evaluated using a synthetic frontend intern dataset containing 500 records. Reported metrics reflect evaluation on synthetic data and should not be interpreted as proof of real-world predictive performance.

## 🧪 Testing and Project Management

- **Jira:** Tracks tasks, defects, and project progress.
- **Apache JMeter:** Evaluates application response behavior under simulated concurrent requests.
- **Git and GitHub:** Maintain source-code history and support version control.

## 🗄️ Database

The application uses SQLite to store application data locally at:

`data/database/skilltrack.db`

The database is managed by the application's database scripts.
