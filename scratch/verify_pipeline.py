import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('data/skilltrack_frontend_interns_500x10_29cols.csv')
df = df.sort_values(['intern_id', 'week']).reset_index(drop=True)

# Engineer features strictly using week <= w
df['test_case_pass_rate'] = df['test_cases_passed'] / df['test_cases_total'].replace(0, 1)

# Historical expanding features
for col in ['coding_score', 'mcq_score', 'debugging_score', 'code_review_score', 'skill_score', 'learning_hours', 'engagement_rate', 'compile_error_rate', 'runtime_error_rate']:
    df[f'historical_{col}_avg'] = df.groupby('intern_id')[col].transform(lambda x: x.expanding().mean())

for col in ['learning_hours', 'tasks_completed', 'videos_completed', 'guides_completed', 'coding_attempts']:
    df[f'cumulative_{col}'] = df.groupby('intern_id')[col].transform(lambda x: x.cumsum())

# Shift targets
df['target_skill_gap'] = df.groupby('intern_id')['skill_gap'].shift(-1)
df['target_performance_score'] = df.groupby('intern_id')['final_performance_score'].shift(-1)
df['target_resource_action'] = df.groupby('intern_id')['resource_action'].shift(-1)

labeled = df.dropna(subset=['target_skill_gap']).copy()
print(f"Labeled training samples: {len(labeled)}")

# Check predictive model on next-week target
features = [
    'week', 'days_since_start', 'learning_hours', 'videos_completed', 'guides_completed',
    'tasks_completed', 'engagement_rate', 'coding_attempts', 'execution_count',
    'average_time_minutes', 'compile_error_rate', 'runtime_error_rate', 'test_cases_total',
    'test_cases_passed', 'test_case_pass_rate', 'coding_score', 'mcq_attempts', 'mcq_score',
    'debugging_score', 'code_review_score', 'previous_skill_score', 'skill_score',
    'score_improvement', 'final_performance_score',
    'historical_coding_score_avg', 'historical_mcq_score_avg', 'historical_debugging_score_avg',
    'historical_code_review_score_avg', 'historical_skill_score_avg', 'historical_learning_hours_avg',
    'historical_engagement_rate_avg', 'historical_compile_error_rate_avg', 'historical_runtime_error_rate_avg',
    'cumulative_learning_hours', 'cumulative_tasks_completed', 'cumulative_videos_completed',
    'cumulative_guides_completed', 'cumulative_coding_attempts'
]

X = labeled[features]
y = labeled['target_skill_gap']

# Time-based split or Intern-based split
train_mask = labeled['intern_id'] <= 10400 # 80% interns train, 20% interns test
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"\nNext-Week Skill Gap Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
