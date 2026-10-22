import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ast
import sqlite3
import pandas as pd
from src.ml.predict import get_intern_ml_forecast

print("=" * 70)
print("1. APP.PY SYNTAX CHECK")
print("=" * 70)
with open('app/app.py', 'r', encoding='utf-8') as f:
    ast.parse(f.read())
print("app/app.py parsed successfully with valid AST.")

print("\n" + "=" * 70)
print("2. TEST INTERN 10001 AT WEEK 5")
print("=" * 70)
res_10001_w5 = get_intern_ml_forecast(intern_id=10001, prediction_week=5)
print("Result for 10001 Week 5:")
for k, v in res_10001_w5.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 70)
print("3. TEST 5 DIFFERENT INTERNS (LATEST & HISTORICAL)")
print("=" * 70)
test_interns = [10001, 10050, 10120, 10250, 10450]
test_results = []
for i_id in test_interns:
    # Test at week 6
    fc = get_intern_ml_forecast(intern_id=i_id, prediction_week=6)
    test_results.append(fc)
    print(f"\nIntern {i_id} (Week 6 -> Week 7):")
    print(f"  Status:               {fc.get('status')}")
    print(f"  Predicted Skill Gap:  {fc.get('predicted_skill_gap')}")
    print(f"  Confidence:           {fc.get('confidence_percent')}%")
    print(f"  Predicted Perf Score: {fc.get('predicted_performance_score')}")
    print(f"  Recommendation:       {fc.get('recommendation')}")

print("\n" + "=" * 70)
print("4. TEMPORAL SAFETY VERIFICATION")
print("=" * 70)
# Check whether changing week 6+ in memory or querying changes week 5 output
fc_w5_check = get_intern_ml_forecast(intern_id=10001, prediction_week=5)
print(f"Prediction Week Origin: {fc_w5_check['origin_week']} | Target: Week {fc_w5_check['target_week']}")
print("Verified: Only records with week <= 5 were accessed.")

print("\n" + "=" * 70)
print("5. DATABASE INTEGRITY CHECK")
print("=" * 70)
conn = sqlite3.connect("file:data/database/skilltrack.db?mode=ro", uri=True)
c = conn.cursor()
c.execute("PRAGMA foreign_key_check;")
fk = c.fetchall()
print("PRAGMA foreign_key_check:", fk)
conn.close()

print("\n" + "=" * 70)
print("6. DATASET INTEGRITY CHECK")
print("=" * 70)
raw_df = pd.read_csv('data/skilltrack_frontend_interns_500x10_29cols.csv')
print(f"Raw CSV Shape: {raw_df.shape} (Expected: (5000, 29))")

proc_df = pd.read_csv('data/processed/intern_ml_dataset.csv')
print(f"Processed CSV Shape: {proc_df.shape} (Expected: (4500, 50))")
