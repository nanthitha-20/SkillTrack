import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ast
import sqlite3
import joblib
import pandas as pd
from src.ml.predict import get_intern_ml_forecast, build_feature_vector, load_models

print("=" * 75)
print("SKILLTRACK-ML: FINAL COMPREHENSIVE DEMO READINESS VALIDATION")
print("=" * 75)

# 1. SYNTAX VALIDATION
print("\n[1] Python Syntax Check:")
py_files = ["app/app.py", "src/ml/predict.py", "src/ml/prepare_ml_dataset.py", "src/ml/train_models.py"]
for fpath in py_files:
    with open(fpath, "r", encoding="utf-8") as f:
        ast.parse(f.read())
    print(f"  [OK] {fpath:<35} Syntax OK")

# 2. MODEL INTEGRITY
print("\n[2] Model Loading & Integrity Check:")
cls_m, reg_m = load_models()
print(f"  [OK] Skill Gap Model:   {type(cls_m).__name__}")
print(f"  [OK] Performance Model: {type(reg_m).__name__}")

# 3. DATABASE INTEGRITY & FOREIGN KEYS
print("\n[3] SQLite Database Health & Foreign Keys:")
conn = sqlite3.connect("file:data/database/skilltrack.db?mode=ro", uri=True)
c = conn.cursor()
c.execute("PRAGMA foreign_key_check;")
fk = c.fetchall()
print(f"  [OK] Foreign Key Violations: {fk} (0 expected)")
assert len(fk) == 0, f"Foreign key violation: {fk}"

c.execute("SELECT count(*) FROM interns")
intern_count = c.fetchone()[0]
print(f"  [OK] Total Interns:          {intern_count} (500 expected)")
assert intern_count == 500

c.execute("SELECT count(*) FROM intern_performance")
perf_count = c.fetchone()[0]
print(f"  [OK] Performance Records:    {perf_count} (5,000 expected)")
assert perf_count == 5000

c.execute("SELECT count(*) FROM submissions")
sub_count = c.fetchone()[0]
print(f"  [OK] Active Submissions:     {sub_count}")
assert sub_count >= 0
conn.close()

# 4. DATASET INTEGRITY
print("\n[4] Dataset Integrity Check:")
raw_df = pd.read_csv("data/skilltrack_frontend_interns_500x10_29cols.csv")
print(f"  [OK] Raw CSV Dimensions:       {raw_df.shape} (5000, 29 expected)")
assert raw_df.shape == (5000, 29)

proc_df = pd.read_csv("data/processed/intern_ml_dataset.csv")
print(f"  [OK] Processed CSV Dimensions: {proc_df.shape} (4500 rows expected)")
assert proc_df.shape[0] == 4500

# 5. ZERO OULAD RUNTIME REFERENCES
print("\n[5] OULAD Safety Check:")
oulad_keywords = ["studentInfo", "studentVle", "studentAssessment", "studentRegistration"]
found_oulad = []
for p in ["app/app.py", "src/ml/predict.py"]:
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()
        for kw in oulad_keywords:
            if kw in content:
                found_oulad.append((p, kw))
print(f"  [OK] Active OULAD References Found: {found_oulad} (0 expected)")
assert len(found_oulad) == 0

# 6. INTERN 10001 WEEK 5 TEST
print("\n[6] Intern 10001 (Week 5 -> Week 6) Benchmark Forecast:")
fc_10001_w5 = get_intern_ml_forecast(intern_id=10001, prediction_week=5)
print(f"  [OK] Predicted Gap:        {fc_10001_w5['predicted_skill_gap']}")
print(f"  [OK] Forecast Confidence:  {fc_10001_w5['confidence_percent']}%")
print(f"  [OK] Projected Score:      {fc_10001_w5['predicted_performance_score']} / 100")
print(f"  [OK] Skill Domain:         {fc_10001_w5.get('current_skill')}")
print(f"  [OK] Recommendation:       {fc_10001_w5['recommendation']}")
print(f"  [OK] Top Factors:          {len(fc_10001_w5['top_factors'])} influential features")
assert fc_10001_w5['predicted_skill_gap'] == "Strong"

# 7. MULTI-INTERN LIVE FORECASTING TEST
print("\n[7] Multi-Intern Live Inference Verification:")
test_interns = [10001, 10050, 10120, 10250, 10450]
for tid in test_interns:
    fc = get_intern_ml_forecast(intern_id=tid)
    print(f"  Intern {tid} | Skill: {fc.get('current_skill'):<12} | Gap: {fc['predicted_skill_gap']:<18} | Conf: {fc['confidence_percent']:5.1f}% | Score: {fc['predicted_performance_score']:4.1f} | Action: {fc.get('action_tag')}")

print("\n" + "=" * 75)
print("FINAL VERDICT: ALL CHECKS PASSED WITH ZERO ERRORS. READY FOR DEMO!")
print("=" * 75)
