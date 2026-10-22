"""
Export complete SkillTrack-ML dataset tables to Excel workbook (.xlsx).
"""

import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "database" / "skilltrack.db"
CSV_UNIFIED_PATH = BASE_DIR / "data" / "processed" / "intern_ml_dataset.csv"
CSV_RAW_PATH = BASE_DIR / "data" / "skilltrack_frontend_interns_500x10_29cols.csv"
OUTPUT_XLSX = BASE_DIR / "data" / "SkillTrack_ML_Dataset.xlsx"

print(f"Connecting to database at {DB_PATH}...")
conn = sqlite3.connect(DB_PATH)

# 1. Unified ML Dataset (4,500 supervised transition rows)
if CSV_UNIFIED_PATH.exists():
    df_unified = pd.read_csv(CSV_UNIFIED_PATH)
else:
    df_unified = pd.read_sql_query("SELECT * FROM intern_performance WHERE week < 10", conn)

# 2. Historical Baseline Performance (5,000 rows)
df_perf = pd.read_sql_query("SELECT * FROM intern_performance ORDER BY intern_id, week", conn)

# 3. Live Application Observations
df_obs = pd.read_sql_query("SELECT * FROM ml_observations ORDER BY observation_id DESC", conn)

# 4. Intern Profiles
df_interns = pd.read_sql_query("SELECT * FROM interns ORDER BY intern_id", conn)

# 5. Curated Learning Resources
df_resources = pd.read_sql_query("SELECT * FROM resources ORDER BY resource_id", conn)

# 6. Model Registry & Online Updates
df_models = pd.read_sql_query("SELECT * FROM model_registry ORDER BY model_id DESC", conn)
df_online = pd.read_sql_query("SELECT * FROM online_learning_updates ORDER BY update_id DESC", conn)

# 7. Dataset Versions
df_versions = pd.read_sql_query("SELECT * FROM ml_dataset_versions ORDER BY dataset_version_id DESC", conn)

conn.close()

print(f"Writing Excel workbook to {OUTPUT_XLSX}...")
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    df_unified.to_excel(writer, sheet_name="Unified_ML_Dataset", index=False)
    df_perf.to_excel(writer, sheet_name="Historical_Performance_5000", index=False)
    df_obs.to_excel(writer, sheet_name="Live_Observations", index=False)
    df_interns.to_excel(writer, sheet_name="Intern_Profiles", index=False)
    df_resources.to_excel(writer, sheet_name="Learning_Resources", index=False)
    df_models.to_excel(writer, sheet_name="Model_Registry", index=False)
    df_online.to_excel(writer, sheet_name="Online_Learning_Log", index=False)
    df_versions.to_excel(writer, sheet_name="Dataset_Versions", index=False)

print(f"[OK] Successfully exported Excel workbook: {OUTPUT_XLSX}")
print(f"File size: {OUTPUT_XLSX.stat().st_size / (1024*1024):.2f} MB")
