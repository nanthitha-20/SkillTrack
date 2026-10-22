import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data.live_data_ingestion import get_observation_summary
from src.ml.online_learner import get_online_learning_status
from src.ml.model_registry import get_active_production_model

conn = sqlite3.connect(BASE_DIR / 'data' / 'database' / 'skilltrack.db')
cur = conn.cursor()

summary = get_observation_summary()
online_status = get_online_learning_status()
prod_model = get_active_production_model()

cur.execute('SELECT COUNT(*) FROM resources')
total_res = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM resources WHERE resource_type = 'video'")
video_res = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM resources WHERE resource_type = 'documentation'")
doc_res = cur.fetchone()[0]

cur.execute('SELECT DISTINCT category FROM resources')
categories = [r[0] for r in cur.fetchall()]

cur.execute('SELECT COUNT(*) FROM chat_messages')
chat_count = cur.fetchone()[0]

cur.execute('SELECT dataset_version, training_eligible_rows FROM ml_dataset_versions ORDER BY dataset_version_id DESC LIMIT 1')
ds_ver = cur.fetchone()

conn.close()

print('=== DATASET ===')
print('Total observations:', summary['total_active_observations'])
print('Synthetic observations:', summary['synthetic_baseline_count'])
print('Application observations:', summary['application_generated_count'])
print('Immediate supervised rows:', summary['immediate_supervised_count'])
print('Pending forecasting rows:', summary['pending_future_label_count'])
print('Labeled forecasting rows:', summary['resolved_future_label_count'])
print('Total ML training rows:', summary['training_eligible_samples'])
print('Dataset version:', ds_ver[0] if ds_ver else 'v2.0-live')

print('\n=== ONLINE LEARNING ===')
print('Online algorithm:', online_status['online_algorithm'])
print('Online model version:', online_status['online_model_version'])
print('Total online updates:', online_status['total_online_updates'])
print('Last update:', online_status['last_online_update'])
print('Batch model version:', prod_model['model_version'])
print('Batch model validation metrics:', prod_model['validation_metrics'])
print('Average online step loss:', online_status['average_step_loss'])

print('\n=== RESOURCES ===')
print('Total Java resources:', total_res)
print('Videos:', video_res)
print('Reference materials:', doc_res)
print('Categories:', categories)

print('\n=== CHAT ===')
print('Total chat messages persisted:', chat_count)
