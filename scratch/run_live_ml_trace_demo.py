"""
CLI Runner for Live ML Trace Demo.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.live_trace_demo import run_live_ml_trace_demo, print_live_trace_demo

if __name__ == "__main__":
    res = run_live_ml_trace_demo(intern_id=10001, quiz_score=95.0)
    print_live_trace_demo(res)
