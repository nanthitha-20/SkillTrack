import sys

with open('app/app.py', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        low = line.lower()
        if 'live ml' in low or 'live_ml' in low or 'run_live' in low or 'trace demo' in low or 'retrain' in low:
            # print ascii only to avoid cp1252 crash
            safe_line = line.strip()[:80].encode('ascii', errors='replace').decode('ascii')
            print(f"{idx+1}: {safe_line}")
