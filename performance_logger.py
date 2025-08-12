import json, os
from datetime import datetime

LOG_DIR = "/mnt/data/perf_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def log_closed_signal(payload: dict):
    path = os.path.join(LOG_DIR, f"closed_{payload['signal_id']}.json")
    payload['logged_at'] = datetime.utcnow().isoformat()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def summarize_signal(signal_id: str) -> str:
    # TODO: compute stats across closed signals
    return f"Summary for {signal_id} (stub)"
