from datetime import datetime, timezone, timedelta

def now_utc():
    return datetime.now(timezone.utc)

def humanize_delta(delta):
    s = int(delta.total_seconds())
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    out = []
    if d: out.append(f"{d}d")
    if h: out.append(f"{h}h")
    if m: out.append(f"{m}m")
    if s and not out: out.append(f"{s}s")
    return " ".join(out) or "0m"
