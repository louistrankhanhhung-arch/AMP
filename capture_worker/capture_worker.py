# capture_worker/capture_worker.py
import os, time, subprocess, pathlib, requests

EXCHANGE     = os.getenv("EXCHANGE", "KUCOIN")
CAPTURE_TFS  = os.getenv("CAPTURE_TFS", "60,240,D")
INTERVAL_MIN = int(os.getenv("INTERVAL_MIN", "60"))
MIN_BUCKET   = os.getenv("MIN_BUCKET", "A")
MIN_SCORE    = os.getenv("MIN_SCORE", "7")
OUT_DIR      = os.getenv("OUT_DIR", "/app/out_batch_triggers")
STRUCTS_URL  = os.getenv("STRUCTS_URL", "")
SYMBOLS      = os.getenv("SYMBOLS", "")

BATCH_SCRIPT = "batch_triggers.py" if pathlib.Path("batch_triggers.py").exists() else "batch_trigger.py"

def run_once():
    args = ["python", BATCH_SCRIPT,
            "--exchange", EXCHANGE,
            "--capture", "--capture-tfs", CAPTURE_TFS,
            "--out", OUT_DIR,
            "--min-bucket", MIN_BUCKET, "--min-score", str(MIN_SCORE)]
    tmp_path = None
    if STRUCTS_URL:
        print(f"[worker] Fetching structs from {STRUCTS_URL} ...")
        r = requests.get(STRUCTS_URL, timeout=30); r.raise_for_status()
        tmp_path = pathlib.Path("/tmp/structs.json"); tmp_path.write_text(r.text, encoding="utf-8")
        args.extend(["--structs-json", str(tmp_path)])
    elif SYMBOLS:
        print(f"[worker] Using SYMBOLS env: {SYMBOLS}")
        args.extend(["--symbols", SYMBOLS])
    else:
        print("[worker] No STRUCTS_URL or SYMBOLS provided; skip this round."); return
    print("[worker] Running:", " ".join(args))
    subprocess.run(args, check=False); print("[worker] Done.")

def main():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    while True:
        try: run_once()
        except Exception as e: print("[worker] Error:", repr(e))
        time.sleep(INTERVAL_MIN * 60)

if __name__ == "__main__": main()
