# capture_worker/capture_worker.py
import os, time, subprocess, pathlib, requests, traceback

EXCHANGE     = os.getenv("EXCHANGE", "KUCOIN")
CAPTURE_TFS  = os.getenv("CAPTURE_TFS", "60,240,D")
INTERVAL_MIN = int(os.getenv("INTERVAL_MIN", "60"))
MIN_BUCKET   = os.getenv("MIN_BUCKET", "A")
MIN_SCORE    = os.getenv("MIN_SCORE", "7")
OUT_DIR      = os.getenv("OUT_DIR", "/app/out_batch_triggers")
STRUCTS_URL  = os.getenv("STRUCTS_URL", "")
SYMBOLS      = os.getenv("SYMBOLS", "")
RETENTION_HOURS = int(os.getenv("RETENTION_HOURS", "6"))

# chọn script batch
BATCH_SCRIPT = "batch_triggers.py" if pathlib.Path("batch_triggers.py").exists() else "batch_trigger.py"

def cleanup_old_files(dir_path, hours=6):
    try:
        cutoff = time.time() - hours * 3600
        p = pathlib.Path(dir_path)
        for f in p.glob("*"):
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
            except Exception:
                pass
    except Exception:
        print("[worker] cleanup_old_files error:", traceback.format_exc())

def run_once():
    # Dùng 'cmd' thay vì 'args' để tránh nhầm với argparse
    cmd = [
        "python", BATCH_SCRIPT,
        "--exchange", EXCHANGE,
        "--capture", "--capture-tfs", CAPTURE_TFS,
        "--out", OUT_DIR,
        "--min-bucket", MIN_BUCKET, "--min-score", str(MIN_SCORE)
    ]

    tmp_path = None
    try:
        if STRUCTS_URL:
            print(f"[worker] Fetching structs from {STRUCTS_URL} ...")
            r = requests.get(STRUCTS_URL, timeout=30)
            r.raise_for_status()
            tmp_path = pathlib.Path("/tmp/structs.json")
            tmp_path.write_text(r.text, encoding="utf-8")
            cmd.extend(["--structs-json", str(tmp_path)])
        elif SYMBOLS:
            print(f"[worker] Using SYMBOLS env: {SYMBOLS}")
            cmd.extend(["--symbols", SYMBOLS])
        else:
            print("[worker] No STRUCTS_URL or SYMBOLS provided; skip this round.")
            return

        print("[worker] Running:", " ".join(cmd))
        subprocess.run(cmd, check=False)
        print("[worker] Done.")
    except Exception:
        print("[worker] Error while building/running command:")
        print(traceback.format_exc())
    finally:
        cleanup_old_files(OUT_DIR, RETENTION_HOURS)

def main():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    while True:
        try:
            run_once()
        except Exception:
            print("[worker] Fatal loop error:\n", traceback.format_exc())
        time.sleep(INTERVAL_MIN * 60)

if __name__ == "__main__":
    main()
