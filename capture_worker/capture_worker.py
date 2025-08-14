# capture_worker/capture_worker.py
import os
import time
import subprocess
import pathlib
import requests
import traceback
from datetime import datetime
import json
import base64
from universe import get_universe_from_env

EXCHANGE         = os.getenv("EXCHANGE", "KUCOIN")
CAPTURE_TFS      = os.getenv("CAPTURE_TFS", "60,240,D")
INTERVAL_MIN     = int(os.getenv("INTERVAL_MIN", "60"))
MIN_BUCKET       = os.getenv("MIN_BUCKET", "A")
MIN_SCORE        = os.getenv("MIN_SCORE", "7")
OUT_DIR          = os.getenv("OUT_DIR", "/app/out_batch_triggers")
STRUCTS_URL      = os.getenv("STRUCTS_URL", "")
SYMBOLS          = os.getenv("SYMBOLS", "")
RETENTION_HOURS  = int(os.getenv("RETENTION_HOURS", "6"))

# Chọn script batch (tùy repo bạn đặt tên)
BATCH_SCRIPT = "batch_triggers.py" if pathlib.Path("batch_triggers.py").exists() else "batch_trigger.py"


def dump_outputs(dir_path: str, preview_json_chars: int = 1200, preview_image_bytes: int = 0):
    """
    In ra danh sách file trong OUT_DIR (tên, size, mtime).
    - Với .json: in trước preview_json_chars ký tự đầu để kiểm tra nội dung.
    - Với ảnh: có thể in base64 vài byte đầu nếu cần (preview_image_bytes>0), mặc định tắt để tránh dài log.
    """
    p = pathlib.Path(dir_path)
    print(f"[worker] Output dir: {p}")
    if not p.exists():
        print("[worker] Output dir not found.")
        return
    for f in sorted(p.glob("*")):
        try:
            ts = datetime.utcfromtimestamp(f.stat().st_mtime).isoformat() + "Z"
            print(f"[worker] -> {f.name} | {f.stat().st_size}B | mtime={ts}")
            if f.suffix.lower() == ".json":
                try:
                    s = f.read_text(encoding="utf-8", errors="ignore")
                    print("[worker] JSON head:", s[:preview_json_chars])
                except Exception as e:
                    print("[worker] JSON read error:", e)
            elif preview_image_bytes > 0 and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                try:
                    b64 = base64.b64encode(f.read_bytes()[:preview_image_bytes]).decode()
                    print("[worker] IMG head(base64):", b64[:200])
                except Exception as e:
                    print("[worker] Image read error:", e)
        except Exception as e:
            print("[worker] stat error:", e)


def cleanup_old_files(dir_path: str, hours: int = 6):
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
    cmd = [
        "python", BATCH_SCRIPT,
        "--exchange", EXCHANGE,
        "--capture", "--capture-tfs", CAPTURE_TFS,
        "--out", OUT_DIR,
        "--min-bucket", MIN_BUCKET, "--min-score", str(MIN_SCORE),
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
            syms = ",".join(get_universe_from_env())
            print(f"[worker] Using DEFAULT_UNIVERSE: {syms}")
            cmd.extend(["--symbols", syms])

        print("[worker] Running:", " ".join(cmd))
        subprocess.run(cmd, check=False)
        print("[worker] Done.")
    except Exception:
        print("[worker] Error while building/running command:")
        print(traceback.format_exc())
    finally:
        cleanup_old_files(OUT_DIR, RETENTION_HOURS)
        dump_outputs(OUT_DIR)


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
