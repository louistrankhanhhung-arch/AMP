FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

# Python envs (1 lần, trước khi copy code)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Cài deps theo layer cacheable
COPY capture_worker/requirements.txt /tmp/requirements.txt
RUN pip install -U pip && pip install -r /tmp/requirements.txt

# Copy mã nguồn (toàn repo để worker thấy universe.py, batch_triggers.py, v.v.)
COPY . /app

# Tuỳ chọn: base image đã có browsers; nếu muốn chắc ăn thì giữ dòng dưới
# RUN playwright install chromium

# Giá trị mặc định có thể override qua Railway Variables
ENV EXCHANGE=KUCOIN \
    CAPTURE_TFS=60,240,D \
    INTERVAL_MIN=60 \
    MIN_BUCKET=A \
    MIN_SCORE=7 \
    OUT_DIR=/app/out_batch_triggers
    # STRUCTS_URL và SYMBOLS không đặt ở đây; cấu hình qua Variables khi cần

CMD ["python", "capture_worker/capture_worker.py"]
