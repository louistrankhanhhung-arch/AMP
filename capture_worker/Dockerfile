# capture_worker/Dockerfile
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY capture_worker/requirements.txt /app/requirements.txt
RUN pip install -U pip && pip install -r /app/requirements.txt
COPY . /app
RUN playwright install chromium
ENV EXCHANGE=KUCOIN CAPTURE_TFS=60,240,D INTERVAL_MIN=60 MIN_BUCKET=A MIN_SCORE=7 OUT_DIR=/app/out_batch_triggers STRUCTS_URL= SYMBOLS=
CMD ["python", "capture_worker/capture_worker.py"]
