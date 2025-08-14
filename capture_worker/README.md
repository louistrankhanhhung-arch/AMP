# capture_worker/README.md

Worker chụp ảnh TradingView + trigger 1H chạy nền trên Railway/Cloud.

## Triển khai
1) Push repo (chứa thư mục `capture_worker/`) lên GitHub.
2) Railway → New Service → Deploy from GitHub.
3) Root Directory: repo root. Dockerfile Path: `capture_worker/Dockerfile`.
4) Env cần thiết:
   - EXCHANGE=KUCOIN
   - CAPTURE_TFS=60,240,D
   - INTERVAL_MIN=60
   - MIN_BUCKET=A
   - MIN_SCORE=7
   - OUT_DIR=/app/out_batch_triggers
   - (chọn 1) STRUCTS_URL=https://your-app/structs.json  hoặc  SYMBOLS=SUI/USDT,BTC/USDT
5) Deploy. Worker sẽ lặp mỗi INTERVAL_MIN phút.

Ảnh & JSON nằm trong OUT_DIR.
