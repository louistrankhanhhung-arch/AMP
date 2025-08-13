
# Signal App (Lean Pipeline)

Hai cách chạy:
- **Image-only**: chỉ chụp chart từ TradingView → gửi GPT phân tích.
- **Hybrid Lite (khuyên dùng)**: chụp chart + gửi kèm JSON từ `dump_structs.py` để cố định số liệu (EMA/RSI/ATR/SR/ETA…).

## 1) Cài đặt

```bash
python -m venv .venv
# Git Bash (Windows)
source .venv/Scripts/activate

pip install -U pip
pip install -r requirements.txt

# Cài browser cho Playwright (1 lần)
playwright install chromium
```

## 2) Chụp chart TradingView (Ảnh)

```bash
# SUI 4H & 1D từ KuCoin
python capture_tv.py --exchange KUCOIN --symbol SUIUSDT --tfs 240,D --outdir out_tv

# BTC 1H/4H/1D từ Binance
python capture_tv.py --exchange BINANCE --symbol BTCUSDT --tfs 60,240,D --outdir out_tv
```

- Output: PNG tại `out_tv/` (ví dụ: `SUIUSDT_240_tv.png`, `SUIUSDT_D_tv.png`).
- Timezone = UTC, indicators = **EMA20/EMA50**, **Bollinger(20,2)**, **RSI(14)**, **Volume**.

## 3) (Tuỳ chọn) Xuất JSON cho GPT

```bash
# JSON đầy đủ với futures extras + market context
python dump_structs.py --symbol SUI/USDT --tfs 4H,1D --limit 300 \
  --with-futures-extras --with-market-context \
  --out out/SUI_4H1D.json
```

JSON này chứa:
- `snapshot` (MA/BB/RSI/ATR/Volume…)
- `levels` (sr_up/sr_down + soft_sr)
- `events` (breakout/breakdown + volume_ok)
- `targets` (up/down bands) + `eta_hint`
- `futures_sentiment` (OI delta 24h, L/S ratio nếu bật extras)
- `context_guidance` (SR mềm từ khung lớn)

## 4) Gửi vào GPT

- **Ảnh**: 1D & 4H PNG từ `out_tv/`.
- **Context**: JSON từ `out/` (toàn bộ hoặc rút gọn).
- Prompt gợi ý: “Ưu tiên đọc ảnh để nhận diện cấu trúc/nến; dùng JSON để xác nhận số (MA/RSI/ATR/SR/ETA). Chỉ đề xuất setup nếu bucket A (score ≥ 7).”

## 5) Lỗi thường gặp

- **`ValueError: relative path can't be expressed as a file URI`** (Windows):
  → ĐÃ FIX trong `capture_tv.py` bằng `page.set_content(...)` (không dùng `.as_uri()` nữa).

- **`playwright` thiếu browser**:
  → Chạy `playwright install chromium`.

- **Rate limit** khi fetch: chạy lại không kèm `--with-futures-extras` hoặc giảm `--limit`.

## 6) Dọn dependencies

Nếu bạn không tự vẽ chart, **không cần** `matplotlib`/`mplfinance`. File `requirements.txt` đã tối giản.

---

Happy trading! 🚀
