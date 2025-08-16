# glue_tick.py (rút gọn)
import time
from notifier import TelegramNotifier
from signal_tracker import SignalTracker
from price_watcher import PriceWatcher, BarCloseCallback

notifier = TelegramNotifier()
tracker = SignalTracker(notifier)
watcher = PriceWatcher()

# 1) Sau khi bạn đã post signal (hoặc dev thì giả lập)
from notifier import PostRef
sig = {
    "symbol": "AVAX/USDT", "side": "long",
    "entries": [24.70, 24.46], "stop": 23.90, "tps": [26.29, 27.29, 28.42],
    "leverage": "x5",
}
ref = PostRef(chat_id="console", message_id=1)        # dev: console
signal_id = "AVAX-12345"
tracker.register_post(signal_id, ref, sig, sl_mode="tick")  # hoặc "close_4h"

# 2) Callback tick: nối PriceWatcher -> SignalTracker
def on_tick(symbol: str, price: float, ts: str):
    st = tracker.get_state_by_symbol(symbol)
    if not st or not st.is_open():
        return
    side = st.side()
    entries = st.entries()
    stop = st.current_sl if st.current_sl is not None else st.stop()
    tps = st.tps()

    # Entry fill
    if side == "long":
        for i, e in enumerate(entries):
            if (not st.entry_filled(i)) and price <= e:
                tracker.on_entry_fill(st.signal_id, price, ts); break
        # TPs
        for i, tp in enumerate(tps):
            if (not st.tp_hit_flag(i)) and price >= tp:
                tracker.on_tp_hit(st.signal_id, i, price, ts)
        # SL (tick-mode)
        if st.sl_mode != "close_4h" and price <= stop:
            tracker.on_stopped_or_completed(st.signal_id, status="SL", price=stop, ts=ts)
    else:  # short
        for i, e in enumerate(entries):
            if (not st.entry_filled(i)) and price >= e:
                tracker.on_entry_fill(st.signal_id, price, ts); break
        for i, tp in enumerate(tps):
            if (not st.tp_hit_flag(i)) and price <= tp:
                tracker.on_tp_hit(st.signal_id, i, price, ts)
        if st.sl_mode != "close_4h" and price >= stop:
            tracker.on_stopped_or_completed(st.signal_id, status="SL", price=stop, ts=ts)

watcher.subscribe_tick("AVAX/USDT", on_tick)

# 3) (Tuỳ chọn) SL conservative: check khi đóng nến 4H
def on_bar_close_4h(symbol: str, timeframe: str, bar: dict):
    st = tracker.get_state_by_symbol(symbol)
    if not st or not st.is_open() or st.sl_mode != "close_4h":
        return
    close = float(bar.get("close"))
    sl = st.current_sl if st.current_sl is not None else st.stop()
    if st.side() == "long":
        if close <= sl:
            tracker.on_stopped_or_completed(st.signal_id, status="SL", price=close, ts=bar.get("ts"))
    else:
        if close >= sl:
            tracker.on_stopped_or_completed(st.signal_id, status="SL", price=close, ts=bar.get("ts"))

watcher.subscribe_bar_close("AVAX/USDT", "4H", on_bar_close_4h)

# 4) Giả tick để test
for p in [24.80, 24.70, 24.55, 25.10, 26.30, 27.30, 28.50]:
    watcher.emit_tick("AVAX/USDT", p)
    time.sleep(0.2)
# đóng nến 4H giả (nếu dùng close_4h cho SL)
watcher.emit_bar_close("AVAX/USDT", "4H", {"close": 23.85, "ts": "2025-08-15T12:00:00Z"})
