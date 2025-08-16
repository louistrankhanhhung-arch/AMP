# glue_tick.py
from typing import Callable
from price_watcher import PriceWatcher
from signal_tracker import SignalTracker

def wire_symbol(watcher: PriceWatcher, tracker: SignalTracker, symbol: str) -> None:
    """
    Nối PriceWatcher -> SignalTracker cho 1 symbol.
    - Tick: kiểm tra fill/TP/SL theo side & trạng thái hiện tại
    - 4H close: nếu sl_mode == 'close_4h' thì dừng lệnh khi đóng nến chạm SL
    """
    # --- tick callback ---
    def on_tick(sym: str, price: float, ts: str):
        if sym != symbol:
            return
        st = tracker.get_state_by_symbol(sym)
        if not st or not st.is_open():
            return

        side    = st.side()
        entries = st.entries()
        stop    = st.current_sl if st.current_sl is not None else st.stop()
        tps     = st.tps()

        # Entry fill
        if side == "long":
            for i, e in enumerate(entries):
                if not st.entry_filled(i) and price <= e:
                    tracker.on_entry_fill(st.signal_id, price, ts)
                    break
            # TP
            for i, tp in enumerate(tps):
                if not st.tp_hit_flag(i) and price >= tp:
                    tracker.on_tp_hit(st.signal_id, i, price, ts)
            # SL (tick-mode)
            if st.sl_mode != "close_4h" and price <= stop:
                tracker.on_stopped_or_completed(st.signal_id, status="SL", price=stop, ts=ts)
        else:  # short
            for i, e in enumerate(entries):
                if not st.entry_filled(i) and price >= e:
                    tracker.on_entry_fill(st.signal_id, price, ts)
                    break
            for i, tp in enumerate(tps):
                if not st.tp_hit_flag(i) and price <= tp:
                    tracker.on_tp_hit(st.signal_id, i, price, ts)
            if st.sl_mode != "close_4h" and price >= stop:
                tracker.on_stopped_or_completed(st.signal_id, status="SL", price=stop, ts=ts)

    # --- 4H bar-close callback (cho sl_mode='close_4h') ---
    def on_bar_close(sym: str, timeframe: str, bar: dict):
        if sym != symbol or timeframe != "4H":
            return
        st = tracker.get_state_by_symbol(sym)
        if not st or not st.is_open() or st.sl_mode != "close_4h":
            return
        close = float(bar.get("close"))
        ts    = bar.get("ts")
        sl    = st.current_sl if st.current_sl is not None else st.stop()
        if st.side() == "long":
            if close <= sl:
                tracker.on_stopped_or_completed(st.signal_id, status="SL", price=close, ts=ts)
        else:
            if close >= sl:
                tracker.on_stopped_or_completed(st.signal_id, status="SL", price=close, ts=ts)

    # Đăng ký vào watcher
    watcher.subscribe_tick(symbol, on_tick)
    watcher.subscribe_bar_close(symbol, "4H", on_bar_close)


# -------------------- DEMO (không chạy trong production) --------------------
if __name__ == "__main__":
    import time
    from notifier import TelegramNotifier, PostRef
    from price_watcher import PriceWatcher

    notifier = TelegramNotifier()
    tracker  = SignalTracker(notifier)
    watcher  = PriceWatcher()

    # ví dụ mở 1 lệnh để test
    signal_id = "AVAX-12345"
    tracker.register_post(
        signal_id,
        PostRef(chat_id="console", message_id=1),  # dev: in ra console
        {
            "symbol": "AVAX/USDT",
            "side": "long",
            "entries": [24.70, 24.46],
            "stop": 23.90,
            "tps": [26.29, 27.29, 28.42],
            "leverage": "x5",
        },
        sl_mode="tick"
    )

    # nối dây cho AVAX/USDT
    wire_symbol(watcher, tracker, "AVAX/USDT")

    # giả tick & bar-close
    for p in [24.80, 24.70, 24.55, 25.10, 26.30, 27.30, 28.50]:
        watcher.emit_tick("AVAX/USDT", p); time.sleep(0.2)
    watcher.emit_bar_close("AVAX/USDT", "4H", {"close": 23.85, "ts": "2025-08-15T12:00:00Z"})
