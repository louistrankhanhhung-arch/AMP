# price_watcher.py
from __future__ import annotations
from typing import Callable, Dict, List, Tuple, DefaultDict
from collections import defaultdict
from datetime import datetime

TickCallback = Callable[[str, float, str], None]             # (symbol, price, ts)
BarCloseCallback = Callable[[str, str, dict], None]          # (symbol, timeframe, bar_dict)

class PriceWatcher:
    """
    Event-bus đơn giản cho tick và close-bar.
    - subscribe_tick(symbol, cb): đăng ký nhận tick cho 1 symbol
    - emit_tick(symbol, price, ts=None): bắn tick (gọi các cb)
    - subscribe_bar_close(symbol, timeframe, cb): đăng ký nhận sự kiện đóng nến
    - emit_bar_close(symbol, timeframe, bar): bắn sự kiện close bar
    """
    def __init__(self):
        self._tick_subs: DefaultDict[str, List[TickCallback]] = defaultdict(list)
        self._bar_subs: DefaultDict[Tuple[str, str], List[BarCloseCallback]] = defaultdict(list)

    # ------- tick -------
    def subscribe_tick(self, symbol: str, cb: TickCallback):
        self._tick_subs[symbol].append(cb)

    def unsubscribe_tick(self, symbol: str, cb: TickCallback):
        if cb in self._tick_subs.get(symbol, []):
            self._tick_subs[symbol].remove(cb)

    def emit_tick(self, symbol: str, price: float, ts: str | None = None):
        ts = ts or (datetime.utcnow().isoformat() + "Z")
        for cb in list(self._tick_subs.get(symbol, [])):
            try:
                cb(symbol, float(price), ts)
            except Exception as e:
                print(f"[watcher] tick cb error {symbol}: {e}")

    # ------- bar close -------
    def subscribe_bar_close(self, symbol: str, timeframe: str, cb: BarCloseCallback):
        key = (symbol, timeframe.upper())
        self._bar_subs[key].append(cb)

    def unsubscribe_bar_close(self, symbol: str, timeframe: str, cb: BarCloseCallback):
        key = (symbol, timeframe.upper())
        if cb in self._bar_subs.get(key, []):
            self._bar_subs[key].remove(cb)

    def emit_bar_close(self, symbol: str, timeframe: str, bar: dict):
        """
        bar ví dụ: {'open':..., 'high':..., 'low':..., 'close':..., 'ts':'...'}
        Dùng để kiểm tra SL conservative 'đóng 4H'.
        """
        key = (symbol, timeframe.upper())
        for cb in list(self._bar_subs.get(key, [])):
            try:
                cb(symbol, key[1], bar)
            except Exception as e:
                print(f"[watcher] bar-close cb error {symbol}/{timeframe}: {e}")
