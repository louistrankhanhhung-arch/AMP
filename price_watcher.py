from typing import Callable, Dict, Any

class PriceWatcher:
    def __init__(self):
        self.listeners = []  # (symbol, callback)

    def subscribe(self, symbol: str, callback: Callable[[Dict[str,Any]], None]):
        self.listeners.append((symbol, callback))

    def emit_tick(self, symbol: str, price: float):
        for sym, cb in self.listeners:
            if sym == symbol:
                cb({"symbol": symbol, "price": price})
