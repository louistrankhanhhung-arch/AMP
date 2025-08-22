# app/kucoin_api.py
import time
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import ccxt


TIMEFRAME_MAP = {
    "1H": "1h",
    "4H": "4h",
    "1D": "1d",
    "1W": "1w",
}


def _normalize_symbol(symbol: str) -> str:
    """
    Convert common forms like 'BTCUSDT' or 'btc/usdt' into 'BTC/USDT' for ccxt.
    """
    s = symbol.strip().upper()
    if "/" in s:
        base, quote = s.split("/", 1)
        return f"{base}/{quote}"
    if s.endswith("USDT"):
        base = s[:-4]
        return f"{base}/USDT"
    if s.endswith("USD"):
        base = s[:-3]
        return f"{base}/USD"
    # Fallback (let ccxt validate)
    return s


def _exchange(kucoin_key: Optional[str] = None,
              kucoin_secret: Optional[str] = None,
              kucoin_passphrase: Optional[str] = None) -> ccxt.kucoin:
    """
    Create a ccxt KuCoin exchange instance. Public data does not require keys.
    """
    cfg = {
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    }
    if kucoin_key and kucoin_secret and kucoin_passphrase:
        cfg.update({
            "apiKey": kucoin_key,
            "secret": kucoin_secret,
            "password": kucoin_passphrase,
        })
    ex = ccxt.kucoin(cfg)
    ex.load_markets()
    return ex

def _to_dataframe(ohlcv):
    """
    ccxt OHLCV: [ts_ms, open, high, low, close, volume]
    """
    if not ohlcv:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = (df.set_index("time")[["open","high","low","close","volume"]]
            .astype(float)
            .sort_index())          # BẮT BUỘC: đảm bảo ASC (cũ → mới)
    df = df[~df.index.duplicated(keep="last")]  # bỏ nến trùng
    return df

def fetch_ohlcv(symbol: str, timeframe: str="4H", limit: int=300, since_ms=None,
                kucoin_key=None, kucoin_secret=None, kucoin_passphrase=None) -> pd.DataFrame:
    tf = TIMEFRAME_MAP.get(timeframe.upper(), timeframe)
    ex = ccxt.kucoin({
        "apiKey": kucoin_key or "",
        "secret": kucoin_secret or "",
        "password": kucoin_passphrase or "",
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}
    })
    ex.load_markets()
    sym = symbol.replace("-", "/").replace("USDTUSDT","USDT")
    if "/" not in sym:
        sym = sym[:-4] + "/" + sym[-4:]          # DOTUSDT -> DOT/USDT

    raw = ex.fetch_ohlcv(sym, timeframe=tf, since=since_ms, limit=limit)
    df  = _to_dataframe(raw)

    # (khuyên dùng) bỏ nến đang chạy: giữ nguyên nếu bạn cần “realtime”
    # bar_ms = ex.parse_timeframe(tf) * 1000
    # if len(df) >= 2:
    #     if (int(df.index[-1].value/1e6) % int(bar_ms)) != 0:
    #         df = df.iloc[:-1]

    return df

def fetch_batch(symbol: str, timeframes=("1H","4H","1D"), limit=300, since_ms=None, **keys):
    return {tf: fetch_ohlcv(symbol, timeframe=tf, limit=limit, since_ms=since_ms, **keys)
            for tf in timeframes}

