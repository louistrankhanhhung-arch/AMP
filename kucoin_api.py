
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


def _to_dataframe(ohlcv: List[List[float]]) -> pd.DataFrame:
    """
    Convert raw OHLCV to a pandas DataFrame with UTC datetime index.
    ccxt format: [ timestamp(ms), open, high, low, close, volume ]
    """
    if not ohlcv:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("time")[["open","high","low","close","volume"]]
    return df


def fetch_ohlcv(symbol: str,
                timeframe: str = "4H",
                limit: int = 100,
                since_ms: Optional[int] = None,
                kucoin_key: Optional[str] = None,
                kucoin_secret: Optional[str] = None,
                kucoin_passphrase: Optional[str] = None,
                max_retries: int = 3,
                retry_wait: float = 1.5) -> pd.DataFrame:
    """
    Fetch real OHLCV from KuCoin using ccxt.
    - symbol: e.g. 'BTCUSDT' or 'BTC/USDT'
    - timeframe: '4H', '1D', etc. (mapped internally to ccxt timeframes)
    - limit: number of candles (KuCoin typically supports up to 1500)
    - since_ms: optional starting timestamp in milliseconds (UTC)
    """
    tf = TIMEFRAME_MAP.get(timeframe.upper())
    if tf is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Use one of {list(TIMEFRAME_MAP.keys())}.")

    norm_symbol = _normalize_symbol(symbol)
    ex = _exchange(kucoin_key, kucoin_secret, kucoin_passphrase)

    err: Optional[Exception] = None
    for i in range(max_retries):
        try:
            data = ex.fetch_ohlcv(norm_symbol, timeframe=tf, since=since_ms, limit=limit)
            return _to_dataframe(data)
        except Exception as e:
            err = e
            time.sleep(retry_wait * (i + 1))
    # After retries failed:
    raise RuntimeError(f"KuCoin fetch_ohlcv failed for {symbol} {timeframe}: {err}")


def fetch_batch(symbol: str,
                timeframes: Iterable[str] = ("4H","1D"),
                limit: int = 100,
                since_ms: Optional[int] = None,
                kucoin_key: Optional[str] = None,
                kucoin_secret: Optional[str] = None,
                kucoin_passphrase: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Fetch multiple timeframes at once. Returns dict mapping tf -> DataFrame.
    """
    out: Dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        out[tf] = fetch_ohlcv(symbol,
                              timeframe=tf,
                              limit=limit,
                              since_ms=since_ms,
                              kucoin_key=kucoin_key,
                              kucoin_secret=kucoin_secret,
                              kucoin_passphrase=kucoin_passphrase)
    return out
