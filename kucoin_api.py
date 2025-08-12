import pandas as pd
from datetime import datetime
from typing import List

# TODO: replace with real KuCoin REST/WS calls.
def fetch_ohlcv(symbol: str, timeframe: str = "4H", limit: int = 300) -> pd.DataFrame:
    """Return dummy OHLCV DataFrame with expected columns.
    Replace this with real API fetch + resample.
    """
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq="4H" if timeframe=="4H" else "1D")
    df = pd.DataFrame({
        "open": 100 + pd.Series(range(limit)).rolling(5, min_periods=1).mean(),
        "high": 100 + pd.Series(range(limit)).rolling(5, min_periods=1).mean() + 2,
        "low":  100 + pd.Series(range(limit)).rolling(5, min_periods=1).mean() - 2,
        "close":100 + pd.Series(range(limit)).rolling(5, min_periods=1).mean() + 1,
        "volume": 1_000_000
    }, index=idx)
    df.index.name = "time"
    return df
