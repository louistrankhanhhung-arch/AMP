import pandas as pd
import numpy as np

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower, (upper - lower) / mid * 100

def rolling_zscore(series: pd.Series, window=20):
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / sigma.replace(0, np.nan)

def calc_vp(df: pd.DataFrame, bins: int = 20):
    price_bins = np.linspace(df['low'].min(), df['high'].max(), bins)
    vol_profile = []
    for i in range(len(price_bins)-1):
        mask = (df['close'] >= price_bins[i]) & (df['close'] < price_bins[i+1])
        vol_profile.append({
            "price_range": (price_bins[i], price_bins[i+1]),
            "volume_sum": df.loc[mask, 'volume'].sum()
        })
    return sorted(vol_profile, key=lambda x: -x["volume_sum"])[:5]  # top 5 zone

def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['ema20'] = ema(out['close'], 20)
    out['ema50'] = ema(out['close'], 50)
    ub, mid, lb, width_pct = bollinger(out['close'], 20, 2.0)
    out['bb_upper'], out['bb_mid'], out['bb_lower'], out['bb_width_pct'] = ub, mid, lb, width_pct
    out['rsi14'] = rsi(out['close'], 14)
    out['atr14'] = atr(out, 14)
    out['vol_sma20'] = sma(out['volume'], 20)
    out['vol_ratio'] = out['volume'] / out['vol_sma20'].replace(0, np.nan)
    out['dist_to_ema20_pct'] = (out['close'] - out['ema20']) / out['ema20'] * 100
    out['dist_to_ema50_pct'] = (out['close'] - out['ema50']) / out['ema50'] * 100
    return out

def enrich_more(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Volume features
    out['vol_z20'] = rolling_zscore(out['volume'], 20)
    out['vol_up']  = (out['close'] > out['open']).astype(int)   # nến xanh
    out['vol_dn']  = (out['close'] < out['open']).astype(int)   # nến đỏ
    # Candle anatomy
    body = (out['close'] - out['open']).abs()
    rng  = (out['high'] - out['low']).replace(0, np.nan)
    out['body_pct'] = (body / rng * 100).clip(0, 100)
    out['upper_wick_pct'] = ((out['high'] - out[['open','close']].max(axis=1)) / rng * 100).clip(lower=0)
    out['lower_wick_pct'] = (((out[['open','close']].min(axis=1) - out['low']) / rng) * 100).clip(lower=0)

    # Basic soft SR components (để structure_engine gom lại thành SR mềm)
    # đã có ema20/ema50, bb_mid/upper/lower từ enrich_indicators()
    out['sma20'] = out['close'].rolling(20).mean()
    out['sma50'] = out['close'].rolling(50).mean()
    return out
