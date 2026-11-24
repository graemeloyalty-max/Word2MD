"""Technical indicators. Pure functions on pandas Series, tested in tests/test_indicators.py."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average: mean of the last `window` closes."""
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    """Exponential moving average with smoothing 2/(window+1)."""
    return close.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (0-100). Above 70 is 'overbought', below 30 'oversold'."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line (fast EMA - slow EMA), signal line (EMA of MACD) and histogram."""
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Bollinger bands: SMA +/- n_std rolling standard deviations."""
    mid = sma(close, window)
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + n_std * sd, "lower": mid - n_std * sd})


def total_return(close: pd.Series) -> float:
    c = close.dropna()
    return float(c.iloc[-1] / c.iloc[0] - 1) if len(c) > 1 else 0.0


def annualised_volatility(close: pd.Series, periods_per_year: int = 252) -> float:
    r = np.log(close.dropna()).diff().dropna()
    return float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0


def max_drawdown(close: pd.Series) -> float:
    """Largest peak-to-trough fall, as a negative fraction."""
    c = close.dropna()
    return float((c / c.cummax() - 1).min()) if len(c) else 0.0
