"""Strategies. Each returns a DataFrame of target weights indexed by DECISION date.

Decisions only use prices up to and including the decision date; the backtester then
trades them one day later. All are long-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """Last trading day of each period: 'W' weekly, 'M' monthly, 'Q' quarterly."""
    s = pd.Series(index, index=index)
    key = {"W": index.to_period("W"), "M": index.to_period("M"), "Q": index.to_period("Q")}[freq]
    return pd.DatetimeIndex(s.groupby(key).max().values)


def buy_and_hold(prices: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    w = pd.Series(weights) if weights else pd.Series(1 / prices.shape[1], index=prices.columns)
    return pd.DataFrame([w.reindex(prices.columns).fillna(0)], index=[prices.index[0]])


def equal_weight(prices: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Equal weights, reset at every rebalance date (vs buy-and-hold, which lets winners grow)."""
    dates = rebalance_dates(prices.index, freq)
    return pd.DataFrame(1 / prices.shape[1], index=dates, columns=prices.columns)


def sma_crossover(prices: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Per asset: fully invested (equal share) when the fast moving average is above the slow one, else cash."""
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")
    f, s = prices.rolling(fast).mean(), prices.rolling(slow).mean()
    on = (f > s).astype(float).where(s.notna())
    w = on / prices.shape[1]
    w = w.dropna(how="all")
    changed = w.ne(w.shift()).any(axis=1)  # only record days where the signal changes
    return w[changed].fillna(0.0)


def momentum(prices: pd.DataFrame, lookback: int = 12, top_n: int = 5, skip: int = 1, freq: str = "M") -> pd.DataFrame:
    """Cross-sectional momentum: each month, hold the `top_n` assets with the best return over the past
    `lookback` months, skipping the most recent `skip` month(s) (short-term reversal), equal weight."""
    dates = rebalance_dates(prices.index, freq)
    rows = {}
    for d in dates:
        hist = prices.loc[:d].resample("ME").last()  # month-end prices known on date d
        if len(hist) < lookback + skip + 1:
            continue
        end, start = hist.iloc[-1 - skip], hist.iloc[-1 - skip - lookback]
        ret = (end / start - 1).dropna()
        if ret.empty:
            continue
        top = ret.nlargest(min(top_n, len(ret))).index
        rows[d] = pd.Series(1 / len(top), index=top)
    return pd.DataFrame(rows).T.reindex(columns=prices.columns).fillna(0.0)


def inverse_volatility(prices: pd.DataFrame, window: int = 63, freq: str = "M") -> pd.DataFrame:
    """Weights proportional to 1 / recent volatility: calmer assets get more capital (a simple risk-parity)."""
    vol = prices.pct_change().rolling(window).std()
    rows = {}
    for d in rebalance_dates(prices.index, freq):
        v = vol.loc[d].dropna()
        v = v[v > 0]
        if v.empty:
            continue
        inv = 1 / v
        rows[d] = inv / inv.sum()
    return pd.DataFrame(rows).T.reindex(columns=prices.columns).fillna(0.0)


def from_scores(scores: pd.DataFrame, top_fraction: float = 0.2, higher_is_better: bool = True) -> pd.DataFrame:
    """Hold the top `top_fraction` of assets by factor score at each date (equal weight)."""
    rows = {}
    for d, s in scores.iterrows():
        s = s.dropna()
        if len(s) < 3:
            continue
        k = max(1, int(round(len(s) * top_fraction)))
        top = (s.nlargest(k) if higher_is_better else s.nsmallest(k)).index
        rows[d] = pd.Series(1 / k, index=top)
    return pd.DataFrame(rows).T.reindex(columns=scores.columns).fillna(0.0)


STRATEGIES = {
    "Buy & hold (equal start)": buy_and_hold,
    "Equal weight, monthly rebalance": equal_weight,
    "Momentum rotation": momentum,
    "Trend following (SMA crossover)": sma_crossover,
    "Inverse volatility": inverse_volatility,
}
