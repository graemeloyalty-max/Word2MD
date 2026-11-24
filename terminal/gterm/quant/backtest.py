"""Backtesting engine.

Plain English: give it daily prices and a list of "on this date, I want these weights"
decisions. It simulates holding the portfolio day by day:

- **No look-ahead.** A decision made with data up to the close of day D is traded at the
  close of day D + `execution_lag` trading days (default 1). You cannot trade at the very
  price you used to make the decision.
- **Weights drift** between rebalances as prices move, like a real portfolio.
- **Transaction costs**: every trade costs `cost_bps` basis points of the value traded
  (turnover x cost). 10 bps = 0.10%. Turnover = sum of absolute weight changes.
- Any weight not invested (weights summing to less than 1) sits in cash earning 0%.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics


@dataclass
class BacktestResult:
    equity: pd.Series  # portfolio value, starts at 1.0
    returns: pd.Series  # daily returns after costs
    weights: pd.DataFrame  # weights right after each trade
    turnover: pd.Series  # turnover at each trade date
    costs: pd.Series  # cost (fraction of value) at each trade date
    name: str = "strategy"

    def summary(self, risk_free: float = 0.0) -> dict:
        s = metrics.summary(self.equity, risk_free)
        years = max((self.equity.index[-1] - self.equity.index[0]).days / 365.25, 1e-9)
        # Exclude the initial investment: turnover should measure ongoing trading, not buying in.
        s["Annual turnover"] = float(self.turnover.iloc[1:].sum() / years) if len(self.turnover) > 1 else 0.0
        s["Total costs"] = float(self.costs.sum())
        s["Trades"] = int((self.turnover > 1e-9).sum())
        return s


def run(prices: pd.DataFrame, decisions: pd.DataFrame, cost_bps: float = 10.0, execution_lag: int = 1,
        name: str = "strategy") -> BacktestResult:
    """prices: dates x assets (one currency). decisions: decision dates x assets, target weights."""
    prices = prices.sort_index().ffill()
    decisions = decisions.reindex(columns=prices.columns).fillna(0.0).sort_index()
    dates = prices.index
    rets = prices.pct_change().fillna(0.0).to_numpy()

    # Map each decision to its execution date: `execution_lag` trading days after the decision date.
    trades = {}
    for d, row in decisions.iterrows():
        pos = dates.searchsorted(d, side="right") - 1  # last trading day on or before the decision date
        exec_pos = max(pos, 0) + execution_lag
        if 0 <= exec_pos < len(dates):
            trades[exec_pos] = row.to_numpy(dtype=float)  # later decisions for the same day override earlier ones

    n = len(prices.columns)
    w = np.zeros(n)
    value = 1.0
    equity = np.empty(len(dates))
    daily = np.zeros(len(dates))
    w_hist, t_hist, c_hist, t_dates = [], [], [], []
    for i in range(len(dates)):
        start_value = value
        if i > 0 and w.any():
            port_r = float(w @ rets[i])
            value *= 1 + port_r
            grown = w * (1 + rets[i])
            w = grown / (1 + port_r) if (1 + port_r) != 0 else grown
        if i in trades:
            target = trades[i]
            turnover = float(np.abs(target - w).sum())
            cost = turnover * cost_bps / 1e4
            value *= 1 - cost
            w = target.copy()
            w_hist.append(w.copy())
            t_hist.append(turnover)
            c_hist.append(cost)
            t_dates.append(dates[i])
        equity[i] = value
        daily[i] = value / start_value - 1 if i > 0 else 0.0
    idx = pd.DatetimeIndex(t_dates)
    return BacktestResult(
        equity=pd.Series(equity, index=dates, name=name),
        returns=pd.Series(daily, index=dates, name=name),
        weights=pd.DataFrame(w_hist, index=idx, columns=prices.columns),
        turnover=pd.Series(t_hist, index=idx, dtype=float),
        costs=pd.Series(c_hist, index=idx, dtype=float),
        name=name,
    )


def sweep(prices: pd.DataFrame, strategy, grid: dict[str, list], metric: str = "Sharpe", cost_bps: float = 10.0,
          **fixed) -> pd.DataFrame:
    """Run `strategy(prices, **params)` for every combination of a 2-parameter grid; return a metric table."""
    (k1, v1), (k2, v2) = list(grid.items())
    table = pd.DataFrame(index=pd.Index(v1, name=k1), columns=pd.Index(v2, name=k2), dtype=float)
    for a in v1:
        for b in v2:
            try:
                dec = strategy(prices, **{k1: a, k2: b}, **fixed)
                table.loc[a, b] = run(prices, dec, cost_bps).summary()[metric] if len(dec) else np.nan
            except ValueError:
                table.loc[a, b] = np.nan
    return table
