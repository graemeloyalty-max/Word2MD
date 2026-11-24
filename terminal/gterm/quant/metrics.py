"""Performance statistics for an equity curve (value over time).

- CAGR: compound annual growth rate = (end / start) ^ (1 / years) - 1.
- Volatility: standard deviation of daily returns x sqrt(252).
- Sharpe: (CAGR - risk-free) / volatility. Return per unit of total risk.
- Sortino: like Sharpe, but only downside volatility (days below 0) counts as risk.
- Max drawdown: worst peak-to-trough fall.
- Calmar: CAGR / |max drawdown|. Return per unit of worst pain.
- Hit rate: share of calendar months with a positive return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS = 252


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def volatility(equity: pd.Series) -> float:
    r = equity.pct_change().dropna()
    return float(r.std(ddof=1) * np.sqrt(DAYS)) if len(r) > 1 else float("nan")


def downside_vol(equity: pd.Series) -> float:
    r = equity.pct_change().dropna()
    neg = np.minimum(r, 0)
    return float(np.sqrt((neg ** 2).mean()) * np.sqrt(DAYS)) if len(r) > 1 else float("nan")


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def summary(equity: pd.Series, risk_free: float = 0.0) -> dict:
    c, v, dv, dd = cagr(equity), volatility(equity), downside_vol(equity), max_drawdown(equity)
    monthly = equity.resample("ME").last().pct_change().dropna()
    return {
        "CAGR": c,
        "Volatility": v,
        "Sharpe": (c - risk_free) / v if v and v > 0 else float("nan"),
        "Sortino": (c - risk_free) / dv if dv and dv > 0 else float("nan"),
        "Max drawdown": dd,
        "Calmar": c / abs(dd) if dd < 0 else float("nan"),
        "Hit rate (months)": float((monthly > 0).mean()) if len(monthly) else float("nan"),
        "Total return": float(equity.iloc[-1] / equity.iloc[0] - 1),
    }
