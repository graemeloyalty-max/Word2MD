"""Factor research: does a characteristic predict next month's returns across a universe?

Factors (higher score = expected better):
- **Momentum (12-1)**: return over the past 12 months, skipping the latest month.
- **Low volatility**: minus the annualised volatility of the past 6 months.
- **Value (earnings yield)**: latest *published* annual net income / market value.
- **Quality (ROE)**: latest *published* annual net income / shareholders' equity.

**Point-in-time rule.** Annual results are only treated as known `report_lag_days` after the
fiscal year end (default 90 days, when most annual reports are out). Using them earlier
would be look-ahead bias: the backtest would "know" earnings before investors did.

**Known limitations of free data, shown on the page:**
- *Survivorship bias*: the universe is today's companies. Firms that went bust or were delisted
  are missing, which flatters results.
- *Short fundamental history*: yfinance holds ~4 annual statements, so value/quality can only be
  tested over roughly the last 3 years. Momentum and low-vol use prices and go back further.
- Market value uses today's share count throughout (historical share counts aren't available).

Measures:
- **Information coefficient (IC)**: at each month end, the rank correlation (Spearman) between
  factor scores and the following month's returns. Mean IC above ~0.03-0.05 with a t-stat above 2
  is usually considered meaningful.
- **Quantile returns**: sort stocks into groups (quintiles) by score each month and average each
  group's next-month return. A working factor shows returns rising from the bottom to the top group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategies import rebalance_dates

FACTORS = ["Momentum (12-1)", "Low volatility", "Value (earnings yield)", "Quality (ROE)"]

DEFAULT_UNIVERSE = [
    # US
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "BAC", "JNJ", "PFE", "XOM", "CVX", "PG", "KO", "WMT",
    "HD", "CAT", "GE", "NEE", "LIN",
    # Europe
    "SAP.DE", "ASML.AS", "SIE.DE", "ALV.DE", "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AIR.PA", "NESN.SW", "NOVN.SW",
    "SHEL.L", "AZN.L", "ULVR.L", "HSBA.L", "IBE.MC", "ENEL.MI", "NOVO-B.CO", "INGA.AS",
]


def momentum_scores(prices: pd.DataFrame, dates, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    monthly = prices.resample("ME").last()
    rows = {}
    for d in dates:
        hist = monthly[monthly.index <= d]
        if len(hist) < lookback + skip + 1:
            continue
        rows[d] = hist.iloc[-1 - skip] / hist.iloc[-1 - skip - lookback] - 1
    return pd.DataFrame(rows).T


def low_vol_scores(prices: pd.DataFrame, dates, window: int = 126) -> pd.DataFrame:
    vol = prices.pct_change().rolling(window, min_periods=int(window * 0.8)).std() * np.sqrt(252)
    return -vol.reindex(pd.DatetimeIndex(dates), method="ffill")


def point_in_time(annual: pd.Series, dates, report_lag_days: int = 90) -> pd.Series:
    """Value known on each date: the latest fiscal year whose end + lag is on or before that date."""
    if annual is None or annual.dropna().empty:
        return pd.Series(np.nan, index=pd.DatetimeIndex(dates))
    known = annual.dropna().copy()
    known.index = pd.DatetimeIndex(known.index) + pd.Timedelta(days=report_lag_days)
    return known.sort_index().reindex(pd.DatetimeIndex(dates), method="ffill")


def value_scores(prices_base: pd.DataFrame, net_income_base: dict, shares: dict, dates, lag: int = 90) -> pd.DataFrame:
    """Earnings yield = point-in-time net income (base currency) / (price in base currency x shares)."""
    out = {}
    for sym in prices_base.columns:
        ni = point_in_time(net_income_base.get(sym), dates, lag)
        px = prices_base[sym].reindex(pd.DatetimeIndex(dates), method="ffill")
        sh = shares.get(sym)
        out[sym] = ni / (px * sh) if sh else pd.Series(np.nan, index=pd.DatetimeIndex(dates))
    return pd.DataFrame(out)


def quality_scores(net_income: dict, equity: dict, symbols, dates, lag: int = 90) -> pd.DataFrame:
    out = {}
    for sym in symbols:
        ni, eq = point_in_time(net_income.get(sym), dates, lag), point_in_time(equity.get(sym), dates, lag)
        out[sym] = (ni / eq).where(eq > 0)
    return pd.DataFrame(out)


def forward_returns(prices: pd.DataFrame, dates) -> pd.DataFrame:
    """Return from each date to the next date in `dates`."""
    px = prices.reindex(pd.DatetimeIndex(dates), method="ffill")
    return px.shift(-1) / px - 1


def information_coefficient(scores: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 5) -> pd.Series:
    ics = {}
    for d in scores.index.intersection(fwd.index):
        s, r = scores.loc[d], fwd.loc[d]
        ok = s.notna() & r.notna()
        if ok.sum() >= min_names:
            ics[d] = s[ok].rank().corr(r[ok].rank())  # Spearman = Pearson on ranks
    return pd.Series(ics, dtype=float)


def ic_summary(ic: pd.Series) -> dict:
    n = len(ic)
    mean, sd = ic.mean(), ic.std(ddof=1)
    return {"Mean IC": mean, "IC std": sd, "t-stat": mean / sd * np.sqrt(n) if n > 1 and sd > 0 else np.nan,
            "Share of months IC > 0": (ic > 0).mean() if n else np.nan, "Months": n}


def quantile_returns(scores: pd.DataFrame, fwd: pd.DataFrame, n_q: int = 5) -> pd.DataFrame:
    """Average forward return of each score quantile per date (columns Q1 = lowest score ... Qn = highest)."""
    rows = {}
    for d in scores.index.intersection(fwd.index):
        s, r = scores.loc[d], fwd.loc[d]
        ok = s.notna() & r.notna()
        if ok.sum() < n_q * 2:
            continue
        q = pd.qcut(s[ok].rank(method="first"), n_q, labels=[f"Q{i}" for i in range(1, n_q + 1)])
        rows[d] = r[ok].groupby(q, observed=False).mean()
    return pd.DataFrame(rows).T
