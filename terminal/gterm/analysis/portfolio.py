"""Portfolio analytics. Pure functions on price frames, so they're easy to test.

Plain English:
- **Holdings are held constant** over the chosen period (buy-and-hold of today's quantities).
  This answers "how has the portfolio I own now behaved?", not "what did my past trading earn?".
- **Everything is converted to one base currency first** (ZAR, USD or EUR) at each day's FX
  rate, so a rand-based investor sees rand returns, including the currency effect.
- **Different exchanges trade on different days.** Prices are lined up on the union of trading
  days and each price is carried forward over another market's holidays (never backward).

Measures:
- *Return*: end value / start value - 1.
- *Volatility*: standard deviation of daily returns x sqrt(252).
- *Beta*: covariance(portfolio, benchmark) / variance(benchmark), on daily returns in the same
  currency. 1.0 = moves with the benchmark; 1.2 = moves 20% more.
- *Max drawdown*: worst fall from a previous peak.
- *Sharpe ratio*: (annualised return - risk-free) / volatility.
- *Correlation*: how alike two holdings' daily returns are, from -1 (opposite) to +1 (identical).
- *Attribution*: each holding's contribution = its value change / the portfolio's starting value.
  Contributions add up exactly to the portfolio return, so you can sum them by sector or region.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def align(prices: dict[str, pd.Series]) -> pd.DataFrame:
    """Union of dates, forward-filled; rows before every series has started are dropped."""
    df = pd.DataFrame(prices).sort_index().ffill()
    return df.dropna(how="any")


def position_values(prices: pd.DataFrame, quantities: dict[str, float]) -> pd.DataFrame:
    q = pd.Series(quantities).reindex(prices.columns).fillna(0)
    return prices * q


def daily_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()


def annualised_return(values: pd.Series) -> float:
    days = (values.index[-1] - values.index[0]).days
    if days <= 0 or values.iloc[0] <= 0:
        return float("nan")
    return float((values.iloc[-1] / values.iloc[0]) ** (365.25 / days) - 1)


def volatility(values: pd.Series) -> float:
    r = daily_returns(values)
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else float("nan")


def beta(values: pd.Series, benchmark: pd.Series) -> float:
    both = pd.concat([daily_returns(values), daily_returns(benchmark)], axis=1, join="inner").dropna()
    if len(both) < 20:
        return float("nan")
    var = both.iloc[:, 1].var(ddof=1)
    return float(both.cov(ddof=1).iloc[0, 1] / var) if var > 0 else float("nan")


def drawdown(values: pd.Series) -> pd.Series:
    return values / values.cummax() - 1


def sharpe(values: pd.Series, risk_free: float = 0.0) -> float:
    vol = volatility(values)
    return float((annualised_return(values) - risk_free) / vol) if vol and vol > 0 else float("nan")


def correlation(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna().corr()


def contributions(pos_values: pd.DataFrame) -> pd.Series:
    """Each holding's value change divided by the portfolio's starting value. Sums to the total return."""
    start_total = pos_values.iloc[0].sum()
    return (pos_values.iloc[-1] - pos_values.iloc[0]) / start_total


def attribution(pos_values: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    """Start weight, end weight and return contribution, summed by group (e.g. sector or region)."""
    start_w = pos_values.iloc[0] / pos_values.iloc[0].sum()
    end_w = pos_values.iloc[-1] / pos_values.iloc[-1].sum()
    df = pd.DataFrame({"group": pd.Series(groups).reindex(pos_values.columns).fillna("Other"),
                       "start_weight": start_w, "end_weight": end_w, "contribution": contributions(pos_values)})
    out = df.groupby("group")[["start_weight", "end_weight", "contribution"]].sum()
    return out.sort_values("contribution", ascending=False)


# Yahoo exchange suffix -> country (used when the company profile lookup fails)
SUFFIX_COUNTRY = {
    "L": "United Kingdom", "DE": "Germany", "F": "Germany", "PA": "France", "AS": "Netherlands", "SW": "Switzerland",
    "MI": "Italy", "MC": "Spain", "BR": "Belgium", "LS": "Portugal", "VI": "Austria", "IR": "Ireland",
    "ST": "Sweden", "CO": "Denmark", "OL": "Norway", "HE": "Finland", "JO": "South Africa", "TO": "Canada",
    "AX": "Australia", "T": "Japan", "HK": "Hong Kong", "SA": "Brazil", "NS": "India", "BO": "India",
}


def country_from_symbol(symbol: str) -> str:
    if "." not in symbol:
        return "United States" if not symbol.startswith("^") and "=" not in symbol else ""
    return SUFFIX_COUNTRY.get(symbol.rsplit(".", 1)[1].upper(), "")


def region_for(country: str, symbol: str, asset_type: str = "EQUITY") -> str:
    from .multiples import EUROPE

    if asset_type in ("CRYPTOCURRENCY",) or symbol.endswith("-USD"):
        return "Crypto"
    if country == "United States":
        return "US"
    if country in EUROPE:
        return "Europe"
    if country in ("South Africa", "Brazil", "India", "China", "Mexico", "Indonesia", "Turkey", "Saudi Arabia"):
        return "Emerging markets"
    return "Other"
