"""Alternative investments tracked alongside equities.

Plain English: for each asset we show its return, volatility and worst fall, all converted to your
base currency, plus its **correlation with world equities**: close to +1 means it moves with
shares (little diversification), near 0 or negative means it can cushion an equity portfolio.

Notes on the instruments (free Yahoo data):
- **Commodities** are front-month futures (e.g. GC=F gold). The series jumps when one contract rolls
  into the next, so long-run returns differ from what a fund or a held position would earn.
- **Bonds** are shown through bond ETFs (price return, coupons excluded) and through yields.
  Yields are rates, not prices: they are listed separately and excluded from return statistics.
- **Crypto** trades every day; returns and correlations use weekdays only so they line up with markets.
- **REITs** are listed property companies (US, Europe, South Africa).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UNIVERSE = {
    "Crypto": {"BTC-USD": "Bitcoin", "ETH-USD": "Ether", "SOL-USD": "Solana", "XRP-USD": "XRP"},
    "Commodities": {"GC=F": "Gold", "SI=F": "Silver", "CL=F": "WTI crude oil", "BZ=F": "Brent crude oil", "NG=F": "Natural gas",
                    "HG=F": "Copper", "PL=F": "Platinum", "ZW=F": "Wheat", "ZC=F": "Corn"},
    "REITs": {"VNQ": "US REIT ETF (Vanguard)", "PLD": "Prologis (US)", "O": "Realty Income (US)", "VNA.DE": "Vonovia (DE)",
              "SGRO.L": "Segro (UK)", "GRT.JO": "Growthpoint (ZA)"},
    "Bonds (ETFs)": {"SHY": "US Treasury 1-3y", "IEF": "US Treasury 7-10y", "TLT": "US Treasury 20y+", "LQD": "US investment grade",
                     "HYG": "US high yield", "EMB": "EM USD bonds", "IBGL.L": "Euro govt bonds (iShares)"},
    "Equities (reference)": {"URTH": "MSCI World ETF", "^GSPC": "S&P 500", "^STOXX50E": "EURO STOXX 50"},
}
YIELDS = {"^IRX": "US 13-week T-bill", "^FVX": "US 5-year Treasury", "^TNX": "US 10-year Treasury", "^TYX": "US 30-year Treasury"}


def weekdays(df):
    return df[df.index.dayofweek < 5]


def summary(prices: pd.DataFrame, benchmark: pd.Series, labels: dict, asset_class: dict) -> pd.DataFrame:
    """One row per asset: returns over 1m/1y/period, annualised volatility, max drawdown, correlation with the benchmark."""
    px = weekdays(prices).ffill()
    bench_r = weekdays(benchmark.to_frame("b")).ffill()["b"].pct_change()
    rows = []
    for s in px.columns:
        c = px[s].dropna()
        if len(c) < 30:
            continue
        r = c.pct_change().dropna()
        last = c.index[-1]

        def ret(days):
            past = c[c.index <= last - pd.Timedelta(days=days)]
            return c.iloc[-1] / past.iloc[-1] - 1 if len(past) else np.nan

        both = pd.concat([r, bench_r], axis=1, join="inner").dropna()
        rows.append({"Symbol": s, "Name": labels.get(s, s), "Class": asset_class.get(s, ""), "Last": c.iloc[-1],
                     "1 month": ret(30), "1 year": ret(365), "Period": c.iloc[-1] / c.iloc[0] - 1,
                     "Volatility": r.std(ddof=1) * np.sqrt(252), "Max drawdown": (c / c.cummax() - 1).min(),
                     "Corr. with equities": both.iloc[:, 0].corr(both.iloc[:, 1]) if len(both) > 30 else np.nan})
    return pd.DataFrame(rows)


def correlation(prices: pd.DataFrame) -> pd.DataFrame:
    return weekdays(prices).ffill().pct_change().dropna(how="all").corr(min_periods=30)
