"""Offline demo source: deterministic synthetic prices.

Used when there is no internet (or Yahoo is blocked) so the app and tests still run.
Every number it produces is labelled source="demo (synthetic)" so it can never be
mistaken for real market data.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..core.statements import normalise
from .base import DataSource, DataSourceError, Fundamentals, KeyStats, Ownership, PriceHistory, Quote

# A small universe so the demo feels realistic. (name, currency, type, start price, exchange)
UNIVERSE = {
    "AAPL": ("Apple Inc.", "USD", "EQUITY", 180.0, "NASDAQ"),
    "MSFT": ("Microsoft Corporation", "USD", "EQUITY", 380.0, "NASDAQ"),
    "NVDA": ("NVIDIA Corporation", "USD", "EQUITY", 120.0, "NASDAQ"),
    "JPM": ("JPMorgan Chase & Co.", "USD", "EQUITY", 190.0, "NYSE"),
    "SAP.DE": ("SAP SE", "EUR", "EQUITY", 190.0, "XETRA"),
    "ASML.AS": ("ASML Holding N.V.", "EUR", "EQUITY", 700.0, "Euronext Amsterdam"),
    "MC.PA": ("LVMH Moet Hennessy Louis Vuitton", "EUR", "EQUITY", 720.0, "Euronext Paris"),
    "NESN.SW": ("Nestle S.A.", "CHF", "EQUITY", 90.0, "SIX Swiss"),
    "SHEL.L": ("Shell plc", "GBp", "EQUITY", 2600.0, "LSE"),
    "SPY": ("SPDR S&P 500 ETF Trust", "USD", "ETF", 500.0, "NYSE Arca"),
    "^GSPC": ("S&P 500", "USD", "INDEX", 5000.0, "SNP"),
    "^STOXX50E": ("EURO STOXX 50", "EUR", "INDEX", 4800.0, "STOXX"),
    "EURUSD=X": ("EUR/USD", "USD", "CURRENCY", 1.08, "CCY"),
    "USDZAR=X": ("USD/ZAR", "ZAR", "CURRENCY", 18.2, "CCY"),
    "BTC-USD": ("Bitcoin USD", "USD", "CRYPTOCURRENCY", 60000.0, "CCC"),
}

PERIOD_DAYS = {"1mo": 31, "3mo": 92, "6mo": 183, "1y": 365, "2y": 730, "5y": 1826, "10y": 3652, "max": 7300}
INTERVAL_RULE = {"1d": None, "1wk": "W-FRI", "1mo": "ME"}


def _seed(symbol: str) -> int:
    return int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16)


class DemoSource(DataSource):
    name = "demo (synthetic)"
    requires_key = False

    def __init__(self, today: pd.Timestamp | None = None):
        self.today = (today or pd.Timestamp.today()).normalize()

    def _meta(self, symbol: str):
        if symbol in UNIVERSE:
            return UNIVERSE[symbol]
        # Any other symbol still works, so the UI can be exercised freely.
        return (f"{symbol} (demo)", "USD", "EQUITY", 50.0 + _seed(symbol) % 200, "DEMO")

    def _daily(self, symbol: str) -> pd.DataFrame:
        """Full ~20y daily business-day history, deterministic per symbol."""
        name, ccy, qtype, start, _ = self._meta(symbol)
        rng = np.random.default_rng(_seed(symbol))
        dates = pd.bdate_range(end=self.today, periods=20 * 261)
        vol = {"CURRENCY": 0.006, "CRYPTOCURRENCY": 0.035, "INDEX": 0.011}.get(qtype, 0.017)
        drift = 0.0003
        rets = rng.normal(drift, vol, len(dates))
        # Geometric random walk, anchored so the latest close is near a realistic level.
        if qtype == "CURRENCY":
            rets = rets - drift  # no long-run drift for FX
        path = np.cumsum(rets)
        close = start * np.exp(path - path[-1])
        open_ = close * (1 + rng.normal(0, vol / 3, len(dates)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, len(dates))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, len(dates))))
        volume = rng.integers(1_000_000, 50_000_000, len(dates)) if qtype != "CURRENCY" else np.zeros(len(dates))
        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume.astype(float)},
            index=pd.DatetimeIndex(dates, name="Date"),
        )
        return df

    def get_prices(self, symbol: str, period: str = "1y", interval: str = "1d") -> PriceHistory:
        if not symbol:
            raise DataSourceError("Empty symbol")
        if period not in PERIOD_DAYS or interval not in INTERVAL_RULE:
            raise DataSourceError(f"Unsupported period/interval {period}/{interval}")
        df = self._daily(symbol)
        df = df[df.index > self.today - pd.Timedelta(days=PERIOD_DAYS[period])]
        rule = INTERVAL_RULE[interval]
        if rule:
            df = df.resample(rule).agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            ).dropna()
        ccy = self._meta(symbol)[1]
        return PriceHistory(symbol, df, ccy, self.name, datetime.now(timezone.utc))

    def get_quote(self, symbol: str) -> Quote:
        name, ccy, qtype, _, exch = self._meta(symbol)
        h = self.get_prices(symbol, "1mo", "1d")
        return Quote(
            symbol=symbol,
            name=name,
            price=h.last_close,
            previous_close=float(h.frame["Close"].iloc[-2]),
            currency=ccy,
            exchange=exch,
            asset_type=qtype,
            price_date=h.last_date,
            source=self.name,
            as_of=h.as_of,
        )

    def search(self, query: str) -> list[dict]:
        q = query.lower()
        return [
            {"symbol": s, "name": m[0], "exchange": m[4], "type": m[2]}
            for s, m in UNIVERSE.items()
            if q in s.lower() or q in m[0].lower()
        ]


# Sector / country for the demo equities (used by peers and the country overlay).
PROFILE = {
    "AAPL": ("Technology", "Consumer Electronics", "United States"),
    "MSFT": ("Technology", "Software - Infrastructure", "United States"),
    "NVDA": ("Technology", "Semiconductors", "United States"),
    "JPM": ("Financial Services", "Banks - Diversified", "United States"),
    "SAP.DE": ("Technology", "Software - Application", "Germany"),
    "ASML.AS": ("Technology", "Semiconductor Equipment & Materials", "Netherlands"),
    "MC.PA": ("Consumer Cyclical", "Luxury Goods", "France"),
    "NESN.SW": ("Consumer Defensive", "Packaged Foods", "Switzerland"),
    "SHEL.L": ("Energy", "Oil & Gas Integrated", "United Kingdom"),
}
IFRS_COUNTRIES = {"Germany", "Netherlands", "France", "Switzerland", "United Kingdom"}


def _demo_fundamentals(self, symbol: str) -> Fundamentals:
    name, ccy, qtype, _, _ = self._meta(symbol)
    if qtype != "EQUITY":
        raise DataSourceError(f"No financial statements for {symbol} ({qtype.lower()}).")
    sector, industry, country = PROFILE.get(symbol, ("Industrials", "Diversified", "United States"))
    rng = np.random.default_rng(_seed(symbol) + 7)
    price = self.get_quote(symbol).price
    major_div = 100.0 if ccy == "GBp" else 1.0
    fin_ccy = "GBP" if ccy == "GBp" else ccy
    shares = float(1e9 * (1 + _seed(symbol) % 5))
    mcap = price / major_div * shares
    years = pd.to_datetime([f"{self.today.year - k}-12-31" for k in range(4, 0, -1)])
    revenue = mcap / rng.uniform(3, 6) * np.cumprod(1 + rng.normal(0.07, 0.03, 4))
    margin = np.clip(rng.normal(0.22, 0.02, 4), 0.05, 0.5)
    ebit = revenue * margin
    dep = revenue * 0.04
    interest = mcap * 0.002
    pretax = ebit - interest
    tax = pretax * 0.22
    ni = pretax - tax
    equity = mcap / rng.uniform(3, 8) * np.linspace(0.9, 1.05, 4)
    debt = equity * rng.uniform(0.2, 0.6)
    cash = equity * rng.uniform(0.1, 0.3)
    cfo = ni + dep + revenue * rng.normal(0.01, 0.01, 4)
    capex = -(revenue * 0.05)
    ifrs = country in IFRS_COUNTRIES
    inc = {"Total Revenue": revenue, "Operating Income": ebit, "Reconciled Depreciation": dep, "Interest Expense": np.full(4, interest),
           "Pretax Income": pretax, "Tax Provision": tax, "Net Income Common Stockholders": ni,
           "Diluted Average Shares": np.full(4, shares), "Diluted EPS": ni / shares, "Gross Profit": revenue * 0.55}
    bal = {"Total Assets": equity + debt + revenue * 0.3, "Stockholders Equity": equity, "Total Debt": debt,
           "Cash Cash Equivalents And Short Term Investments": cash, "Ordinary Shares Number": np.full(4, shares),
           "Current Assets": cash + revenue * 0.2, "Current Liabilities": revenue * 0.18}
    cf = {"Operating Cash Flow": cfo, "Capital Expenditure": capex, "Cash Dividends Paid": -ni * 0.35,
          "Repurchase Of Capital Stock": -ni * 0.2, "Stock Based Compensation": revenue * 0.01}
    if ifrs:  # exercise the IFRS normalisation paths
        cf["Interest Paid Cff"] = np.full(4, -interest)
        cf["Purchase Of Intangibles"] = -(revenue * 0.01)
    frame = lambda d: pd.DataFrame(d, index=years).T  # noqa: E731
    stmts, notes = normalise(frame(inc), frame(bal), frame(cf))
    return Fundamentals(symbol, name, sector, industry, country, price, ccy, fin_ccy, shares,
                        round(float(rng.uniform(0.7, 1.4)), 2), stmts, notes, self.name, datetime.now(timezone.utc))


def _demo_key_stats(self, symbol: str) -> KeyStats:
    f = self.get_fundamentals(symbol)
    from ..core.statements import latest
    inc, bal = f.statements["income"], f.statements["balance"]
    div = 100.0 if f.price_currency == "GBp" else 1.0
    mcap = f.price / div * f.shares
    ev = mcap + latest(bal, "net_debt", 0.0)
    return KeyStats(symbol, f.name, f.sector, f.country, f.price_currency, mcap,
                    mcap / latest(inc, "net_income"), ev / latest(inc, "ebitda"), mcap / latest(bal, "total_equity"),
                    ev / latest(inc, "revenue"), ev, self.name, f.as_of)


DemoSource.get_fundamentals = _demo_fundamentals
DemoSource.get_key_stats = _demo_key_stats


DEMO_HOLDERS = ["Blackrock Inc.", "Vanguard Group Inc", "State Street Corp", "Fidelity (FMR)", "Capital Research",
                "Norges Bank", "T. Rowe Price", "Geode Capital", "Amundi", "UBS Asset Management"]


def _demo_ownership(self, symbol: str) -> Ownership:
    name, ccy, qtype, _, _ = self._meta(symbol)
    if qtype != "EQUITY":
        raise DataSourceError(f"No ownership data for {symbol} ({qtype.lower()}).")
    rng = np.random.default_rng(_seed(symbol) + 11)
    shares_out = 1e9 * (1 + _seed(symbol) % 5)
    price = self.get_quote(symbol).price
    picks = rng.choice(len(DEMO_HOLDERS), 7, replace=False)
    pct = np.sort(rng.uniform(0.005, 0.08, 7))[::-1]
    inst = pd.DataFrame({"Date Reported": pd.Timestamp(self.today) - pd.Timedelta(days=55), "Holder": [DEMO_HOLDERS[i] for i in picks],
                         "pctHeld": pct, "Shares": (pct * shares_out).round(), "Value": (pct * shares_out * price).round(),
                         "pctChange": rng.normal(0, 0.02, 7)})
    people = ["A. Smith", "B. Jones", "C. Brown", "D. Miller"]
    rows = []
    for k in range(12):
        who = people[k % 4]
        sale = rng.random() < 0.7
        n = int(rng.integers(1000, 50000))
        rows.append({"Shares": n, "Value": n * price, "Text": ("Sale" if sale else "Purchase") + f" at price {price:.2f} per share.",
                     "Insider": who, "Position": ["CEO", "CFO", "Director", "Officer"][k % 4], "Transaction": "",
                     "Start Date": pd.Timestamp(self.today) - pd.Timedelta(days=int(15 * k)), "Ownership": "D"})
    roster = pd.DataFrame({"Name": people, "Position": ["CEO", "CFO", "Director", "Officer"],
                           "Shares Owned Directly": rng.integers(1e4, 1e6, 4)})
    return Ownership(symbol, inst, inst.iloc[:4].assign(pctHeld=inst["pctHeld"].iloc[:4] / 3), pd.DataFrame(rows), roster,
                     {"insiders_pct": 0.01, "institutions_pct": float(pct.sum() * 4)}, self.name, datetime.now(timezone.utc))


DemoSource.get_ownership = _demo_ownership
