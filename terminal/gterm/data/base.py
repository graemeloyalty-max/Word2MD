"""The common data adapter interface.

Every data source (yfinance today; SEC EDGAR, EODHD and others later) implements
`DataSource`. Analysis code and pages only ever talk to this interface, so a new
source can be plugged in without touching them.

Every result carries `source` and `as_of`, so the UI can show where each number
came from and how fresh it is (ground rule 2 of the project brief).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

# Standard period / interval vocabulary shared by all sources.
PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
INTERVALS = ["1d", "1wk", "1mo"]

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataSourceError(Exception):
    """Raised when a source cannot supply the requested data."""


@dataclass
class PriceHistory:
    symbol: str
    frame: pd.DataFrame  # index: DatetimeIndex (naive, exchange-local dates); columns: PRICE_COLUMNS
    currency: str
    source: str
    as_of: datetime  # when the data was retrieved from the source
    from_cache: bool = False

    @property
    def last_close(self) -> float:
        return float(self.frame["Close"].iloc[-1])

    @property
    def last_date(self) -> pd.Timestamp:
        return self.frame.index[-1]


@dataclass
class Quote:
    symbol: str
    name: str
    price: float
    previous_close: float
    currency: str
    exchange: str
    asset_type: str  # EQUITY, ETF, INDEX, CURRENCY, CRYPTOCURRENCY, ...
    price_date: pd.Timestamp  # date of the last price
    source: str
    as_of: datetime
    extra: dict = field(default_factory=dict)

    @property
    def change(self) -> float:
        return self.price - self.previous_close

    @property
    def change_pct(self) -> float:
        if not self.previous_close:
            return 0.0
        return self.change / self.previous_close * 100.0


@dataclass
class Fundamentals:
    """Company profile + normalised statements (see gterm/core/statements.py).

    statements: {'income','balance','cashflow'} -> DataFrame, rows = standard lines,
    columns = fiscal year-end dates, oldest -> newest. Values in `financial_currency`.
    """

    symbol: str
    name: str
    sector: str
    industry: str
    country: str
    price: float
    price_currency: str  # may be a minor unit, e.g. GBp
    financial_currency: str  # currency the statements are reported in
    shares: float
    beta: float | None
    statements: dict
    adjustments: list
    source: str
    as_of: datetime
    history_source: str = ""  # e.g. "+ SEC EDGAR (2009-2026)"


@dataclass
class KeyStats:
    """Lightweight snapshot used for peer comparison (one call per company)."""

    symbol: str
    name: str
    sector: str
    country: str
    currency: str
    market_cap: float | None
    pe: float | None
    ev_ebitda: float | None
    pb: float | None
    ev_sales: float | None
    enterprise_value: float | None
    source: str
    as_of: datetime


@dataclass
class Ownership:
    """Who owns a company. DataFrames may be empty when the source has no data (common outside the US)."""

    symbol: str
    institutions: pd.DataFrame  # Holder, pctHeld, Shares, Value, Date Reported
    funds: pd.DataFrame  # same columns, mutual funds
    insider_transactions: pd.DataFrame  # Insider, Position, Transaction/Text, Shares, Value, Start Date
    insider_roster: pd.DataFrame  # Name, Position, Shares Owned Directly, Latest Transaction Date
    summary: dict  # e.g. {"insiders_pct": 0.02, "institutions_pct": 0.62}
    source: str
    as_of: datetime


class DataSource(ABC):
    """Interface every data source implements."""

    name: str = "base"
    #: True if the source needs an API key to work.
    requires_key: bool = False

    @abstractmethod
    def get_prices(self, symbol: str, period: str = "1y", interval: str = "1d") -> PriceHistory:
        ...

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        ...

    def search(self, query: str) -> list[dict]:
        """Optional symbol search. Returns dicts with symbol, name, exchange, type."""
        return []

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        raise DataSourceError(f"{self.name} does not supply fundamentals.")

    def get_key_stats(self, symbol: str) -> KeyStats:
        raise DataSourceError(f"{self.name} does not supply key stats.")

    def get_ownership(self, symbol: str) -> Ownership:
        raise DataSourceError(f"{self.name} does not supply ownership data.")
