"""Load a universe of prices (and fundamentals for value/quality) in one base currency."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..core import fx
from ..core.statements import series
from ..data import registry
from ..data.base import DataSourceError


@dataclass
class Universe:
    prices: pd.DataFrame  # base-currency daily closes, union of dates, forward-filled
    base: str
    source: str
    errors: list = field(default_factory=list)


def load_prices(symbols: list[str], base: str, period: str = "5y") -> Universe:
    data, errors = {}, []
    for s in symbols:
        try:
            h = registry.prices(s, period, "1d")
            data[s] = fx.convert_series(h.frame["Close"], h.currency, base)[0]
        except (DataSourceError, ValueError) as exc:
            errors.append(f"{s}: {exc}")
    if not data:
        raise DataSourceError("No prices for any symbol in the universe. " + "; ".join(errors[:3]))
    df = pd.DataFrame(data).sort_index().ffill()
    return Universe(df, base, registry.active_source_name(), errors)


def load_fundamentals(symbols: list[str], base: str) -> tuple[dict, dict, dict, list]:
    """Annual net income (converted to base currency at each fiscal year-end's FX rate), equity
    (statement currency; only used in ratios with net income) and share counts."""
    ni_base, ni, eq, shares, errors = {}, {}, {}, {}, []
    for s in symbols:
        try:
            f = registry.fundamentals(s)
        except DataSourceError as exc:
            errors.append(f"{s}: {exc}")
            continue
        n = series(f.statements["income"], "net_income")
        e = series(f.statements["balance"], "total_equity")
        if n.empty:
            continue
        ni[s], eq[s], shares[s] = n, e, f.shares
        try:
            ni_base[s] = fx.convert_series(n, f.financial_currency, base, period="10y")[0]
        except (DataSourceError, ValueError) as exc:
            errors.append(f"{s}: FX {exc}")
    return ni_base, ni, eq, shares, errors
