"""Fetch everything the portfolio page needs, through the data adapter layer."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..analysis import portfolio as pa
from ..data import registry
from ..data.base import DataSourceError
from . import fx


@dataclass
class LoadedPortfolio:
    prices_base: pd.DataFrame  # daily closes converted to base currency, aligned
    quantities: dict
    meta: pd.DataFrame  # index symbol: name, type, currency, sector, country, region, last_price (native), cost_price
    source: str
    errors: list = field(default_factory=list)
    fx_notes: list = field(default_factory=list)


def profile(symbol: str) -> dict:
    q = registry.quote(symbol)
    sector, country = "", ""
    if q.asset_type == "EQUITY":
        try:
            ks = registry.key_stats(symbol)
            sector, country = ks.sector, ks.country
        except DataSourceError:
            pass
    if not sector:
        sector = {"EQUITY": "Unclassified (profile unavailable)", "ETF": "Funds (ETF)", "CRYPTOCURRENCY": "Crypto", "INDEX": "Index", "MUTUALFUND": "Funds"}.get(q.asset_type, "Unknown")
    if not country and q.asset_type in ("EQUITY", "ETF"):
        country = pa.country_from_symbol(symbol)  # profile lookup failed or is missing: infer from the exchange
    return {"name": q.name, "type": q.asset_type, "currency": q.currency, "sector": sector, "country": country,
            "region": pa.region_for(country, symbol, q.asset_type), "last_price": q.price, "price_date": q.price_date}


def load(holdings: list[dict], base_ccy: str, period: str = "1y") -> LoadedPortfolio:
    prices, metas, errors, notes = {}, {}, [], set()
    quantities = {}
    source = registry.active_source_name()
    for h in holdings:
        sym = h["symbol"]
        try:
            hist = registry.prices(sym, period, "1d")
            conv, note = fx.convert_series(hist.frame["Close"], hist.currency, base_ccy)
            prices[sym] = conv
            notes.add(note)
            metas[sym] = {**profile(sym), "cost_price": h.get("cost_price")}
            quantities[sym] = h["quantity"]
        except (DataSourceError, ValueError) as exc:
            errors.append(f"{sym}: {exc}")
    if not prices:
        raise DataSourceError("No holdings could be priced. " + "; ".join(errors))
    return LoadedPortfolio(pa.align(prices), quantities, pd.DataFrame(metas).T, source, errors, sorted(notes))


def benchmark(symbol: str, base_ccy: str, period: str) -> pd.Series:
    hist = registry.prices(symbol, period, "1d")
    return fx.convert_series(hist.frame["Close"], hist.currency, base_ccy)[0]
