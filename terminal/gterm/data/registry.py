"""Picks the active data source and wraps it with the local cache.

Pages call `prices()` / `quote()` here and never import a concrete source.
"""
from __future__ import annotations

import dataclasses
from functools import lru_cache

from ..core import cache, config
from . import edgar_source
from .base import DataSource, DataSourceError, Fundamentals, KeyStats, PriceHistory, Quote
from .demo_source import DemoSource
from .yfinance_source import YFinanceSource

SOURCES = {"yfinance": YFinanceSource, "demo": DemoSource}

# Cache lifetimes. Daily bars barely change intraday; quotes go stale fast.
PRICE_TTL_S = 6 * 3600
QUOTE_TTL_S = 5 * 60
FUNDAMENTALS_TTL_S = 24 * 3600


@lru_cache(maxsize=1)
def _yfinance_reachable() -> bool:
    try:
        YFinanceSource().get_prices("SPY", "1mo", "1d")
        return True
    except Exception:
        return False


def active_source_name() -> str:
    choice = config.load_settings().get("data_source", "auto")
    if choice == "auto":
        return "yfinance" if _yfinance_reachable() else "demo"
    return choice


def get_source(name: str | None = None) -> DataSource:
    return SOURCES[name or active_source_name()]()


def _mark_cached(obj):
    return dataclasses.replace(obj, from_cache=True) if isinstance(obj, PriceHistory) else obj


def prices(symbol: str, period: str = "1y", interval: str = "1d", use_cache: bool = True) -> PriceHistory:
    symbol = symbol.strip().upper()
    src = active_source_name()
    key = f"prices|{src}|{symbol}|{period}|{interval}"
    if use_cache and (hit := cache.get(key, PRICE_TTL_S)) is not None:
        return _mark_cached(hit)
    result = get_source(src).get_prices(symbol, period, interval)
    cache.put(key, result)
    return result


def quote(symbol: str, use_cache: bool = True) -> Quote:
    symbol = symbol.strip().upper()
    src = active_source_name()
    key = f"quote|{src}|{symbol}"
    if use_cache and (hit := cache.get(key, QUOTE_TTL_S)) is not None:
        return hit
    result = get_source(src).get_quote(symbol)
    cache.put(key, result)
    return result


def fundamentals(symbol: str, use_edgar: bool = False, use_cache: bool = True) -> Fundamentals:
    """Normalised statements. With use_edgar, US filers get older years from SEC EDGAR."""
    symbol = symbol.strip().upper()
    src = active_source_name()
    key = f"fund|{src}|{symbol}"
    f = cache.get(key, FUNDAMENTALS_TTL_S) if use_cache else None
    if f is None:
        f = get_source(src).get_fundamentals(symbol)
        cache.put(key, f)
    if use_edgar and src != "demo" and "." not in symbol:
        ua = config.load_settings().get("sec_user_agent", "")
        hist = edgar_source.fetch_history(symbol, ua)  # raises DataSourceError with a clear message
        stmts, note = edgar_source.merge_history(f.statements, hist)
        f = dataclasses.replace(f, statements=stmts, history_source=note)
    return f


def key_stats(symbol: str) -> KeyStats:
    symbol = symbol.strip().upper()
    src = active_source_name()
    key = f"stats|{src}|{symbol}"
    if (hit := cache.get(key, FUNDAMENTALS_TTL_S)) is not None:
        return hit
    result = get_source(src).get_key_stats(symbol)
    cache.put(key, result)
    return result


def search(query: str) -> list[dict]:
    return get_source().search(query) if query.strip() else []


__all__ = ["prices", "quote", "search", "active_source_name", "get_source", "DataSourceError"]


def risk_free_usd() -> tuple[float, str] | None:
    """Live US 10-year Treasury yield from ^TNX (quoted in percent), or None if unavailable."""
    if active_source_name() == "demo":
        return None
    try:
        h = prices("^TNX", "1mo", "1d")
        return h.last_close / 100.0, f"US 10y Treasury (^TNX) {h.last_close:.2f}% on {h.last_date:%Y-%m-%d}, {h.source}"
    except DataSourceError:
        return None
