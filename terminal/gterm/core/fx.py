"""Currency conversion at the historical FX rate for each date (ground rule 4).

Rates come through the same data adapter, via Yahoo-style pair symbols
(e.g. EURUSD=X = USD per 1 EUR). Pence-quoted London prices (GBp) are handled.
"""
from __future__ import annotations

import pandas as pd

MINOR_UNITS = {"GBp": ("GBP", 100.0), "GBX": ("GBP", 100.0), "ZAc": ("ZAR", 100.0), "ILA": ("ILS", 100.0)}


def normalise_currency(ccy: str) -> tuple[str, float]:
    """Return (major currency, divisor). E.g. GBp -> (GBP, 100)."""
    if ccy in MINOR_UNITS:
        return MINOR_UNITS[ccy]
    return ccy.upper(), 1.0


def pair_symbol(from_ccy: str, to_ccy: str) -> str:
    return f"{from_ccy}{to_ccy}=X"


def convert_series(series: pd.Series, from_ccy: str, to_ccy: str, period: str = "max", fetch=None) -> tuple[pd.Series, str]:
    """Convert a dated price series to `to_ccy` using each date's FX close.

    Returns the converted series and a note describing the rate used.
    `fetch(symbol, period)` returns a PriceHistory; injectable for tests.
    """
    major, div = normalise_currency(from_ccy)
    s = series / div
    if major == to_ccy:
        note = f"{from_ccy} -> {to_ccy}: /{div:g}" if div != 1 else "no conversion"
        return s, note
    if fetch is None:
        from ..data import registry

        fetch = lambda sym, per: registry.prices(sym, per, "1d")  # noqa: E731
    fx = fetch(pair_symbol(major, to_ccy), period).frame["Close"]
    # Align each price date to the most recent FX close on or before it (no look-ahead).
    rate = fx.sort_index().reindex(s.index, method="ffill")
    if rate.isna().all():
        raise ValueError(f"No FX history for {major}/{to_ccy}")
    rate = rate.bfill()
    return s * rate, f"converted {from_ccy}->{to_ccy} at daily {pair_symbol(major, to_ccy)} close"
