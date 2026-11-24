"""Deribit public API: crypto (BTC, ETH) option chains. Free, no key.

Deribit quotes option prices in the coin itself (e.g. 0.05 BTC), so we use its mark implied
volatility (`mark_iv`, in percent) directly, alongside the bid/ask converted to USD.
Instrument names look like BTC-27DEC26-100000-C (coin, expiry, strike in USD, C/P).
Expiry time is 08:00 UTC.
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from .base import DataSourceError
from .macro import _get

URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={c}&kind=option"


def parse_instrument(name: str) -> tuple[pd.Timestamp, float, str]:
    coin, exp, strike, cp = name.split("-")
    expiry = datetime.strptime(exp, "%d%b%y").replace(hour=8)
    return pd.Timestamp(expiry), float(strike), "call" if cp == "C" else "put"


def parse_book_summary(payload: dict) -> tuple[pd.DataFrame, float]:
    rows = payload.get("result")
    if not rows:
        raise DataSourceError("Deribit returned no options")
    out = []
    for r in rows:
        try:
            exp, k, kind = parse_instrument(r["instrument_name"])
        except (ValueError, KeyError):
            continue
        u = r.get("underlying_price") or 0
        coin_to_usd = lambda v: v * u if v else None  # noqa: E731
        out.append({"expiry": exp, "strike": k, "type": kind, "bid": coin_to_usd(r.get("bid_price")),
                    "ask": coin_to_usd(r.get("ask_price")), "last": coin_to_usd(r.get("mark_price")),
                    "quoted_iv": (r.get("mark_iv") or 0) / 100 or None, "open_interest": r.get("open_interest"),
                    "volume": r.get("volume"), "underlying": u})
    df = pd.DataFrame(out)
    # Each expiry has its own futures-based underlying; use the nearest expiry's as "spot".
    spot = float(df.sort_values("expiry")["underlying"].iloc[0])
    return df, spot


def options_chain(coin: str = "BTC") -> tuple[pd.DataFrame, float, str]:
    df, spot = parse_book_summary(json.loads(_get(URL.format(c=coin), 5 * 60)))
    return df, spot, "USD"
