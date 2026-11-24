import math

import numpy as np
import pandas as pd
import pytest

from gterm.analysis import options as op
from gterm.data import deribit


def test_textbook_value():
    # Hull, Options Futures and Other Derivatives: S=42, K=40, T=0.5, r=10%, sigma=20% -> call 4.76, put 0.81
    assert op.bs_price(42, 40, 0.5, 0.10, 0.20, "call") == pytest.approx(4.76, abs=0.01)
    assert op.bs_price(42, 40, 0.5, 0.10, 0.20, "put") == pytest.approx(0.81, abs=0.01)


def test_put_call_parity_with_dividends():
    S, K, T, r, q, v = 100, 95, 0.75, 0.04, 0.02, 0.3
    c, p = op.bs_price(S, K, T, r, v, "call", q), op.bs_price(S, K, T, r, v, "put", q)
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T))


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("sigma", [0.08, 0.25, 0.9])
def test_implied_vol_round_trip(kind, sigma):
    price = op.bs_price(100, 110, 0.4, 0.03, sigma, kind)
    assert op.implied_vol(price, 100, 110, 0.4, 0.03, kind) == pytest.approx(sigma, abs=1e-5)


def test_implied_vol_rejects_arbitrage_prices():
    assert math.isnan(op.implied_vol(0.5, 100, 80, 0.5, 0.0, "call"))  # below intrinsic value 20
    assert math.isnan(op.implied_vol(150, 100, 80, 0.5, 0.0, "call"))  # above the share price


def test_greeks_match_finite_differences():
    S, K, T, r, v = 100, 100, 0.5, 0.03, 0.25
    g = op.greeks(S, K, T, r, v, "call")
    h = 0.01
    assert g["delta"] == pytest.approx((op.bs_price(S + h, K, T, r, v) - op.bs_price(S - h, K, T, r, v)) / (2 * h), rel=1e-4)
    assert g["gamma"] == pytest.approx((op.bs_price(S + h, K, T, r, v) - 2 * op.bs_price(S, K, T, r, v) + op.bs_price(S - h, K, T, r, v)) / h ** 2, rel=1e-3)
    assert g["vega"] == pytest.approx((op.bs_price(S, K, T, r, v + 0.0001) - op.bs_price(S, K, T, r, v - 0.0001)) / 0.0002 / 100, rel=1e-4)
    day = 1 / 365
    assert g["theta"] == pytest.approx(op.bs_price(S, K, T - day, r, v) - op.bs_price(S, K, T, r, v), rel=2e-2)


def test_surface_uses_otm_options_and_recovers_vol():
    asof = pd.Timestamp("2026-01-01")
    S, r, sig = 100.0, 0.04, 0.3
    rows = []
    for exp in ["2026-03-01", "2026-07-01"]:
        T = (pd.Timestamp(exp) - asof).total_seconds() / (365.25 * 86400)
        for K in [80, 90, 100, 110, 120]:
            for kind in ("call", "put"):
                p = op.bs_price(S, K, T, r, sig, kind)
                rows.append({"expiry": exp, "strike": K, "type": kind, "bid": p * 0.99, "ask": p * 1.01, "last": p})
    pts = op.surface_points(pd.DataFrame(rows), S, r, asof=asof)
    assert set(pts[pts["strike"] < 100]["type"]) == {"put"} and set(pts[pts["strike"] >= 100]["type"]) == {"call"}
    assert np.allclose(pts["iv"], sig, atol=0.01)
    ts = op.term_structure(pts, S)
    assert len(ts) == 2 and np.allclose(ts.values, sig, atol=0.01)


def test_deribit_parsing():
    exp, k, kind = deribit.parse_instrument("BTC-27DEC26-100000-C")
    assert exp == pd.Timestamp("2026-12-27 08:00") and k == 100000 and kind == "call"
    payload = {"result": [
        {"instrument_name": "BTC-27DEC26-100000-C", "underlying_price": 60000, "bid_price": 0.01, "ask_price": 0.012,
         "mark_price": 0.011, "mark_iv": 55.0, "open_interest": 10, "volume": 1},
        {"instrument_name": "BTC-26DEC26-50000-P", "underlying_price": 59900, "bid_price": None, "ask_price": 0.02,
         "mark_price": 0.019, "mark_iv": 60.0},
        {"instrument_name": "garbage"}]}
    df, spot = deribit.parse_book_summary(payload)
    assert len(df) == 2 and spot == 59900  # nearest expiry's underlying
    row = df[df["type"] == "call"].iloc[0]
    assert row["bid"] == pytest.approx(600) and row["quoted_iv"] == pytest.approx(0.55)
