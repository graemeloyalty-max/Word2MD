import numpy as np
import pandas as pd
import pytest

from gterm.analysis import portfolio as pa
from gterm.core import portfolio_loader, watchlists


def s(vals, start="2026-01-05"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)), dtype=float)


def test_align_forward_fills_holidays_never_backfills():
    a = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]))
    b = pd.Series([10.0, 30.0], index=pd.to_datetime(["2026-01-05", "2026-01-07"]))  # holiday on the 6th
    df = pa.align({"A": a, "B": b})
    assert df.loc["2026-01-06", "B"] == 10.0  # carried forward, not the future 30
    late = pd.Series([5.0], index=pd.to_datetime(["2026-01-07"]))
    assert pa.align({"A": a, "L": late}).index[0] == pd.Timestamp("2026-01-07")  # no backfill before listing


def test_contributions_sum_to_portfolio_return():
    prices = pd.DataFrame({"A": s([10, 12, 15]), "B": s([20, 18, 19])})
    pos = pa.position_values(prices, {"A": 10, "B": 5})  # start 100 + 100 = 200; end 150 + 95 = 245
    total = pos.sum(axis=1)
    c = pa.contributions(pos)
    assert c["A"] == pytest.approx(50 / 200) and c["B"] == pytest.approx(-5 / 200)
    assert c.sum() == pytest.approx(total.iloc[-1] / total.iloc[0] - 1)


def test_attribution_groups_and_weights():
    prices = pd.DataFrame({"A": s([10, 12]), "B": s([20, 18]), "C": s([5, 5])})
    pos = pa.position_values(prices, {"A": 10, "B": 5, "C": 20})
    att = pa.attribution(pos, {"A": "Tech", "B": "Tech", "C": "Energy"})
    assert att.loc["Tech", "start_weight"] == pytest.approx(200 / 300)
    assert att.loc["Tech", "contribution"] == pytest.approx((20 - 10) / 300)
    assert att["end_weight"].sum() == pytest.approx(1.0)


def test_beta_of_levered_copy_is_two():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 300)
    bench = s(100 * np.cumprod(1 + r))
    port = s(100 * np.cumprod(1 + 2 * r))
    assert pa.beta(port, bench) == pytest.approx(2.0, rel=1e-9)
    assert pa.beta(bench, bench) == pytest.approx(1.0)


def test_drawdown_vol_and_sharpe():
    v = s([100, 120, 90, 130])
    assert pa.drawdown(v).min() == pytest.approx(90 / 120 - 1)
    flat = s([100] * 50)
    assert pa.volatility(flat) == 0
    assert np.isnan(pa.sharpe(flat))


def test_annualised_return_one_year():
    v = pd.Series([100.0, 110.0], index=pd.to_datetime(["2025-01-01", "2026-01-01"]))
    assert pa.annualised_return(v) == pytest.approx(0.10, abs=1e-3)


def test_region_and_suffix_fallback():
    assert pa.country_from_symbol("NESN.SW") == "Switzerland"
    assert pa.country_from_symbol("NPN.JO") == "South Africa"
    assert pa.region_for("South Africa", "NPN.JO") == "Emerging markets"
    assert pa.region_for("Germany", "SAP.DE") == "Europe"
    assert pa.region_for("", "BTC-USD", "CRYPTOCURRENCY") == "Crypto"


def test_portfolio_store_round_trip():
    watchlists.save_portfolio("Test", [{"symbol": "aapl", "quantity": 3, "cost_price": 150.0},
                                       {"symbol": "SAP.DE", "quantity": 2, "cost_price": float("nan")},
                                       {"symbol": "", "quantity": 1}, {"symbol": "MSFT", "quantity": 0}])
    h = watchlists.get_holdings("Test")
    assert h == [{"symbol": "AAPL", "quantity": 3.0, "cost_price": 150.0}, {"symbol": "SAP.DE", "quantity": 2.0, "cost_price": None}]
    watchlists.delete_portfolio("Test")
    assert "Test" not in watchlists.list_portfolios()


def test_loader_converts_to_base_currency_with_demo_data():
    lp = portfolio_loader.load([{"symbol": "SAP.DE", "quantity": 1}, {"symbol": "AAPL", "quantity": 1}], "ZAR", "6mo")
    assert set(lp.prices_base.columns) == {"SAP.DE", "AAPL"}
    assert any("EUR->ZAR" in n for n in lp.fx_notes) and any("USD->ZAR" in n for n in lp.fx_notes)
    assert lp.meta.loc["SAP.DE", "region"] == "Europe"
