import numpy as np
import pytest

from gterm.analysis import options as op
from gterm.analysis import strategies_opt as so


def test_long_call_payoff_breakeven_and_limits():
    legs = [so.Leg("call", "long", 100, 5.0, 1, 30, 0.2)]
    s = np.linspace(50, 150, 1001)
    pnl = so.payoff_at_expiry(legs, s)
    assert pnl[0] == pytest.approx(-500)  # lose the premium x 100
    assert so.breakevens(s, pnl) == [pytest.approx(105.0, abs=0.05)]
    profit, loss = so.max_profit_loss(s, pnl)
    assert profit is None and loss == pytest.approx(-500)  # unlimited upside


def test_short_call_unlimited_loss_and_credit():
    legs = [so.Leg("call", "short", 100, 4.0)]
    s = np.linspace(50, 200, 301)
    profit, loss = so.max_profit_loss(s, so.payoff_at_expiry(legs, s))
    assert profit == pytest.approx(400) and loss is None
    assert so.net_premium(legs) == -400  # credit received


def test_bull_call_spread_capped_both_ways():
    legs = [so.Leg("call", "long", 100, 6), so.Leg("call", "short", 110, 2)]
    s = np.linspace(60, 160, 1001)
    pnl = so.payoff_at_expiry(legs, s)
    profit, loss = so.max_profit_loss(s, pnl)
    assert profit == pytest.approx((10 - 4) * 100) and loss == pytest.approx(-400)
    assert so.breakevens(s, pnl) == [pytest.approx(104, abs=0.05)]


def test_straddle_two_breakevens_and_covered_call():
    straddle = [so.Leg("call", "long", 100, 5), so.Leg("put", "long", 100, 5)]
    s = np.linspace(50, 150, 1001)
    assert so.breakevens(s, so.payoff_at_expiry(straddle, s)) == [pytest.approx(90, abs=0.05), pytest.approx(110, abs=0.05)]
    covered = [so.Leg("stock", "long", 0, 100), so.Leg("call", "short", 110, 3)]
    profit, loss = so.max_profit_loss(s, so.payoff_at_expiry(covered, s))
    assert profit == pytest.approx((10 + 3) * 100)  # capped above 110


def test_value_today_converges_to_expiry_payoff_and_greeks_add_up():
    legs = [so.Leg("call", "long", 100, 5, 1, 30, 0.25), so.Leg("put", "short", 95, 2, 1, 30, 0.25)]
    s = np.array([90.0, 100.0, 110.0])
    near = so.value_today(legs, s, r=0.04, days_passed=29.999)
    assert np.allclose(near, so.payoff_at_expiry(legs, s), atol=2.0)  # ATM keeps a sliver of time value (~1.65)
    g = so.net_greeks(legs, 100, r=0.04)
    c = op.greeks(100, 100, 30 / 365, 0.04, 0.25, "call")
    p = op.greeks(100, 95, 30 / 365, 0.04, 0.25, "put")
    assert g["delta"] == pytest.approx(100 * (c["delta"] - p["delta"]))
    assert so.net_greeks([so.Leg("stock", "long", 0, 100, 2)], 100)["delta"] == 200


@pytest.mark.parametrize("name", so.PRESETS)
def test_presets_are_well_formed(name):
    legs = so.preset(name, 187.3)
    assert legs and all(l.kind in ("call", "put", "stock") for l in legs)
    s = so.price_grid(187.3, legs)
    pnl = so.payoff_at_expiry(legs, s)
    assert np.isfinite(pnl).all()
    if name == "Iron condor":
        profit, loss = so.max_profit_loss(s, pnl)
        assert profit is not None and loss is not None and profit > 0 > loss  # defined risk, net credit
