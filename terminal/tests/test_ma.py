import numpy as np
import pandas as pd
import pytest

from gterm.analysis import ma


def base(**kw):
    d = dict(acq_price=50, acq_shares=100, acq_net_income=500, tgt_price=20, tgt_shares=50, tgt_net_income=100,
             premium=0.25, pct_stock=1.0, cost_of_debt=0.06, tax_rate=0.25)
    d.update(kw)
    return ma.DealInputs(**d)


def test_all_stock_deal_hand_calculation():
    r = ma.analyse(base())
    # offer 25 x 50 = 1250 paid in stock at 50 -> 25 new shares; PF NI 600 / 125 shares = 4.80 vs EPS 5.00
    assert r.offer_price == 25 and r.purchase_equity == 1250 and r.new_shares == 25
    assert r.pro_forma_eps == pytest.approx(4.8) and r.accretion == pytest.approx(-0.04)
    assert r.ownership_target_holders == pytest.approx(0.2)
    assert r.offer_pe == pytest.approx(12.5)


def test_all_cash_deal_and_breakeven_synergies():
    r = ma.analyse(base(pct_stock=0.0))
    fin = 1250 * 0.06 * 0.75  # 56.25
    assert r.after_tax_financing_cost == pytest.approx(fin)
    assert r.pro_forma_eps == pytest.approx((600 - fin) / 100)
    # stock deal is dilutive by 0.20/share x 125 = 25 of after-tax earnings -> 33.33 pre-tax synergies to break even
    s = ma.analyse(base()).breakeven_synergies
    assert s == pytest.approx(25 / 0.75)
    assert ma.analyse(base(synergies=s)).accretion == pytest.approx(0, abs=1e-12)


def test_deal_multiples_and_sensitivity_direction():
    r = ma.analyse(base(tgt_net_debt=200, tgt_ebitda=290, tgt_revenue=1000))
    assert r.offer_ev == 1450 and r.offer_ev_ebitda == pytest.approx(5.0) and r.offer_ev_sales == pytest.approx(1.45)
    grid = ma.sensitivity(base())
    assert (grid.diff(axis=0).dropna() < 0).all().all()  # higher premium -> less accretive
    with pytest.raises(ValueError):
        ma.analyse(base(acq_price=0))


def test_implied_price_from_user_comps():
    comps = pd.DataFrame({"Deal": ["A", "B", "C"], "EV/EBITDA": [8, 10, 12], "EV/Sales": [2, None, 3], "Premium %": [20, 30, 40]})
    out = ma.implied_from_comps(comps, tgt_ebitda=100, tgt_revenue=400, tgt_net_debt=200, tgt_shares=50).set_index("Multiple")
    assert out.loc["EV/EBITDA", "Implied price / share"] == pytest.approx((1000 - 200) / 50)
    assert out.loc["EV/Sales", "Deals"] == 2 and out.loc["EV/Sales", "Median"] == 2.5
    assert out.loc["Premium paid", "Median"] == pytest.approx(0.30)
    empty = ma.implied_from_comps(pd.DataFrame(columns=["Deal", "EV/EBITDA"]), 100, 400, 0, 50)
    assert np.isnan(empty.set_index("Multiple").loc["EV/EBITDA", "Median"])
