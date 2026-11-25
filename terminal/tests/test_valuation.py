from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from gterm.analysis import dcf, multiples, quality
from gterm.core import fx
from gterm.core.statements import normalise
from gterm.data import edgar_source
from gterm.data.base import KeyStats, PriceHistory

# ------------------------------------------------------------------ DCF


def base_inputs(**kw):
    d = dict(base_fcf=100.0, growth_high=0.0, years_high=5, years_total=10, terminal_growth=0.0, risk_free=0.05,
             beta=1.0, equity_risk_premium=0.05, country_risk_premium=0.0, equity_value_market=1000.0, debt=0.0,
             cash=0.0, shares=10.0)
    d.update(kw)
    return dcf.DCFInputs(**d)


def test_zero_growth_perpetuity_equals_fcf_over_rate():
    # With no growth anywhere, value = FCF / r exactly (a perpetuity), whatever the horizon.
    r = dcf.run(base_inputs())
    assert r.wacc == pytest.approx(0.10)
    assert r.enterprise_value == pytest.approx(100 / 0.10)
    assert r.value_per_share == pytest.approx(100.0)


def test_gordon_growth_perpetuity():
    # Constant growth g everywhere -> EV = FCF1 / (r - g) with FCF1 = FCF0 * (1+g).
    r = dcf.run(base_inputs(growth_high=0.03, terminal_growth=0.03))
    assert r.enterprise_value == pytest.approx(100 * 1.03 / (0.10 - 0.03))


def test_equity_bridge_debt_cash_minorities():
    r = dcf.run(base_inputs(debt=200, cash=50, minority_interest=30, equity_value_market=1e12))
    assert r.equity_value == pytest.approx(1000 - 200 + 50 - 30)


def test_wacc_weights_and_tax_shield():
    ke, kd, we, w = dcf.wacc(base_inputs(debt=1000, pre_tax_cost_of_debt=0.06, tax_rate=0.25))
    assert we == pytest.approx(0.5)
    assert kd == pytest.approx(0.045)
    assert w == pytest.approx(0.5 * 0.10 + 0.5 * 0.045)


def test_country_risk_premium_scaled_by_lambda():
    assert dcf.cost_of_equity(base_inputs(country_risk_premium=0.04, crp_exposure=0.5)) == pytest.approx(0.12)


def test_growth_fades_linearly_to_terminal():
    g = dcf.growth_path(base_inputs(growth_high=0.10, terminal_growth=0.02, years_high=5, years_total=10))
    assert list(g[:5]) == pytest.approx([0.10] * 5)
    assert g[-1] == pytest.approx(0.02)
    assert np.all(np.diff(g[5:]) < 0)


def test_terminal_growth_must_be_below_wacc():
    with pytest.raises(ValueError):
        dcf.run(base_inputs(terminal_growth=0.12))


def test_negative_fcf_warns():
    assert dcf.run(base_inputs(base_fcf=-10)).warnings


def test_reverse_dcf_round_trip():
    i = base_inputs(growth_high=0.12, terminal_growth=0.025)
    price = dcf.run(i).value_per_share
    assert dcf.implied_growth(base_inputs(terminal_growth=0.025), price) == pytest.approx(0.12, abs=1e-4)


def test_sensitivity_monotonic():
    s = dcf.sensitivity(base_inputs(growth_high=0.05, terminal_growth=0.02))
    assert (s.diff(axis=0).dropna() < 0).all().all()  # higher WACC -> lower value
    assert (s.diff(axis=1).iloc[:, 1:] > 0).all().all()  # higher growth -> higher value


# ------------------------------------------------------------------ statements normalisation

YEARS = pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"])


def raw(d):
    return pd.DataFrame(d, index=YEARS).T


def test_ifrs_interest_paid_in_financing_moved_to_operating():
    inc = raw({"Total Revenue": [100, 110, 120], "Operating Income": [20, 22, 24]})
    bal = raw({"Stockholders Equity": [50, 55, 60], "Total Debt": [30, 30, 30], "Cash And Cash Equivalents": [10, 10, 10]})
    cf = raw({"Operating Cash Flow": [25, 26, 27], "Capital Expenditure": [-5, -5, -5], "Interest Paid Cff": [-2, -2, -2],
              "Interest Received Cfi": [1, 1, 1], "Purchase Of Intangibles": [-1, -1, -1]})
    stmts, notes = normalise(inc, bal, cf)
    c = stmts["cashflow"]
    assert c.loc["operating_cash_flow"].iloc[-1] == 27 - 2 + 1
    assert c.loc["capex"].iloc[-1] == -6
    assert c.loc["free_cash_flow"].iloc[-1] == 26 - 6
    assert len(notes) >= 3
    assert stmts["balance"].loc["net_debt"].iloc[-1] == 20
    assert stmts["income"].loc["operating_margin"].iloc[-1] == pytest.approx(0.2)
    assert list(stmts["income"].columns) == sorted(stmts["income"].columns)  # oldest -> newest


def test_us_gaap_statements_need_no_cash_flow_adjustment():
    cf = raw({"Operating Cash Flow": [25, 26, 27], "Capital Expenditure": [-5, -5, -5]})
    stmts, notes = normalise(raw({"Total Revenue": [1, 1, 1]}), raw({"Total Debt": [0, 0, 0]}), cf)
    assert stmts["cashflow"].loc["operating_cash_flow"].iloc[-1] == 27
    assert not any("Interest" in n for n in notes)


# ------------------------------------------------------------------ quality


def test_linear_score_clips():
    assert quality.linear_score(0.10, 0.0, 0.20) == pytest.approx(50)
    assert quality.linear_score(0.50, 0.0, 0.20) == 100
    assert quality.linear_score(5.0, 4.0, 0.0) == 0  # leverage: lower is better
    assert np.isnan(quality.linear_score(np.nan, 0, 1))


def test_quality_on_known_statements():
    inc = raw({"Total Revenue": [100, 100, 100], "Operating Income": [20, 20, 20], "Pretax Income": [20, 20, 20],
               "Tax Provision": [5, 5, 5], "Net Common Stockholders": [15, 15, 15], "Net Income": [15, 15, 15], "EBITDA": [25, 25, 25]})
    bal = raw({"Stockholders Equity": [75, 75, 75], "Total Debt": [0, 0, 0], "Cash And Cash Equivalents": [0, 0, 0],
               "Invested Capital": [75, 75, 75]})
    cf = raw({"Operating Cash Flow": [20, 20, 20], "Capital Expenditure": [-5, -5, -5]})
    stmts, _ = normalise(inc, bal, cf)
    q = quality.compute(stmts)
    assert q.values["roic"] == pytest.approx(20 * 0.75 / 75)  # 20%
    assert q.scores["roic"] == 100
    assert q.values["margin_stability"] == pytest.approx(0.0)
    assert q.values["leverage"] == 0
    assert q.values["cash_conversion"] == pytest.approx(1.0)
    assert q.overall == 100


# ------------------------------------------------------------------ multiples


def ks(sym, pe, ev_ebitda, pb, ev_sales, mcap=100.0, ev=120.0):
    return KeyStats(sym, sym, "Technology", "Germany", "EUR", mcap, pe, ev_ebitda, pb, ev_sales, ev, "test",
                    datetime.now(timezone.utc))


def test_implied_prices_equity_and_firm_multiples():
    target = ks("T", pe=10, ev_ebitda=6, pb=2, ev_sales=1.2)
    peers = [ks("A", 20, 12, 4, 2.4), ks("B", 20, 12, 4, 2.4), ks("C", 999, 12, 4, 2.4)]  # median ignores outlier
    imp = multiples.implied_prices(target, peers, price=50).set_index("Multiple")
    assert imp.loc["P/E", "Implied price"] == pytest.approx(100)  # 50 * 20/10
    assert imp.loc["P/B", "Implied price"] == pytest.approx(100)
    # EV doubles from 120 to 240; net debt 20 held constant -> mcap 220 -> price 50 * 220/100
    assert imp.loc["EV/EBITDA", "Implied price"] == pytest.approx(110)


def test_default_peers_exclude_self_and_pick_region():
    assert "SAP.DE" not in multiples.default_peers("SAP.DE", "Technology", "Germany")
    assert "ASML.AS" in multiples.default_peers("SAP.DE", "Technology", "Germany")
    assert "MSFT" in multiples.default_peers("AAPL", "Technology", "United States")


# ------------------------------------------------------------------ FX


def test_latest_rate_minor_units():
    def fetch(sym, per):
        assert sym == "GBPUSD=X"
        df = pd.DataFrame({"Close": [1.30]}, index=pd.to_datetime(["2026-11-24"]))
        return PriceHistory(sym, df, "USD", "test", datetime.now(timezone.utc))

    rate, _ = fx.latest_rate("GBp", "USD", fetch=fetch)
    assert 2500 * rate == pytest.approx(32.5)  # 2,500p = £25 = $32.50
    assert fx.latest_rate("GBP", "GBp")[0] == 100


# ------------------------------------------------------------------ SEC EDGAR parsing (fixture, no network)


def fact(start, end, val, form="10-K", filed="2020-02-01"):
    return {"start": start, "end": end, "val": val, "form": form, "filed": filed}


FACTS = {"facts": {"us-gaap": {
    "Revenues": {"units": {"USD": [
        fact("2017-01-01", "2017-12-31", 100), fact("2018-01-01", "2018-12-31", 110),
        fact("2018-10-01", "2018-12-31", 30),  # a quarter inside a 10-K: must be dropped
        fact("2018-01-01", "2018-12-31", 111, filed="2021-02-01"),  # restated later: latest filing wins
        fact("2019-01-01", "2019-12-31", 120, form="10-Q"),  # not annual: dropped
    ]}},
    "Assets": {"units": {"USD": [{"end": "2018-12-31", "val": 500, "form": "10-K", "filed": "2019-02-01"}]}},
    "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [fact("2018-01-01", "2018-12-31", 40)]}},
    "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [fact("2018-01-01", "2018-12-31", 10)]}},
}}}


def test_edgar_annual_lines():
    f = edgar_source.annual_lines(FACTS)
    rev = f["income"].loc["revenue"]
    assert list(rev.values) == [100, 111]
    assert f["balance"].loc["total_assets"].iloc[0] == 500
    assert f["cashflow"].loc["free_cash_flow"].iloc[0] == 30


def test_edgar_merge_adds_only_older_years():
    yf_inc = pd.DataFrame({pd.Timestamp("2018-12-29"): [999.0]}, index=["revenue"])
    merged, note = edgar_source.merge_history({"income": yf_inc, "balance": pd.DataFrame(), "cashflow": pd.DataFrame()},
                                              edgar_source.annual_lines(FACTS))
    rev = merged["income"].loc["revenue"]
    assert list(rev.values) == [100, 999]  # 2018 kept from yfinance (end dates 2 days apart = same year)
    assert note == "+ SEC EDGAR (2017-2017)"


def test_edgar_requires_contact_user_agent():
    from gterm.data.base import DataSourceError

    with pytest.raises(DataSourceError, match="User-Agent"):
        edgar_source._get_json("https://data.sec.gov/x", "")


def test_terminal_growth_default_capped_at_risk_free_and_thin_spread_warns():
    assert dcf.default_terminal_growth(0.007) == 0.007  # Swiss franc: cap at the low risk-free rate
    assert dcf.default_terminal_growth(0.045) == 0.025
    assert dcf.default_terminal_growth(-0.002) == 0.0
    r = dcf.run(base_inputs(terminal_growth=0.09))  # WACC 10% -> 1-point spread
    assert any("percentage points above terminal growth" in w for w in r.warnings)
    assert not any("percentage points" in w for w in dcf.run(base_inputs(terminal_growth=0.02)).warnings)
