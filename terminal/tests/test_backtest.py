import numpy as np
import pandas as pd
import pytest

from gterm.quant import backtest, metrics, strategies


def frame(cols: dict, start="2025-01-01"):
    n = len(next(iter(cols.values())))
    return pd.DataFrame(cols, index=pd.bdate_range(start, periods=n), dtype=float)


def test_no_lookahead_decision_trades_next_day():
    # Asset jumps +100% on day 2. A decision made ON day 2 (seeing the jump) must not capture it.
    p = frame({"A": [10, 10, 20, 20, 20]})
    dec = pd.DataFrame({"A": [1.0]}, index=[p.index[2]])
    r = backtest.run(p, dec, cost_bps=0)
    assert r.equity.iloc[-1] == pytest.approx(1.0)
    # Decided on day 0 -> traded at the close of day 1 -> holds through the day-2 jump.
    dec1 = pd.DataFrame({"A": [1.0]}, index=[p.index[0]])
    assert backtest.run(p, dec1, cost_bps=0).equity.iloc[-1] == pytest.approx(2.0)


def test_execution_lag_zero_is_more_optimistic():
    p = frame({"A": [10, 10, 20, 20]})
    dec = pd.DataFrame({"A": [1.0]}, index=[p.index[1]])
    assert backtest.run(p, dec, 0, execution_lag=0).equity.iloc[-1] == pytest.approx(2.0)
    assert backtest.run(p, dec, 0, execution_lag=1).equity.iloc[-1] == pytest.approx(1.0)


def test_costs_charged_on_turnover():
    p = frame({"A": [10] * 5, "B": [10] * 5})
    dec = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=[p.index[0], p.index[2]])
    r = backtest.run(p, dec, cost_bps=10)
    assert list(r.turnover) == [1.0, 2.0]  # buy A (1.0), then sell A + buy B (2.0)
    assert r.equity.iloc[-1] == pytest.approx((1 - 0.001) * (1 - 0.002))


def test_weights_drift_between_rebalances():
    p = frame({"A": [10, 10, 20, 20], "B": [10, 10, 10, 10]})
    dec = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[p.index[0]])
    r = backtest.run(p, dec, cost_bps=0)
    assert r.equity.iloc[-1] == pytest.approx(1.5)  # 0.5 * 2 + 0.5 * 1


def test_cash_earns_zero():
    p = frame({"A": [10, 10, 20]})
    dec = pd.DataFrame({"A": [0.5]}, index=[p.index[0]])
    assert backtest.run(p, dec, 0).equity.iloc[-1] == pytest.approx(1.5)


def test_metrics_hand_checked():
    eq = pd.Series([100.0, 50.0, 150.0], index=pd.to_datetime(["2024-01-01", "2025-01-01", "2026-01-01"]))
    assert metrics.max_drawdown(eq) == pytest.approx(-0.5)
    assert metrics.cagr(eq) == pytest.approx(1.5 ** (1 / ((eq.index[-1] - eq.index[0]).days / 365.25)) - 1)


def test_rebalance_dates_month_end():
    idx = pd.bdate_range("2025-01-01", "2025-03-31")
    d = strategies.rebalance_dates(idx, "M")
    assert list(d.strftime("%m-%d")) == ["01-31", "02-28", "03-31"]


def test_momentum_picks_winner_and_uses_only_past_data():
    idx = pd.bdate_range("2023-01-02", "2025-06-30")
    up = pd.Series(np.linspace(10, 30, len(idx)), index=idx)
    down = pd.Series(np.linspace(30, 10, len(idx)), index=idx)
    p = pd.DataFrame({"UP": up, "DOWN": down})
    dec = strategies.momentum(p, lookback=6, top_n=1, skip=1)
    assert (dec["UP"] == 1).all() and (dec["DOWN"] == 0).all()
    # First decision only once 6+1 months of history exist
    assert dec.index[0] >= idx[0] + pd.DateOffset(months=7) - pd.Timedelta(days=31)


def test_sma_crossover_switches_to_cash_in_downtrend():
    idx = pd.bdate_range("2024-01-01", periods=300)
    vals = np.r_[np.linspace(10, 20, 150), np.linspace(20, 8, 150)]
    dec = strategies.sma_crossover(pd.DataFrame({"A": vals}, index=idx), fast=10, slow=50)
    assert dec["A"].iloc[0] == 1.0 and dec["A"].iloc[-1] == 0.0
    with pytest.raises(ValueError):
        strategies.sma_crossover(pd.DataFrame({"A": vals}, index=idx), fast=50, slow=10)


def test_inverse_vol_gives_calm_asset_more_weight():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=200)
    p = pd.DataFrame({"CALM": 100 * np.cumprod(1 + rng.normal(0, 0.005, 200)),
                      "WILD": 100 * np.cumprod(1 + rng.normal(0, 0.02, 200))}, index=idx)
    w = strategies.inverse_volatility(p).iloc[-1]
    assert w["CALM"] > 0.7 and w.sum() == pytest.approx(1.0)


def test_sweep_shape():
    idx = pd.bdate_range("2022-01-03", periods=600)
    rng = np.random.default_rng(1)
    p = pd.DataFrame({c: 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 600)) for c in "ABCD"}, index=idx)
    t = backtest.sweep(p, strategies.momentum, {"lookback": [3, 6], "top_n": [1, 2]})
    assert t.shape == (2, 2) and t.notna().all().all()


def test_annual_turnover_excludes_initial_purchase():
    p = frame({"A": [10.0] * 300})
    r = backtest.run(p, strategies.buy_and_hold(p), cost_bps=0)
    assert r.summary()["Annual turnover"] == 0.0
