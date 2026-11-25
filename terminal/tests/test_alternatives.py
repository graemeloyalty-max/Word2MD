import numpy as np
import pandas as pd
import pytest

from gterm.analysis import alternatives as alt


def test_summary_returns_vol_and_correlation():
    idx = pd.date_range("2025-01-01", periods=400, freq="D")  # includes weekends (crypto calendar)
    rng = np.random.default_rng(0)
    b = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 400)), index=idx)
    same = b * 2  # perfectly correlated
    inverse = pd.Series(10000 / b.values, index=idx)  # moves opposite
    prices = pd.DataFrame({"SAME": same, "INV": inverse})
    t = alt.summary(prices, b, {"SAME": "Same"}, {"SAME": "Test"}).set_index("Symbol")
    assert t.loc["SAME", "Corr. with equities"] == pytest.approx(1.0)
    assert t.loc["INV", "Corr. with equities"] < -0.95
    assert t.loc["SAME", "Period"] == pytest.approx(b.iloc[-1] / b.iloc[0] - 1)
    assert t.loc["SAME", "Name"] == "Same" and t.loc["INV", "Name"] == "INV"


def test_weekdays_only_and_short_series_skipped():
    idx = pd.date_range("2025-01-01", periods=20, freq="D")
    df = pd.DataFrame({"A": range(20)}, index=idx)
    assert (alt.weekdays(df).index.dayofweek < 5).all()
    t = alt.summary(df.astype(float) + 1, df["A"].astype(float) + 1, {}, {})
    assert t.empty  # fewer than 30 observations


def test_universe_is_well_formed():
    syms = [s for group in alt.UNIVERSE.values() for s in group]
    assert len(syms) == len(set(syms)) and "GC=F" in syms and "BTC-USD" in syms
