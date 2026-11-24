import numpy as np
import pandas as pd
import pytest

from gterm.quant import factors


def test_point_in_time_respects_reporting_lag():
    annual = pd.Series([100.0, 200.0], index=pd.to_datetime(["2024-12-31", "2025-12-31"]))
    dates = pd.to_datetime(["2025-03-30", "2025-03-31", "2026-03-30", "2026-03-31"])
    v = factors.point_in_time(annual, dates, report_lag_days=90)
    # FY2024 known from 2025-03-31 (Dec 31 + 90 days); FY2025 from 2026-03-31.
    assert np.isnan(v.iloc[0]) and v.iloc[1] == 100 and v.iloc[2] == 100 and v.iloc[3] == 200


def test_momentum_skips_last_month():
    idx = pd.bdate_range("2024-01-01", "2025-06-30")
    p = pd.DataFrame({"A": np.linspace(100, 200, len(idx))}, index=idx)
    p.loc["2025-06-01":, "A"] = 50.0  # crash in the most recent month: must be ignored by 12-1 momentum
    d = pd.Timestamp("2025-06-30")
    s = factors.momentum_scores(p, [d])
    m = p.resample("ME").last()
    assert s.loc[d, "A"] == pytest.approx(m.iloc[-2, 0] / m.iloc[-14, 0] - 1)
    assert s.loc[d, "A"] > 0


def test_ic_perfect_and_inverse():
    d = pd.to_datetime(["2025-01-31", "2025-02-28"])
    scores = pd.DataFrame([[1, 2, 3, 4, 5]] * 2, index=d, columns=list("ABCDE"), dtype=float)
    fwd_good = pd.DataFrame([[0.01, 0.02, 0.03, 0.04, 0.05]] * 2, index=d, columns=list("ABCDE"))
    assert factors.information_coefficient(scores, fwd_good).tolist() == pytest.approx([1.0, 1.0])
    assert factors.information_coefficient(scores, -fwd_good).tolist() == pytest.approx([-1.0, -1.0])


def test_quantile_returns_monotonic_for_perfect_signal():
    d = pd.to_datetime(["2025-01-31"])
    cols = [f"S{i}" for i in range(10)]
    scores = pd.DataFrame([np.arange(10)], index=d, columns=cols, dtype=float)
    fwd = pd.DataFrame([np.arange(10) / 100], index=d, columns=cols)
    q = factors.quantile_returns(scores, fwd, 5).iloc[0]
    assert list(q.index) == ["Q1", "Q2", "Q3", "Q4", "Q5"] and q.is_monotonic_increasing
    assert q["Q1"] == pytest.approx(0.005) and q["Q5"] == pytest.approx(0.085)


def test_forward_returns_align_to_next_date():
    idx = pd.bdate_range("2025-01-01", "2025-03-31")
    p = pd.DataFrame({"A": np.arange(len(idx), dtype=float) + 100}, index=idx)
    d = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])
    f = factors.forward_returns(p, d)
    assert f.loc["2025-01-31", "A"] == pytest.approx(p.loc["2025-02-28", "A"] / p.loc["2025-01-31", "A"] - 1)
    assert np.isnan(f.loc["2025-03-31", "A"])


def test_value_and_quality_scores():
    d = pd.to_datetime(["2026-06-30"])
    px = pd.DataFrame({"A": [10.0], "B": [10.0]}, index=d)
    ni = {"A": pd.Series([100.0], index=pd.to_datetime(["2025-12-31"])), "B": pd.Series([50.0], index=pd.to_datetime(["2025-12-31"]))}
    v = factors.value_scores(px, ni, {"A": 100, "B": 100}, d)
    assert v.loc[d[0], "A"] == pytest.approx(0.10) and v.loc[d[0], "B"] == pytest.approx(0.05)
    eq = {"A": pd.Series([1000.0], index=pd.to_datetime(["2025-12-31"])), "B": pd.Series([-5.0], index=pd.to_datetime(["2025-12-31"]))}
    q = factors.quality_scores(ni, eq, ["A", "B"], d)
    assert q.loc[d[0], "A"] == pytest.approx(0.10) and np.isnan(q.loc[d[0], "B"])  # negative equity: not scored


def test_ic_summary_t_stat():
    ic = pd.Series([0.1, 0.1, 0.1, 0.3])
    s = factors.ic_summary(ic)
    assert s["Mean IC"] == pytest.approx(0.15)
    assert s["t-stat"] == pytest.approx(0.15 / ic.std(ddof=1) * 2)
