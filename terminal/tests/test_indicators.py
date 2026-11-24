import numpy as np
import pandas as pd
import pytest

from gterm.core import indicators as ind


def s(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_sma_matches_hand_calculation():
    out = ind.sma(s([1, 2, 3, 4, 5]), 3)
    assert out.isna().sum() == 2
    assert out.iloc[-1] == pytest.approx(4.0)  # (3+4+5)/3


def test_ema_constant_series_is_constant():
    assert ind.ema(s([7] * 30), 10).dropna().eq(7).all()


def test_rsi_bounds_and_extremes():
    up = ind.rsi(s(range(1, 40)))
    assert up.dropna().iloc[-1] == pytest.approx(100.0)  # only gains
    down = ind.rsi(s(range(40, 1, -1)))
    assert down.dropna().iloc[-1] == pytest.approx(0.0)  # only losses
    noisy = ind.rsi(s(np.random.default_rng(0).normal(100, 1, 200)))
    assert noisy.dropna().between(0, 100).all()


def test_rsi_known_value():
    # Alternating +1/-1 moves: average gain == average loss -> RSI 50.
    vals = [10 + (i % 2) for i in range(100)]
    assert ind.rsi(s(vals)).iloc[-1] == pytest.approx(50.0, abs=2.0)


def test_macd_hist_is_line_minus_signal():
    m = ind.macd(s(np.linspace(1, 50, 100)))
    assert np.allclose(m["hist"], m["macd"] - m["signal"])


def test_bollinger_band_symmetry():
    b = ind.bollinger(s(np.random.default_rng(1).normal(50, 2, 60))).dropna()
    assert np.allclose(b.upper - b.mid, b.mid - b.lower)


def test_returns_vol_drawdown():
    c = s([100, 110, 99, 120])
    assert ind.total_return(c) == pytest.approx(0.20)
    assert ind.max_drawdown(c) == pytest.approx(99 / 110 - 1)
    assert ind.annualised_volatility(s([100] * 10)) == 0.0
