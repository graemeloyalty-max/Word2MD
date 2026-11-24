from datetime import datetime, timezone

import pandas as pd
import pytest

from gterm.core import fx
from gterm.data.base import PriceHistory


def fake_fetch(rates: dict):
    def fetch(symbol, period):
        idx = pd.DatetimeIndex(list(rates))
        df = pd.DataFrame({"Close": list(rates.values())}, index=idx)
        return PriceHistory(symbol, df, "USD", "test", datetime.now(timezone.utc))

    return fetch


def test_uses_each_dates_rate_not_todays():
    prices = pd.Series([100.0, 100.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    fetch = fake_fetch({"2024-01-02": 1.10, "2024-01-03": 1.20})
    out, _ = fx.convert_series(prices, "EUR", "USD", fetch=fetch)
    assert list(out) == pytest.approx([110.0, 120.0])


def test_missing_fx_date_uses_previous_rate_no_lookahead():
    prices = pd.Series([100.0, 100.0], index=pd.to_datetime(["2024-01-02", "2024-01-04"]))
    fetch = fake_fetch({"2024-01-02": 18.0, "2024-01-05": 19.0})
    out, _ = fx.convert_series(prices, "USD", "ZAR", fetch=fetch)
    assert list(out) == pytest.approx([1800.0, 1800.0])  # 01-04 uses 01-02, not future 01-05


def test_pence_to_pounds_without_fetch():
    prices = pd.Series([2600.0], index=pd.to_datetime(["2024-01-02"]))
    out, _ = fx.convert_series(prices, "GBp", "GBP", fetch=None)
    assert out.iloc[0] == pytest.approx(26.0)


def test_pence_to_usd_divides_then_converts():
    prices = pd.Series([2600.0], index=pd.to_datetime(["2024-01-02"]))
    out, _ = fx.convert_series(prices, "GBp", "USD", fetch=fake_fetch({"2024-01-02": 1.25}))
    assert out.iloc[0] == pytest.approx(32.5)
