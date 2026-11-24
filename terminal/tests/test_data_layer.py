import pandas as pd
import pytest

from gterm.core import cache, config, watchlists
from gterm.data import registry
from gterm.data.base import PRICE_COLUMNS, DataSourceError
from gterm.data.demo_source import DemoSource


def test_demo_source_contract():
    src = DemoSource(pd.Timestamp("2026-11-24"))
    h = src.get_prices("AAPL", "1y", "1d")
    assert list(h.frame.columns) == PRICE_COLUMNS
    assert h.frame.index.is_monotonic_increasing
    assert (h.frame.High >= h.frame.Low).all()
    assert h.source.startswith("demo") and h.currency == "USD"
    q = src.get_quote("SAP.DE")
    assert q.currency == "EUR" and q.price > 0


def test_demo_is_deterministic():
    a = DemoSource(pd.Timestamp("2026-11-24")).get_prices("MSFT", "6mo")
    b = DemoSource(pd.Timestamp("2026-11-24")).get_prices("MSFT", "6mo")
    pd.testing.assert_frame_equal(a.frame, b.frame)


def test_weekly_resample():
    h = DemoSource(pd.Timestamp("2026-11-24")).get_prices("SPY", "1y", "1wk")
    assert 50 <= len(h.frame) <= 54


def test_bad_period_raises():
    with pytest.raises(DataSourceError):
        DemoSource().get_prices("AAPL", "7y")


def test_registry_caches_and_flags_cached():
    first = registry.prices("aapl", "3mo")
    assert not first.from_cache and first.symbol == "AAPL"
    second = registry.prices("AAPL", "3mo")
    assert second.from_cache
    assert second.as_of == first.as_of  # original retrieval time is preserved
    assert cache.stats()["entries"] >= 1
    assert cache.clear() >= 1


def test_watchlists_crud_and_alerts():
    watchlists.seed_defaults()
    assert "Europe" in watchlists.list_watchlists()
    watchlists.create_watchlist("Mine", ["aapl", "AAPL", "msft"])
    assert watchlists.symbols("Mine") == ["AAPL", "MSFT"]
    watchlists.remove_symbol("Mine", "AAPL")
    assert watchlists.symbols("Mine") == ["MSFT"]
    watchlists.delete_watchlist("Mine")
    assert "Mine" not in watchlists.list_watchlists()
    watchlists.add_alert("AAPL", "above", 100)
    a = watchlists.list_alerts()[0]
    assert a.triggered(101) and not a.triggered(99)


def test_api_keys_go_to_env_file():
    assert config.require_key("ANTHROPIC_API_KEY") is None  # paid module can show "add your key"
    config.set_api_key("FRED_API_KEY", "abc123")
    assert "FRED_API_KEY='abc123'" in config.ENV_FILE.read_text() or "FRED_API_KEY=abc123" in config.ENV_FILE.read_text()
    assert config.key_status()["FRED_API_KEY"]
    with pytest.raises(ValueError):
        config.set_api_key("NOT_A_KEY", "x")
