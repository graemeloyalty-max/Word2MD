import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Each test gets its own data dir and .env, and the offline demo source."""
    from gterm.core import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "data" / "settings.json")
    config.save_settings({"base_currency": "USD", "data_source": "demo"})
    for k in config.API_KEYS:
        monkeypatch.delenv(k, raising=False)
    yield
