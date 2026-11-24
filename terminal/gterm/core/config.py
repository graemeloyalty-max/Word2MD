"""App settings.

- API keys live in `.env` (never in code, never in the database).
- Non-secret preferences (base currency, data source) live in `data/settings.json`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import dotenv_values, set_key

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("GTERM_DATA_DIR", ROOT / "data"))
ENV_FILE = Path(os.environ.get("GTERM_ENV_FILE", ROOT / ".env"))
SETTINGS_FILE = DATA_DIR / "settings.json"

BASE_CURRENCIES = ["USD", "EUR", "ZAR"]

# Every key any module will ever need, with the module that uses it and the phase.
API_KEYS = {
    "ANTHROPIC_API_KEY": ("Agentic Research (AI chat, agents)", "Phase 5"),
    "FRED_API_KEY": ("Macro: FRED series (free key)", "Phase 3"),
    "EODHD_API_KEY": ("Emerging markets fundamentals", "Later phase"),
    "DATABENTO_API_KEY": ("Professional US options data", "Optional"),
    "AISSTREAM_API_KEY": ("Maritime ship positions (free key)", "Phase 6"),
    "ALPACA_API_KEY": ("Paper trading", "Phase 6"),
    "ALPACA_SECRET_KEY": ("Paper trading", "Phase 6"),
}

DEFAULTS = {"base_currency": "USD", "data_source": "auto"}


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict:
    _ensure_dirs()
    if SETTINGS_FILE.exists():
        try:
            return {**DEFAULTS, **json.loads(SETTINGS_FILE.read_text())}
        except json.JSONDecodeError:
            pass
    return dict(DEFAULTS)


def save_settings(settings: dict) -> None:
    _ensure_dirs()
    clean = {k: settings[k] for k in DEFAULTS if k in settings}
    SETTINGS_FILE.write_text(json.dumps(clean, indent=2))


def get_key(name: str) -> str | None:
    val = os.environ.get(name)
    if not val and ENV_FILE.exists():
        val = dotenv_values(ENV_FILE).get(name)
    return val or None


def set_api_key(name: str, value: str) -> None:
    if name not in API_KEYS:
        raise ValueError(f"Unknown key {name}")
    ENV_FILE.touch(exist_ok=True)
    set_key(str(ENV_FILE), name, value.strip())
    os.environ[name] = value.strip()


def key_status() -> dict[str, bool]:
    return {k: bool(get_key(k)) for k in API_KEYS}


def require_key(name: str) -> str | None:
    """For paid modules: return the key, or None so the page can show
    'add your key in Settings' instead of crashing."""
    return get_key(name)
