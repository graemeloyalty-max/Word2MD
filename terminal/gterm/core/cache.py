"""Local SQLite cache so repeated lookups don't hit the data source again.

Cached entries keep the original source name and retrieval time, so the UI's
"source / as of" labels stay honest when data is served from cache.
"""
from __future__ import annotations

import io
import pickle
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from . import config


def _db_path() -> Path:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return config.DATA_DIR / "cache.sqlite"


def _connect():
    con = sqlite3.connect(_db_path())
    con.execute(
        "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, stored_at REAL NOT NULL, payload BLOB NOT NULL)"
    )
    return con


def get(key: str, max_age_s: float):
    with closing(_connect()) as con:
        row = con.execute("SELECT stored_at, payload FROM cache WHERE key=?", (key,)).fetchone()
    if not row:
        return None
    stored_at, payload = row
    if time.time() - stored_at > max_age_s:
        return None
    return pickle.load(io.BytesIO(payload))


def put(key: str, value) -> None:
    buf = pickle.dumps(value)
    with closing(_connect()) as con, con:
        con.execute(
            "INSERT OR REPLACE INTO cache (key, stored_at, payload) VALUES (?,?,?)", (key, time.time(), buf)
        )


def clear() -> int:
    with closing(_connect()) as con, con:
        n = con.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        con.execute("DELETE FROM cache")
    return n


def stats() -> dict:
    with closing(_connect()) as con:
        n = con.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    size = _db_path().stat().st_size if _db_path().exists() else 0
    return {"entries": n, "size_kb": round(size / 1024, 1)}
