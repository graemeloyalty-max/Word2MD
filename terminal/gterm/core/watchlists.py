"""Watchlists and price alerts, stored in a local SQLite file."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass

from . import config


def _connect():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DATA_DIR / "workspace.sqlite")
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS watchlist (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL);
        CREATE TABLE IF NOT EXISTS watchlist_item (
            watchlist_id INTEGER NOT NULL REFERENCES watchlist(id) ON DELETE CASCADE,
            symbol TEXT NOT NULL,
            PRIMARY KEY (watchlist_id, symbol));
        CREATE TABLE IF NOT EXISTS alert (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL CHECK (direction IN ('above','below')),
            level REAL NOT NULL);
        """
    )
    return con


def list_watchlists() -> list[str]:
    with closing(_connect()) as con:
        return [r[0] for r in con.execute("SELECT name FROM watchlist ORDER BY id")]


def create_watchlist(name: str, symbols: list[str] | None = None) -> None:
    name = name.strip()
    if not name:
        raise ValueError("Watchlist name is empty")
    with closing(_connect()) as con, con:
        con.execute("INSERT OR IGNORE INTO watchlist(name) VALUES (?)", (name,))
    for s in symbols or []:
        add_symbol(name, s)


def delete_watchlist(name: str) -> None:
    with closing(_connect()) as con, con:
        con.execute("DELETE FROM watchlist WHERE name=?", (name,))


def _id(con, name: str) -> int:
    row = con.execute("SELECT id FROM watchlist WHERE name=?", (name,)).fetchone()
    if not row:
        raise KeyError(name)
    return row[0]


def symbols(name: str) -> list[str]:
    with closing(_connect()) as con:
        wid = _id(con, name)
        return [r[0] for r in con.execute("SELECT symbol FROM watchlist_item WHERE watchlist_id=? ORDER BY rowid", (wid,))]


def add_symbol(name: str, symbol: str) -> None:
    symbol = symbol.strip().upper()
    if not symbol:
        return
    with closing(_connect()) as con, con:
        con.execute("INSERT OR IGNORE INTO watchlist_item VALUES (?,?)", (_id(con, name), symbol))


def remove_symbol(name: str, symbol: str) -> None:
    with closing(_connect()) as con, con:
        con.execute("DELETE FROM watchlist_item WHERE watchlist_id=? AND symbol=?", (_id(con, name), symbol))


@dataclass
class Alert:
    id: int
    symbol: str
    direction: str
    level: float

    def triggered(self, price: float) -> bool:
        return price >= self.level if self.direction == "above" else price <= self.level


def add_alert(symbol: str, direction: str, level: float) -> None:
    with closing(_connect()) as con, con:
        con.execute("INSERT INTO alert(symbol, direction, level) VALUES (?,?,?)", (symbol.upper(), direction, float(level)))


def list_alerts() -> list[Alert]:
    with closing(_connect()) as con:
        return [Alert(*r) for r in con.execute("SELECT id, symbol, direction, level FROM alert ORDER BY id")]


def delete_alert(alert_id: int) -> None:
    with closing(_connect()) as con, con:
        con.execute("DELETE FROM alert WHERE id=?", (alert_id,))


def seed_defaults() -> None:
    """First-run starter lists so the app isn't empty."""
    if list_watchlists():
        return
    create_watchlist("US large caps", ["AAPL", "MSFT", "NVDA", "JPM", "SPY"])
    create_watchlist("Europe", ["SAP.DE", "ASML.AS", "MC.PA", "NESN.SW", "SHEL.L"])
    create_watchlist("Macro & FX", ["^GSPC", "^STOXX50E", "EURUSD=X", "USDZAR=X", "BTC-USD"])


# ---------------------------------------------------------------- portfolios (Phase 3)

def _pf_connect():
    con = _connect()
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS portfolio (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL);
        CREATE TABLE IF NOT EXISTS holding (
            portfolio_id INTEGER NOT NULL REFERENCES portfolio(id) ON DELETE CASCADE,
            symbol TEXT NOT NULL,
            quantity REAL NOT NULL,
            cost_price REAL,              -- per share, in the share's trading currency (optional)
            PRIMARY KEY (portfolio_id, symbol));
        """
    )
    return con


def list_portfolios() -> list[str]:
    with closing(_pf_connect()) as con:
        return [r[0] for r in con.execute("SELECT name FROM portfolio ORDER BY id")]


def save_portfolio(name: str, holdings: list[dict]) -> None:
    """Replace a portfolio's holdings. Each holding: {symbol, quantity, cost_price (optional)}."""
    name = name.strip()
    if not name:
        raise ValueError("Portfolio name is empty")
    clean = {}
    for h in holdings:
        sym = str(h.get("symbol") or "").strip().upper()
        qty = h.get("quantity")
        if not sym or qty is None or (isinstance(qty, float) and qty != qty) or float(qty) == 0:
            continue
        cost = h.get("cost_price")
        cost = None if cost is None or (isinstance(cost, float) and cost != cost) else float(cost)
        clean[sym] = (float(qty), cost)
    with closing(_pf_connect()) as con, con:
        con.execute("INSERT OR IGNORE INTO portfolio(name) VALUES (?)", (name,))
        pid = con.execute("SELECT id FROM portfolio WHERE name=?", (name,)).fetchone()[0]
        con.execute("DELETE FROM holding WHERE portfolio_id=?", (pid,))
        con.executemany("INSERT INTO holding VALUES (?,?,?,?)", [(pid, s, q, c) for s, (q, c) in clean.items()])


def get_holdings(name: str) -> list[dict]:
    with closing(_pf_connect()) as con:
        row = con.execute("SELECT id FROM portfolio WHERE name=?", (name,)).fetchone()
        if not row:
            raise KeyError(name)
        return [{"symbol": s, "quantity": q, "cost_price": c}
                for s, q, c in con.execute("SELECT symbol, quantity, cost_price FROM holding WHERE portfolio_id=? ORDER BY rowid", (row[0],))]


def delete_portfolio(name: str) -> None:
    with closing(_pf_connect()) as con, con:
        con.execute("DELETE FROM portfolio WHERE name=?", (name,))


def seed_portfolio() -> None:
    if list_portfolios():
        return
    save_portfolio("Sample global portfolio", [
        {"symbol": "AAPL", "quantity": 20}, {"symbol": "MSFT", "quantity": 10}, {"symbol": "JPM", "quantity": 15},
        {"symbol": "SAP.DE", "quantity": 25}, {"symbol": "ASML.AS", "quantity": 5}, {"symbol": "NESN.SW", "quantity": 40},
        {"symbol": "SHEL.L", "quantity": 150}, {"symbol": "SPY", "quantity": 5},
    ])
