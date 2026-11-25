"""Multi-leg option strategies: payoff diagrams, breakevens, max profit/loss and net Greeks.

Plain English: a strategy is a list of legs (buy or sell a call, a put, or the shares themselves).
- **Payoff at expiry**: what the position is worth on expiry day for each possible share price,
  minus what it cost to set up. Options are then worth only their intrinsic value.
- **Value today**: the same, but options priced with Black-Scholes using the time left and the
  volatility you enter, so you can see how the position behaves before expiry.
- **Breakevens**: share prices where the expiry P&L crosses zero.
- **Max profit / loss**: over the price range shown; "unlimited" when the P&L is still rising/falling at the edge
  (e.g. a short call).
- **Net Greeks**: sum of each leg's Greeks x quantity x contract multiplier (sign flips for sold legs).

Premiums are per share; one contract usually covers 100 shares (`multiplier`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import options as op


@dataclass
class Leg:
    kind: str  # "call" | "put" | "stock"
    side: str  # "long" | "short"
    strike: float = 0.0  # ignored for stock
    premium: float = 0.0  # per share paid (long) or received (short); for stock: entry price
    quantity: float = 1.0  # contracts (or lots of `multiplier` shares for stock)
    days: float = 30.0  # days to expiry (options)
    iv: float = 0.25  # implied volatility used for "value today" and Greeks

    @property
    def sign(self) -> int:
        return 1 if self.side == "long" else -1


def leg_value_at_expiry(leg: Leg, s):
    s = np.asarray(s, dtype=float)
    if leg.kind == "stock":
        return s
    return np.maximum(s - leg.strike, 0) if leg.kind == "call" else np.maximum(leg.strike - s, 0)


def payoff_at_expiry(legs: list[Leg], s, multiplier: int = 100):
    total = np.zeros_like(np.asarray(s, dtype=float))
    for leg in legs:
        total += leg.sign * leg.quantity * multiplier * (leg_value_at_expiry(leg, s) - leg.premium)
    return total


def value_today(legs: list[Leg], s, r: float = 0.04, q: float = 0.0, days_passed: float = 0.0, multiplier: int = 100):
    s = np.asarray(s, dtype=float)
    total = np.zeros_like(s)
    for leg in legs:
        if leg.kind == "stock":
            v = s
        else:
            t = max(leg.days - days_passed, 0) / 365.0
            v = np.array([op.bs_price(x, leg.strike, t, r, leg.iv, leg.kind, q) for x in s])
        total += leg.sign * leg.quantity * multiplier * (v - leg.premium)
    return total


def net_greeks(legs: list[Leg], spot: float, r: float = 0.04, q: float = 0.0, multiplier: int = 100) -> dict:
    g = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    for leg in legs:
        scale = leg.sign * leg.quantity * multiplier
        if leg.kind == "stock":
            g["delta"] += scale
            continue
        lg = op.greeks(spot, leg.strike, max(leg.days, 1e-6) / 365.0, r, leg.iv, leg.kind, q)
        for k in g:
            g[k] += scale * lg[k]
    return g


def net_premium(legs: list[Leg], multiplier: int = 100) -> float:
    """Cash paid to open (+) or received (−), options only (stock legs excluded)."""
    return sum(leg.sign * leg.quantity * multiplier * leg.premium for leg in legs if leg.kind != "stock")


def breakevens(s, pnl) -> list[float]:
    s, pnl = np.asarray(s), np.asarray(pnl)
    out = []
    for i in range(len(s) - 1):
        a, b = pnl[i], pnl[i + 1]
        if a == 0:
            out.append(float(s[i]))
        elif a * b < 0:
            out.append(float(s[i] - a * (s[i + 1] - s[i]) / (b - a)))  # linear interpolation
    return sorted(set(round(x, 4) for x in out))


def max_profit_loss(s, pnl) -> tuple:
    """(max profit, max loss) over the range. None = unlimited: the P&L is still rising (profit) or falling (loss)
    at the top of the price range. Prices can't fall below zero, so the low end is always bounded."""
    pnl = np.asarray(pnl)
    slope_hi = pnl[-1] - pnl[-2]
    profit = None if slope_hi > 1e-9 else float(pnl.max())
    loss = None if slope_hi < -1e-9 else float(pnl.min())
    return profit, loss


def price_grid(spot: float, legs: list[Leg], width: float = 0.5, n: int = 401):
    strikes = [leg.strike for leg in legs if leg.kind != "stock" and leg.strike > 0]
    lo = min([spot * (1 - width)] + [k * 0.8 for k in strikes])
    hi = max([spot * (1 + width)] + [k * 1.2 for k in strikes])
    return np.linspace(max(lo, 0.0), hi, n)


def preset(name: str, spot: float, days: int = 30, iv: float = 0.25, r: float = 0.04) -> list[Leg]:
    """Common strategies at round strikes around the spot price, premiums from Black-Scholes."""
    step = 10 ** max(0, int(np.log10(max(spot, 1))) - 1)  # 1, 10, 100…
    atm = round(spot / step) * step
    w = step * max(1, round(spot * 0.05 / step))  # ~5% wing

    def opt(kind, side, k, qty=1):
        p = op.bs_price(spot, k, days / 365, r, iv, kind)
        return Leg(kind, side, float(k), round(p, 2), qty, days, iv)

    stock = Leg("stock", "long", 0.0, spot, 1, days, iv)
    table = {
        "Long call": [opt("call", "long", atm)],
        "Long put": [opt("put", "long", atm)],
        "Covered call": [stock, opt("call", "short", atm + w)],
        "Protective put": [stock, opt("put", "long", atm - w)],
        "Bull call spread": [opt("call", "long", atm), opt("call", "short", atm + w)],
        "Bear put spread": [opt("put", "long", atm), opt("put", "short", atm - w)],
        "Long straddle": [opt("call", "long", atm), opt("put", "long", atm)],
        "Long strangle": [opt("call", "long", atm + w), opt("put", "long", atm - w)],
        "Iron condor": [opt("put", "long", atm - 2 * w), opt("put", "short", atm - w), opt("call", "short", atm + w),
                        opt("call", "long", atm + 2 * w)],
        "Butterfly (calls)": [opt("call", "long", atm - w), opt("call", "short", atm, 2), opt("call", "long", atm + w)],
    }
    return table[name]


PRESETS = ["Long call", "Long put", "Covered call", "Protective put", "Bull call spread", "Bear put spread",
           "Long straddle", "Long strangle", "Iron condor", "Butterfly (calls)"]
