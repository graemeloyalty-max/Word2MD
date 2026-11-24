"""Option pricing (Black-Scholes-Merton), Greeks, implied volatility and the volatility surface.

Plain English:
- **Black-Scholes** prices a European option from the share price S, strike K, time to expiry T
  (years), risk-free rate r, dividend yield q and volatility sigma.
- **Implied volatility (IV)** runs the formula backwards: the volatility at which the formula gives
  the option's market price. It's the market's forecast of how much the share will move.
- **Volatility surface**: IV for every strike and expiry. A flat surface would mean the market
  agrees with Black-Scholes; in practice puts on falls cost more (the "skew"), and short-dated IV
  jumps around events.
- We use out-of-the-money options only (puts below the spot price, calls above), because they
  trade more actively and their prices carry less noise.

Greeks: delta (price change per 1 unit move in S), gamma (change in delta), vega (per 1 vol point,
i.e. 0.01), theta (per calendar day), rho (per 1 percentage point of rates).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SQRT2 = math.sqrt(2.0)


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _d1d2(S, K, T, r, sigma, q):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, kind: str = "call", q: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
        return intrinsic
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    if kind == "call":
        return S * math.exp(-q * T) * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * math.exp(-q * T) * _ncdf(-d1)


def greeks(S, K, T, r, sigma, kind="call", q=0.0) -> dict:
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    disc_q, disc_r = math.exp(-q * T), math.exp(-r * T)
    pdf = _npdf(d1)
    gamma = disc_q * pdf / (S * sigma * math.sqrt(T))
    vega = S * disc_q * pdf * math.sqrt(T) / 100
    if kind == "call":
        delta = disc_q * _ncdf(d1)
        theta = (-S * disc_q * pdf * sigma / (2 * math.sqrt(T)) - r * K * disc_r * _ncdf(d2) + q * S * disc_q * _ncdf(d1)) / 365
        rho = K * T * disc_r * _ncdf(d2) / 100
    else:
        delta = -disc_q * _ncdf(-d1)
        theta = (-S * disc_q * pdf * sigma / (2 * math.sqrt(T)) + r * K * disc_r * _ncdf(-d2) - q * S * disc_q * _ncdf(-d1)) / 365
        rho = -K * T * disc_r * _ncdf(-d2) / 100
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(price: float, S: float, K: float, T: float, r: float, kind: str = "call", q: float = 0.0,
                lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-7) -> float:
    """Bisection on sigma. Returns NaN when the price is outside no-arbitrage bounds."""
    if T <= 0 or price <= 0:
        return float("nan")
    lower = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0) if kind == "call" else max(K * math.exp(-r * T) - S * math.exp(-q * T), 0)
    upper = S * math.exp(-q * T) if kind == "call" else K * math.exp(-r * T)
    if not (lower - 1e-9 < price < upper):
        return float("nan")
    f_lo = bs_price(S, K, T, r, lo, kind, q) - price
    f_hi = bs_price(S, K, T, r, hi, kind, q) - price
    if f_lo * f_hi > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = bs_price(S, K, T, r, mid, kind, q) - price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def surface_points(chain: pd.DataFrame, spot: float, r: float, q: float = 0.0, asof: pd.Timestamp | None = None,
                   min_days: int = 3, moneyness=(0.7, 1.3), min_price: float = 0.05) -> pd.DataFrame:
    """chain columns: expiry, strike, type ('call'/'put'), bid, ask, last. Returns OTM points with our IV.
    If the chain already carries a quoted IV (column 'quoted_iv', e.g. Deribit's mark IV) and no usable
    price, that IV is used and flagged."""
    asof = pd.Timestamp(asof or pd.Timestamp.utcnow().tz_localize(None))
    df = chain.copy()
    df["T"] = (pd.to_datetime(df["expiry"]) - asof).dt.total_seconds() / (365.25 * 86400)
    df["days"] = df["T"] * 365.25
    df["moneyness"] = df["strike"] / spot
    otm = ((df["type"] == "put") & (df["strike"] < spot)) | ((df["type"] == "call") & (df["strike"] >= spot))
    df = df[otm & (df["days"] >= min_days) & df["moneyness"].between(*moneyness)].copy()
    bid, ask = df.get("bid", pd.Series(np.nan, index=df.index)), df.get("ask", pd.Series(np.nan, index=df.index))
    mid = np.where((bid > 0) & (ask > 0) & (ask >= bid), (bid + ask) / 2, df.get("last", np.nan))
    df["price"] = mid
    ivs, how = [], []
    for row in df.itertuples():
        iv = float("nan")
        if row.price and row.price >= min_price:
            iv = implied_vol(float(row.price), spot, float(row.strike), float(row.T), r, row.type, q)
        if np.isnan(iv) and "quoted_iv" in df.columns and pd.notna(getattr(row, "quoted_iv", np.nan)):
            iv, src = float(row.quoted_iv), "quoted"
        else:
            src = "computed"
        ivs.append(iv)
        how.append(src)
    df["iv"], df["iv_source"] = ivs, how
    return df[df["iv"].between(0.01, 3.0)].sort_values(["T", "strike"])


def term_structure(points: pd.DataFrame, spot: float) -> pd.Series:
    """At-the-money IV per expiry: the IV of the strike nearest the spot price."""
    out = {}
    for exp, g in points.groupby("expiry"):
        i = (g["strike"] - spot).abs().idxmin()
        out[pd.Timestamp(exp)] = g.loc[i, "iv"]
    return pd.Series(out).sort_index()
