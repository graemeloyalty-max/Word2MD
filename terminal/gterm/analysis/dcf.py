"""Discounted cash flow (DCF) valuation, free cash flow to the firm (FCFF).

Plain English: a company is worth the cash it will generate in future, discounted
back to today at a rate that reflects risk. We:

1. start from the latest normalised free cash flow (operating cash flow - capex);
2. grow it at `growth_high` for `years_high` years, then fade the growth rate in a
   straight line down to `terminal_growth` by year `years_total`;
3. after that, assume it grows at `terminal_growth` for ever (the terminal value,
   Gordon growth formula: FCF_{N+1} / (WACC - g));
4. discount everything at the WACC (weighted average cost of capital);
5. enterprise value - debt + cash - minority interest = equity value; / shares = value per share.

WACC = E/(D+E) * cost of equity + D/(D+E) * cost of debt * (1 - tax rate)
cost of equity (CAPM + country risk) = risk-free + beta * equity risk premium + lambda * country risk premium
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DCFInputs:
    base_fcf: float  # latest normalised FCF, in financial currency
    growth_high: float = 0.08
    years_high: int = 5
    years_total: int = 10
    terminal_growth: float = 0.025
    risk_free: float = 0.042
    beta: float = 1.0
    equity_risk_premium: float = 0.045
    country_risk_premium: float = 0.0
    crp_exposure: float = 1.0  # "lambda": how exposed the company is to its country's risk
    pre_tax_cost_of_debt: float = 0.05
    tax_rate: float = 0.21
    equity_value_market: float = 1.0  # market cap (financial currency), for WACC weights
    debt: float = 0.0
    cash: float = 0.0
    minority_interest: float = 0.0
    shares: float = 1.0


@dataclass
class DCFResult:
    cost_of_equity: float
    after_tax_cost_of_debt: float
    weight_equity: float
    wacc: float
    projections: pd.DataFrame
    pv_explicit: float
    terminal_value: float
    pv_terminal: float
    enterprise_value: float
    equity_value: float
    value_per_share: float
    terminal_share: float  # fraction of EV that comes from the terminal value
    warnings: list = field(default_factory=list)


def default_terminal_growth(risk_free: float, cap: float = 0.025) -> float:
    """Long-run growth shouldn't exceed the risk-free rate of the cash-flow currency (a common rule of thumb)."""
    return round(max(0.0, min(cap, risk_free)), 4)


def cost_of_equity(i: DCFInputs) -> float:
    return i.risk_free + i.beta * i.equity_risk_premium + i.crp_exposure * i.country_risk_premium


def wacc(i: DCFInputs) -> tuple[float, float, float, float]:
    ke = cost_of_equity(i)
    kd = i.pre_tax_cost_of_debt * (1 - i.tax_rate)
    total = i.equity_value_market + max(i.debt, 0.0)
    we = i.equity_value_market / total if total > 0 else 1.0
    return ke, kd, we, we * ke + (1 - we) * kd


def growth_path(i: DCFInputs) -> np.ndarray:
    """Growth rate for each projection year 1..years_total."""
    g = np.full(i.years_total, i.growth_high, dtype=float)
    fade_years = i.years_total - i.years_high
    if fade_years > 0:
        # Straight-line fade, ending exactly at terminal growth in the final year.
        g[i.years_high:] = np.linspace(i.growth_high, i.terminal_growth, fade_years + 1)[1:]
    return g


def run(i: DCFInputs, discount_rate: float | None = None) -> DCFResult:
    ke, kd, we, w = wacc(i)
    r = discount_rate if discount_rate is not None else w
    warnings = []
    if r <= i.terminal_growth:
        raise ValueError(f"Discount rate ({r:.2%}) must be above terminal growth ({i.terminal_growth:.2%}).")
    if i.base_fcf <= 0:
        warnings.append("Base free cash flow is zero or negative: a DCF is not meaningful on this base. Try a multi-year average or use multiples.")
    g = growth_path(i)
    years = np.arange(1, i.years_total + 1)
    fcf = i.base_fcf * np.cumprod(1 + g)
    df = 1 / (1 + r) ** years
    pv = fcf * df
    tv = fcf[-1] * (1 + i.terminal_growth) / (r - i.terminal_growth)
    pv_tv = tv * df[-1]
    ev = pv.sum() + pv_tv
    equity = ev - i.debt + i.cash - i.minority_interest
    proj = pd.DataFrame({"year": years, "growth": g, "fcf": fcf, "discount_factor": df, "pv_fcf": pv})
    if r - i.terminal_growth < 0.02:
        warnings.append(f"The discount rate ({r:.2%}) is only {(r - i.terminal_growth) * 100:.1f} percentage points above terminal growth "
                        f"({i.terminal_growth:.2%}). The terminal value divides by that gap, so the result is unreliable: lower terminal "
                        "growth or check the risk-free rate and beta.")
    share_tv = pv_tv / ev if ev else np.nan
    if share_tv > 0.8:
        warnings.append(f"{share_tv:.0%} of the value comes from the terminal value, so the result is very sensitive to terminal growth and WACC.")
    return DCFResult(ke, kd, we, r, proj, float(pv.sum()), float(tv), float(pv_tv), float(ev), float(equity),
                     float(equity / i.shares), float(share_tv), warnings)


def sensitivity(i: DCFInputs, wacc_steps=(-0.01, -0.005, 0, 0.005, 0.01), g_steps=(-0.01, -0.005, 0, 0.005, 0.01)) -> pd.DataFrame:
    """Value per share for a grid of WACC (rows) and terminal growth (columns)."""
    base_w = wacc(i)[3]
    rows = {}
    for dw in wacc_steps:
        row = {}
        for dg in g_steps:
            j = DCFInputs(**{**i.__dict__, "terminal_growth": i.terminal_growth + dg})
            try:
                row[f"{j.terminal_growth:.1%}"] = run(j, base_w + dw).value_per_share
            except ValueError:
                row[f"{j.terminal_growth:.1%}"] = np.nan
        rows[f"{base_w + dw:.2%}"] = row
    out = pd.DataFrame(rows).T
    out.index.name = "WACC \\ g"
    return out


def implied_growth(i: DCFInputs, target_value_per_share: float, lo: float = -0.5, hi: float = 1.0) -> float | None:
    """Reverse DCF: the high-growth rate that makes value per share equal the market price (bisection)."""
    if i.base_fcf <= 0:
        return None

    def f(g):
        return run(DCFInputs(**{**i.__dict__, "growth_high": g})).value_per_share - target_value_per_share

    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) < 1e-6 * max(1.0, abs(target_value_per_share)):
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2
