"""Quality score (0-100) from four measures.

Plain English: a high-quality business earns a high return on the capital it uses,
has steady margins, isn't over-borrowed, and turns its reported profit into real cash.

| Measure | Formula | Scores 0 at | Scores 100 at |
|---|---|---|---|
| ROIC | EBIT x (1 - tax rate) / average invested capital, latest year | 0% | 20% or more |
| Margin stability | 1 - (std dev / mean) of operating margin over all years | variation of 50% or more | variation of 5% or less |
| Leverage | net debt / EBITDA, latest year | 4x or more | net cash (0x or less) |
| Cash conversion | total free cash flow / total net income, all years | 30% or less | 100% or more |

Each measure is scored on a straight line between those points; the overall score is
the average of the measures that could be computed. Banks and insurers are flagged:
EBITDA, FCF and net debt do not mean the same thing for them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.statements import latest, series

THRESHOLDS = {  # measure: (value scoring 0, value scoring 100)
    "roic": (0.0, 0.20),
    "margin_stability": (0.50, 0.05),  # coefficient of variation: lower is better
    "leverage": (4.0, 0.0),  # net debt / EBITDA: lower is better
    "cash_conversion": (0.30, 1.00),
}
NAMES = {"roic": "Return on invested capital", "margin_stability": "Margin variation (lower = steadier)",
         "leverage": "Leverage (net debt / EBITDA)", "cash_conversion": "Cash conversion (FCF / net income)"}


def linear_score(value: float, zero_at: float, full_at: float) -> float:
    if value is None or np.isnan(value):
        return np.nan
    frac = (value - zero_at) / (full_at - zero_at)
    return float(np.clip(frac, 0, 1) * 100)


@dataclass
class QualityResult:
    values: dict  # measure -> raw value
    scores: dict  # measure -> 0..100
    overall: float
    notes: list


def compute(stmts: dict, sector: str = "", default_tax: float = 0.21) -> QualityResult:
    inc, bal, cf = stmts["income"], stmts["balance"], stmts["cashflow"]
    notes = []
    if sector in ("Financial Services",):
        notes.append("Financial company: ROIC, leverage and FCF are not comparable for banks/insurers. Treat this score with caution.")
    vals = {}

    tax = latest(inc, "tax_rate")
    if np.isnan(tax):
        tax = default_tax
        notes.append(f"Tax rate unavailable; assumed {default_tax:.0%} for ROIC.")
    ic = series(bal, "invested_capital")
    ebit = latest(inc, "operating_income")
    if len(ic) >= 2:
        avg_ic = (ic.iloc[-1] + ic.iloc[-2]) / 2
    elif len(ic) == 1:
        avg_ic = ic.iloc[-1]
    else:
        avg_ic = np.nan
    vals["roic"] = ebit * (1 - tax) / avg_ic if avg_ic and avg_ic > 0 else np.nan

    m = series(inc, "operating_margin")
    if len(m) >= 3 and m.mean() > 0:
        vals["margin_stability"] = float(m.std(ddof=1) / m.mean())
    else:
        vals["margin_stability"] = np.nan
        if len(m) < 3:
            notes.append("Fewer than 3 years of margins: stability not scored.")

    nd, ebitda = latest(bal, "net_debt"), latest(inc, "ebitda")
    vals["leverage"] = nd / ebitda if ebitda and ebitda > 0 else np.nan

    fcf, ni = series(cf, "free_cash_flow"), series(inc, "net_income")
    common = fcf.index.intersection(ni.index)
    if len(common) and ni[common].sum() > 0:
        vals["cash_conversion"] = float(fcf[common].sum() / ni[common].sum())
    else:
        vals["cash_conversion"] = np.nan

    scores = {k: linear_score(v, *THRESHOLDS[k]) for k, v in vals.items()}
    valid = [s for s in scores.values() if not np.isnan(s)]
    overall = float(np.mean(valid)) if valid else np.nan
    return QualityResult(vals, scores, overall, notes)
