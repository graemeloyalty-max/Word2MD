"""M&A analytics: accretion/dilution and deal multiples.

Plain English:
- The acquirer pays the target's share price plus a **premium**, partly in new acquirer shares
  (**% stock**) and partly in cash (assumed borrowed at the pre-tax **cost of debt**; use the interest
  earned on cash instead if paying from cash on hand).
- **Pro forma earnings** = acquirer net income + target net income + after-tax synergies
  − after-tax interest on the cash part.
- **Pro forma shares** = acquirer shares + new shares issued (stock part / acquirer share price).
- **Accretion** = pro forma EPS / acquirer's stand-alone EPS − 1. Positive = accretive.
- **Breakeven synergies**: the pre-tax synergies at which the deal is neither accretive nor dilutive
  (negative means it's accretive even with none).
- Deal multiples at the offer price (P/E, EV/EBITDA, EV/Sales) can be compared with precedent deals you enter.

All amounts must be in one currency (the page converts the target into the acquirer's statement currency).
One-off deal fees and purchase-price accounting (e.g. amortisation of acquired intangibles) are not modelled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DealInputs:
    acq_price: float
    acq_shares: float
    acq_net_income: float
    tgt_price: float
    tgt_shares: float
    tgt_net_income: float
    premium: float = 0.30
    pct_stock: float = 0.5
    cost_of_debt: float = 0.06
    tax_rate: float = 0.25
    synergies: float = 0.0  # annual, pre-tax
    tgt_net_debt: float = 0.0
    tgt_ebitda: float = float("nan")
    tgt_revenue: float = float("nan")


@dataclass
class DealResult:
    offer_price: float
    purchase_equity: float
    stock_part: float
    cash_part: float
    new_shares: float
    acq_eps: float
    pro_forma_net_income: float
    pro_forma_shares: float
    pro_forma_eps: float
    accretion: float
    breakeven_synergies: float
    after_tax_financing_cost: float
    ownership_target_holders: float  # share of the combined company owned by former target holders
    offer_pe: float
    offer_ev: float
    offer_ev_ebitda: float
    offer_ev_sales: float


def analyse(d: DealInputs) -> DealResult:
    if d.acq_price <= 0 or d.acq_shares <= 0:
        raise ValueError("Acquirer price and shares must be positive.")
    offer = d.tgt_price * (1 + d.premium)
    purchase = offer * d.tgt_shares
    stock = purchase * d.pct_stock
    cash = purchase - stock
    new_shares = stock / d.acq_price
    fin = cash * d.cost_of_debt * (1 - d.tax_rate)
    pf_ni = d.acq_net_income + d.tgt_net_income + d.synergies * (1 - d.tax_rate) - fin
    pf_sh = d.acq_shares + new_shares
    eps_a = d.acq_net_income / d.acq_shares
    pf_eps = pf_ni / pf_sh
    accretion = pf_eps / eps_a - 1 if eps_a > 0 else float("nan")
    breakeven = (eps_a * pf_sh - d.acq_net_income - d.tgt_net_income + fin) / (1 - d.tax_rate)
    ev = purchase + d.tgt_net_debt
    return DealResult(offer, purchase, stock, cash, new_shares, eps_a, pf_ni, pf_sh, pf_eps, accretion, breakeven, fin,
                      new_shares / pf_sh, purchase / d.tgt_net_income if d.tgt_net_income > 0 else float("nan"), ev,
                      ev / d.tgt_ebitda if d.tgt_ebitda and d.tgt_ebitda > 0 else float("nan"),
                      ev / d.tgt_revenue if d.tgt_revenue and d.tgt_revenue > 0 else float("nan"))


def sensitivity(d: DealInputs, premiums=(0.0, 0.15, 0.30, 0.45, 0.60), stock_mix=(0.0, 0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """Accretion for a grid of premium (rows) x share of the price paid in stock (columns)."""
    rows = {}
    for p in premiums:
        rows[f"{p:.0%}"] = {f"{s:.0%} stock": analyse(DealInputs(**{**d.__dict__, "premium": p, "pct_stock": s})).accretion
                            for s in stock_mix}
    out = pd.DataFrame(rows).T
    out.index.name = "premium"
    return out


def implied_from_comps(comps: pd.DataFrame, tgt_ebitda: float, tgt_revenue: float, tgt_net_debt: float, tgt_shares: float) -> pd.DataFrame:
    """Implied target price per share at the median precedent EV/EBITDA and EV/Sales."""
    rows = []
    for col, metric in (("EV/EBITDA", tgt_ebitda), ("EV/Sales", tgt_revenue)):
        vals = pd.to_numeric(comps.get(col), errors="coerce").dropna() if col in comps else pd.Series(dtype=float)
        med = float(vals.median()) if len(vals) else np.nan
        ev = med * metric if metric and not np.isnan(med) else np.nan
        rows.append({"Multiple": col, "Deals": len(vals), "Median": med, "Implied EV": ev,
                     "Implied price / share": (ev - tgt_net_debt) / tgt_shares if not np.isnan(ev) and tgt_shares else np.nan})
    prem = pd.to_numeric(comps.get("Premium %"), errors="coerce").dropna() if "Premium %" in comps else pd.Series(dtype=float)
    rows.append({"Multiple": "Premium paid", "Deals": len(prem), "Median": float(prem.median()) / 100 if len(prem) else np.nan,
                 "Implied EV": np.nan, "Implied price / share": np.nan})
    return pd.DataFrame(rows)
