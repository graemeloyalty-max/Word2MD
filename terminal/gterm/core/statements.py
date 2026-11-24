"""Normalise financial statements into one standard set of lines.

Different sources (and different accounting standards) label and classify
items differently. Every analysis model reads ONLY the standard lines below,
so a US GAAP filer and an IFRS filer are measured the same way.

Adjustments applied (each one is recorded and shown on the page):

1. Interest paid classified in financing cash flow (an IFRS option) is moved
   into operating cash flow, as US GAAP requires.
2. Interest paid classified in operating cash flow is left alone (already US GAAP style).
3. Interest received classified in investing cash flow (an IFRS option) is moved
   into operating cash flow, as US GAAP requires.
4. Purchases of intangibles (e.g. development costs capitalised under IFRS IAS 38,
   which US GAAP mostly expenses) are counted as capital expenditure, so free cash
   flow is comparable whichever way R&D was treated.
5. Total debt includes lease liabilities for every company (IFRS 16 and ASC 842
   both put most leases on the balance sheet).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# standard line -> candidate source labels, first match wins
INCOME_MAP = {
    "revenue": ["Total Revenue", "Operating Revenue", "Revenues"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income", "EBIT", "Total Operating Income As Reported"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "depreciation": ["Reconciled Depreciation", "Depreciation And Amortization"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "pretax_income": ["Pretax Income"],
    "tax_provision": ["Tax Provision"],
    "net_income": ["Net Income Common Stockholders", "Net Income", "Net Income From Continuing Operation Net Minority Interest"],
    "diluted_eps": ["Diluted EPS"],
    "diluted_shares": ["Diluted Average Shares", "Basic Average Shares"],
}
BALANCE_MAP = {
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "total_equity": ["Stockholders Equity", "Common Stock Equity"],
    "minority_interest": ["Minority Interest"],
    "cash": ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"],
    "total_debt": ["Total Debt"],
    "lease_liabilities": ["Capital Lease Obligations"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "invested_capital": ["Invested Capital"],
    "goodwill_intangibles": ["Goodwill And Other Intangible Assets"],
    "shares_outstanding": ["Ordinary Shares Number", "Share Issued"],
}
CASHFLOW_MAP = {
    "operating_cash_flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "capex_ppe": ["Capital Expenditure", "Purchase Of PPE"],
    "purchase_intangibles": ["Purchase Of Intangibles"],
    "interest_paid_cff": ["Interest Paid Cff"],
    "interest_received_cfi": ["Interest Received Cfi"],
    "dividends_paid": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
    "buybacks": ["Repurchase Of Capital Stock", "Common Stock Payments"],
    "stock_based_comp": ["Stock Based Compensation"],
}

LABELS = {
    "revenue": "Revenue", "gross_profit": "Gross profit", "operating_income": "Operating income (EBIT)",
    "ebitda": "EBITDA", "depreciation": "Depreciation & amortisation", "interest_expense": "Interest expense",
    "pretax_income": "Pre-tax income", "tax_provision": "Tax", "net_income": "Net income to shareholders",
    "diluted_eps": "Diluted EPS", "diluted_shares": "Diluted shares",
    "total_assets": "Total assets", "total_liabilities": "Total liabilities", "total_equity": "Shareholders' equity",
    "minority_interest": "Minority interest", "cash": "Cash & short-term investments", "total_debt": "Total debt (incl. leases)",
    "lease_liabilities": "  of which lease liabilities", "current_assets": "Current assets",
    "current_liabilities": "Current liabilities", "invested_capital": "Invested capital",
    "goodwill_intangibles": "Goodwill & intangibles", "shares_outstanding": "Shares outstanding",
    "operating_cash_flow": "Operating cash flow (normalised)", "capex": "Capex (incl. intangibles)",
    "free_cash_flow": "Free cash flow", "dividends_paid": "Dividends paid", "buybacks": "Share buybacks",
    "stock_based_comp": "Stock-based compensation", "net_debt": "Net debt",
    "operating_margin": "Operating margin", "tax_rate": "Effective tax rate",
}


def _pick(raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    out = {}
    for std, candidates in mapping.items():
        for c in candidates:
            if c in raw.index and raw.loc[c].notna().any():
                out[std] = pd.to_numeric(raw.loc[c], errors="coerce")
                break
    df = pd.DataFrame(out).T
    if df.empty:
        return df
    df.columns = pd.DatetimeIndex(df.columns)
    return df.reindex(sorted(df.columns), axis=1)  # oldest -> newest


def normalise(income: pd.DataFrame, balance: pd.DataFrame, cashflow: pd.DataFrame) -> tuple[dict, list[str]]:
    """Return ({'income','balance','cashflow'} standard frames, list of adjustments made)."""
    inc = _pick(income, INCOME_MAP)
    bal = _pick(balance, BALANCE_MAP)
    cf = _pick(cashflow, CASHFLOW_MAP)
    notes: list[str] = []

    # Keep only fiscal years where the core income line exists (drops stub columns).
    if "revenue" in inc.index:
        inc = inc.loc[:, inc.loc["revenue"].notna()]

    if not cf.empty and "operating_cash_flow" in cf.index:
        cfo = cf.loc["operating_cash_flow"].copy()
        if "interest_paid_cff" in cf.index and cf.loc["interest_paid_cff"].notna().any():
            paid = cf.loc["interest_paid_cff"].fillna(0)  # negative number (cash outflow)
            cfo = cfo + paid
            notes.append("Interest paid was in financing cash flow (IFRS option): moved to operating cash flow.")
        if "interest_received_cfi" in cf.index and cf.loc["interest_received_cfi"].notna().any():
            cfo = cfo + cf.loc["interest_received_cfi"].fillna(0)
            notes.append("Interest received was in investing cash flow (IFRS option): moved to operating cash flow.")
        cf.loc["operating_cash_flow"] = cfo

        capex = -cf.loc["capex_ppe"].abs() if "capex_ppe" in cf.index else pd.Series(0.0, index=cf.columns)
        if "purchase_intangibles" in cf.index and cf.loc["purchase_intangibles"].notna().any():
            capex = capex - cf.loc["purchase_intangibles"].abs().fillna(0)
            notes.append("Purchases of intangibles (e.g. capitalised development costs) counted as capex.")
        cf.loc["capex"] = capex
        cf.loc["free_cash_flow"] = cfo + capex
        cf = cf.drop(index=[i for i in ["capex_ppe", "purchase_intangibles", "interest_paid_cff", "interest_received_cfi"] if i in cf.index])

    if not bal.empty:
        if "total_debt" not in bal.index:
            bal.loc["total_debt"] = 0.0
        if "cash" in bal.index:
            bal.loc["net_debt"] = bal.loc["total_debt"].fillna(0) - bal.loc["cash"].fillna(0)
        if "invested_capital" not in bal.index and "total_equity" in bal.index:
            cash = bal.loc["cash"].fillna(0) if "cash" in bal.index else 0.0
            bal.loc["invested_capital"] = bal.loc["total_equity"] + bal.loc["total_debt"].fillna(0) - cash
            notes.append("Invested capital computed as equity + debt - cash.")
        if "lease_liabilities" in bal.index and bal.loc["lease_liabilities"].notna().any():
            notes.append("Total debt includes lease liabilities (IFRS 16 / ASC 842).")

    if not inc.empty:
        if "revenue" in inc.index and "operating_income" in inc.index:
            inc.loc["operating_margin"] = inc.loc["operating_income"] / inc.loc["revenue"]
        if "tax_provision" in inc.index and "pretax_income" in inc.index:
            rate = inc.loc["tax_provision"] / inc.loc["pretax_income"]
            inc.loc["tax_rate"] = rate.where((rate > 0) & (rate < 0.6))
        if "ebitda" not in inc.index and {"operating_income", "depreciation"} <= set(inc.index):
            inc.loc["ebitda"] = inc.loc["operating_income"] + inc.loc["depreciation"]
            notes.append("EBITDA computed as EBIT + depreciation & amortisation.")

    return {"income": inc, "balance": bal, "cashflow": cf}, notes


def latest(df: pd.DataFrame, line: str, default=np.nan) -> float:
    """Most recent non-missing value of a line."""
    if df is None or df.empty or line not in df.index:
        return default
    s = df.loc[line].dropna()
    return float(s.iloc[-1]) if len(s) else default


def series(df: pd.DataFrame, line: str) -> pd.Series:
    if df is None or df.empty or line not in df.index:
        return pd.Series(dtype=float)
    return df.loc[line].dropna().astype(float)
