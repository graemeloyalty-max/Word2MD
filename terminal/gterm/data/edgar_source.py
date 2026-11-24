"""SEC EDGAR XBRL adapter: longer annual statement history for US filers (free).

yfinance holds about 4-5 years of statements. EDGAR's "companyfacts" API holds
every 10-K figure since ~2009. We pull a handful of core lines and merge them
into the normalised statements for the years yfinance doesn't cover.

SEC rules: every request must send a User-Agent naming you and a contact email
(set it in Settings), and stay under 10 requests/second.
Docs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
"""
from __future__ import annotations

import json
import urllib.request

import pandas as pd

from ..core import cache
from .base import DataSourceError

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# standard line -> (statement, [us-gaap concepts, first with data wins], is_flow)
CONCEPTS = {
    "revenue": ("income", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"], True),
    "operating_income": ("income", ["OperatingIncomeLoss"], True),
    "pretax_income": ("income", ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                                 "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"], True),
    "tax_provision": ("income", ["IncomeTaxExpenseBenefit"], True),
    "net_income": ("income", ["NetIncomeLoss"], True),
    "depreciation": ("income", ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization"], True),
    "total_assets": ("balance", ["Assets"], False),
    "total_equity": ("balance", ["StockholdersEquity"], False),
    "cash": ("balance", ["CashAndCashEquivalentsAtCarryingValue"], False),
    "operating_cash_flow": ("cashflow", ["NetCashProvidedByUsedInOperatingActivities"], True),
    "capex": ("cashflow", ["PaymentsToAcquirePropertyPlantAndEquipment"], True),
}


def _get_json(url: str, user_agent: str) -> dict:
    if not user_agent or "@" not in user_agent:
        raise DataSourceError("SEC EDGAR needs a User-Agent with your name and email. Set it in Settings.")
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except Exception as exc:
        raise DataSourceError(f"SEC EDGAR request failed ({url.split('/')[2]}): {exc}") from exc


def cik_for(ticker: str, user_agent: str) -> int:
    mapping = cache.get("edgar|tickers", 7 * 86400)
    if mapping is None:
        raw = _get_json(TICKERS_URL, user_agent)
        mapping = {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}
        cache.put("edgar|tickers", mapping)
    t = ticker.upper().replace(".", "-")
    if t not in mapping:
        raise DataSourceError(f"{ticker} is not an SEC filer (EDGAR covers US-listed companies).")
    return mapping[t]


def annual_lines(facts: dict) -> dict[str, pd.DataFrame]:
    """Turn a companyfacts JSON into {'income','balance','cashflow'} frames of annual (10-K) values."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    out: dict[str, dict] = {"income": {}, "balance": {}, "cashflow": {}}
    for line, (stmt, concepts, is_flow) in CONCEPTS.items():
        for concept in concepts:
            units = gaap.get(concept, {}).get("units", {})
            rows = units.get("USD")
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df = df[df["form"].isin(["10-K", "10-K/A"])].copy()
            df["end"] = pd.to_datetime(df["end"])
            if is_flow:  # keep full-year durations only (drop quarters reported inside 10-Ks)
                df["start"] = pd.to_datetime(df["start"])
                df = df[(df["end"] - df["start"]).dt.days.between(350, 380)]
            if df.empty:
                continue
            # The same period appears in several filings: keep the most recently filed value.
            df = df.sort_values("filed").drop_duplicates("end", keep="last")
            s = df.set_index("end")["val"].astype(float).sort_index()
            if line == "capex":
                s = -s.abs()
            out[stmt][line] = s
            break
    frames = {}
    for stmt, d in out.items():
        df = pd.DataFrame(d).T if d else pd.DataFrame()
        if not df.empty:
            df = df.reindex(sorted(df.columns), axis=1)
            if stmt == "cashflow" and {"operating_cash_flow", "capex"} <= set(df.index):
                df.loc["free_cash_flow"] = df.loc["operating_cash_flow"] + df.loc["capex"]
            if stmt == "income" and {"operating_income", "revenue"} <= set(df.index):
                df.loc["operating_margin"] = df.loc["operating_income"] / df.loc["revenue"]
        frames[stmt] = df
    return frames


def fetch_history(ticker: str, user_agent: str) -> dict[str, pd.DataFrame]:
    key = f"edgar|facts|{ticker.upper()}"
    facts = cache.get(key, 86400)
    if facts is None:
        facts = _get_json(FACTS_URL.format(cik=cik_for(ticker, user_agent)), user_agent)
        cache.put(key, facts)
    return annual_lines(facts)


def merge_history(stmts: dict, history: dict) -> tuple[dict, str]:
    """Add EDGAR years that are older than yfinance's earliest year. yfinance values win where both exist."""
    merged, years = {}, set()
    for k, df in stmts.items():
        h = history.get(k, pd.DataFrame())
        if h.empty or df.empty:
            merged[k] = df
            continue
        first = df.columns.min()
        # Fiscal year-ends can differ by a few days between sources; treat within 20 days as the same year.
        older = [c for c in h.columns if c < first - pd.Timedelta(days=20)]
        add = h[older].reindex(df.index.union(h.index))
        merged[k] = pd.concat([add, df.reindex(add.index)], axis=1).dropna(how="all")
        merged[k] = merged[k].reindex(sorted(merged[k].columns), axis=1)
        years.update(c.year for c in older)
    note = f"+ SEC EDGAR ({min(years)}-{max(years)})" if years else ""
    return merged, note
