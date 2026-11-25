"""Ownership and relationship maps.

Plain English:
- **Insider activity**: each insider transaction is classed as a *buy* (open-market purchase), a *sale*,
  or *other* (gifts, awards, option exercises, tax withholding). Net insider buying is usually read as
  a stronger signal than selling, because insiders sell for many routine reasons.
- **Ownership network** (one company): the company in the middle, its largest institutional and fund
  holders around it (bubble size = % of shares held), and its insiders coloured by net buying/selling.
- **Common ownership** (several companies): companies on an inner ring, holders outside, each holder
  drawn between the companies it owns. Shared holders show where the same institutions are exposed.

Data: holders come from regulatory filings (13F in the US), which are published with a delay of up to
~45 days after each quarter: check the "Date Reported" column.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def classify(text: str, transaction: str = "") -> str:
    t = f"{text} {transaction}".lower()
    if "sale" in t or "sell" in t or "sold" in t:
        return "sale"
    if "purchase" in t or "buy" in t or "bought" in t:
        return "buy"
    return "other"


def insider_activity(tx: pd.DataFrame, since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Per insider: shares and value bought, sold and net, over the period."""
    if tx is None or tx.empty:
        return pd.DataFrame(columns=["Insider", "Position", "bought_shares", "sold_shares", "net_shares", "net_value", "transactions"])
    df = tx.copy()
    df["date"] = pd.to_datetime(df.get("Start Date"), errors="coerce")
    if since is not None:
        df = df[df["date"] >= since]
    col = lambda c: df[c].fillna("").astype(str) if c in df.columns else pd.Series("", index=df.index)  # noqa: E731
    df["type"] = [classify(a, b) for a, b in zip(col("Text"), col("Transaction"))]
    df["Shares"] = pd.to_numeric(df.get("Shares"), errors="coerce").fillna(0)
    df["Value"] = pd.to_numeric(df.get("Value"), errors="coerce").fillna(0)
    rows = []
    for (who, pos), g in df.groupby(["Insider", "Position"], dropna=False):
        b, s = g[g["type"] == "buy"], g[g["type"] == "sale"]
        rows.append({"Insider": who, "Position": pos, "bought_shares": b["Shares"].sum(), "sold_shares": s["Shares"].sum(),
                     "net_shares": b["Shares"].sum() - s["Shares"].sum(), "net_value": b["Value"].sum() - s["Value"].sum(),
                     "transactions": len(g)})
    return pd.DataFrame(rows).sort_values("net_value")


def company_network(symbol: str, institutions: pd.DataFrame, funds: pd.DataFrame, insiders: pd.DataFrame, top: int = 10):
    """Nodes/edges for one company. Returns (nodes DataFrame [id,label,kind,size,x,y,color_value], edges [(a,b)])."""
    nodes = [{"id": symbol, "label": symbol, "kind": "company", "size": 40, "x": 0.0, "y": 0.0, "value": 0.0}]
    edges = []
    holders = []
    for kind, df in (("institution", institutions), ("fund", funds)):
        if df is not None and not df.empty and "Holder" in df:
            d = df.sort_values("pctHeld", ascending=False).head(top)
            holders += [(kind, r["Holder"], float(r.get("pctHeld", 0) or 0)) for _, r in d.iterrows()]
    ins = insiders.head(top) if insiders is not None else pd.DataFrame()
    total = len(holders) + len(ins)
    for i, (kind, name, pct) in enumerate(holders):
        a = 2 * math.pi * i / max(total, 1)
        nodes.append({"id": f"{kind}:{name}", "label": f"{name} ({pct:.1%})", "kind": kind, "size": 10 + 300 * pct,
                      "x": math.cos(a), "y": math.sin(a), "value": pct})
        edges.append((symbol, f"{kind}:{name}"))
    for j, (_, r) in enumerate(ins.iterrows()):
        a = 2 * math.pi * (len(holders) + j) / max(total, 1)
        nodes.append({"id": f"insider:{r['Insider']}", "label": f"{r['Insider']} ({r['Position']})", "kind": "insider",
                      "size": 12, "x": 0.65 * math.cos(a), "y": 0.65 * math.sin(a), "value": float(r["net_value"])})
        edges.append((symbol, f"insider:{r['Insider']}"))
    return pd.DataFrame(nodes), edges


def common_ownership(holdings: dict[str, pd.DataFrame], top: int = 10, min_companies: int = 1):
    """holdings: {symbol: institutions DataFrame}. Returns (nodes, edges, shared table)."""
    syms = list(holdings)
    n = len(syms)
    comp_pos = {s: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) if n > 1 else (0.0, 0.0)
                for i, s in enumerate(syms)}
    owners: dict[str, dict[str, float]] = {}
    for s, df in holdings.items():
        if df is None or df.empty or "Holder" not in df:
            continue
        for _, r in df.sort_values("pctHeld", ascending=False).head(top).iterrows():
            owners.setdefault(str(r["Holder"]), {})[s] = float(r.get("pctHeld", 0) or 0)
    nodes = [{"id": s, "label": s, "kind": "company", "size": 40, "x": comp_pos[s][0], "y": comp_pos[s][1], "value": 0.0} for s in syms]
    edges, shared = [], []
    singles: dict[str, int] = {}
    groups: dict[tuple, list] = {}  # holders owning the same set of companies would land on the same point: spread them out
    for h, owned in owners.items():
        if len(owned) > 1:
            groups.setdefault(tuple(sorted(owned)), []).append(h)
    for h, owned in sorted(owners.items(), key=lambda kv: -len(kv[1])):
        if len(owned) < min_companies:
            continue
        xs = [comp_pos[s][0] for s in owned]
        ys = [comp_pos[s][1] for s in owned]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        if len(owned) == 1:  # fan single-company holders out around their company, away from the centre
            s = next(iter(owned))
            k = singles.get(s, 0)
            singles[s] = k + 1
            if n > 1:
                base = math.atan2(comp_pos[s][1], comp_pos[s][0])
                ang = base + (k - (top - 1) / 2) * 0.28
                cx, cy = comp_pos[s][0] + 0.6 * math.cos(ang), comp_pos[s][1] + 0.6 * math.sin(ang)
            else:
                ang = 2 * math.pi * k / top
                cx, cy = math.cos(ang), math.sin(ang)
        else:
            peers = groups[tuple(sorted(owned))]
            if len(peers) > 1:
                k, m = peers.index(h), len(peers)
                rad = 0.12 + 0.03 * m
                cx, cy = cx + rad * math.cos(2 * math.pi * k / m), cy + rad * math.sin(2 * math.pi * k / m)
            shared.append({"Holder": h, "Companies": len(owned), **{s: owned.get(s, np.nan) for s in syms}})
        nodes.append({"id": f"h:{h}", "label": h, "kind": "shared" if len(owned) > 1 else "institution",
                      "size": 8 + 6 * len(owned), "x": cx, "y": cy, "value": len(owned)})
        edges += [(s, f"h:{h}") for s in owned]
    table = pd.DataFrame(shared).astype({s: float for s in syms}).sort_values("Companies", ascending=False) if shared else pd.DataFrame(columns=["Holder", "Companies", *syms])
    return pd.DataFrame(nodes), edges, table
