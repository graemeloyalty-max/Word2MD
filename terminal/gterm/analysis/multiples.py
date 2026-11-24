"""Relative valuation: the company's multiples against regional sector peers.

Plain English: if similar companies trade at 20x earnings and this one trades at 15x,
it looks cheap on that measure, or the market expects less from it. We compare four
multiples and work out the share price each one implies at the peer median.

- P/E = market cap / net income             (equity multiple)
- P/B = market cap / book equity            (equity multiple)
- EV/EBITDA = enterprise value / EBITDA     (firm multiple)
- EV/Sales = enterprise value / revenue     (firm multiple)

Implied price from an equity multiple: price x (peer median / company multiple).
Implied price from a firm multiple: implied EV = EV x (peer median / company multiple),
then implied market cap = implied EV - (EV - market cap), so net debt is held constant.
Using medians, not averages, stops one extreme peer distorting the result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.base import KeyStats

MULTIPLES = {"pe": "P/E", "ev_ebitda": "EV/EBITDA", "pb": "P/B", "ev_sales": "EV/Sales"}
EQUITY_MULTIPLES = {"pe", "pb"}

# Default peer groups by sector and region. Editable on the page.
DEFAULT_PEERS = {
    "Technology": {"US": ["AAPL", "MSFT", "NVDA", "ORCL", "CSCO", "ADBE", "CRM", "AVGO"],
                   "Europe": ["SAP.DE", "ASML.AS", "IFX.DE", "CAP.PA", "STMPA.PA", "ADYEN.AS", "DSY.PA", "NOKIA.HE"]},
    "Communication Services": {"US": ["GOOGL", "META", "NFLX", "DIS", "VZ", "T", "TMUS"],
                               "Europe": ["DTE.DE", "ORA.PA", "VOD.L", "TEF.MC", "UMG.AS", "PUB.PA"]},
    "Consumer Cyclical": {"US": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW"],
                          "Europe": ["MC.PA", "RMS.PA", "KER.PA", "ITX.MC", "ADS.DE", "BMW.DE", "MBG.DE", "CFR.SW"]},
    "Consumer Defensive": {"US": ["PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL"],
                           "Europe": ["NESN.SW", "ULVR.L", "OR.PA", "BN.PA", "DGE.L", "ABI.BR", "HEIA.AS"]},
    "Healthcare": {"US": ["JNJ", "LLY", "UNH", "PFE", "MRK", "ABBV", "TMO"],
                   "Europe": ["NOVO-B.CO", "ROG.SW", "NOVN.SW", "AZN.L", "SAN.PA", "GSK.L", "BAYN.DE"]},
    "Financial Services": {"US": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK"],
                           "Europe": ["HSBA.L", "BNP.PA", "SAN.MC", "ALV.DE", "INGA.AS", "UBSG.SW", "ZURN.SW"]},
    "Industrials": {"US": ["GE", "CAT", "HON", "UNP", "RTX", "DE", "LMT"],
                    "Europe": ["SIE.DE", "AIR.PA", "SU.PA", "ABBN.SW", "RR.L", "SAF.PA", "VOLV-B.ST"]},
    "Energy": {"US": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX"],
               "Europe": ["SHEL.L", "TTE.PA", "BP.L", "ENI.MI", "EQNR.OL", "REP.MC"]},
    "Basic Materials": {"US": ["LIN", "SHW", "APD", "FCX", "NEM", "DOW"],
                        "Europe": ["AI.PA", "BAS.DE", "RIO.L", "GLEN.L", "SIKA.SW", "AAL.L"]},
    "Utilities": {"US": ["NEE", "DUK", "SO", "D", "AEP", "EXC"],
                  "Europe": ["IBE.MC", "ENEL.MI", "EOAN.DE", "ENGI.PA", "NG.L", "RWE.DE"]},
    "Real Estate": {"US": ["PLD", "AMT", "EQIX", "SPG", "O", "PSA"],
                    "Europe": ["VNA.DE", "URW.PA", "SEGRO.L", "LAND.L", "BLND.L"]},
}
EUROPE = {"Germany", "France", "Netherlands", "Switzerland", "United Kingdom", "Spain", "Italy", "Sweden", "Denmark",
          "Norway", "Finland", "Belgium", "Ireland", "Austria", "Portugal", "Luxembourg"}


def region_of(country: str) -> str:
    return "Europe" if country in EUROPE else "US"


def default_peers(symbol: str, sector: str, country: str) -> list[str]:
    peers = DEFAULT_PEERS.get(sector, {}).get(region_of(country), [])
    return [p for p in peers if p.upper() != symbol.upper()][:7]


def peer_table(target: KeyStats, peers: list[KeyStats]) -> pd.DataFrame:
    rows = []
    for k in [target, *peers]:
        rows.append({"Symbol": k.symbol, "Name": k.name, "Country": k.country, "Ccy": k.currency,
                     "Mkt cap": k.market_cap, **{MULTIPLES[m]: getattr(k, m) for m in MULTIPLES}})
    return pd.DataFrame(rows)


def implied_prices(target: KeyStats, peers: list[KeyStats], price: float) -> pd.DataFrame:
    rows = []
    for m, label in MULTIPLES.items():
        vals = [getattr(p, m) for p in peers if getattr(p, m)]
        med = float(np.median(vals)) if vals else np.nan
        own = getattr(target, m)
        implied = np.nan
        if own and not np.isnan(med):
            if m in EQUITY_MULTIPLES:
                implied = price * med / own
            elif target.enterprise_value and target.market_cap:
                ev, mcap = target.enterprise_value, target.market_cap
                implied_mcap = ev * med / own - (ev - mcap)
                implied = price * implied_mcap / mcap if implied_mcap > 0 else np.nan
        rows.append({"Multiple": label, "Company": own, "Peer median": med, "Peers used": len(vals),
                     "Premium / (discount) to peers": (own / med - 1) if own and med else np.nan,
                     "Implied price": implied})
    return pd.DataFrame(rows)
