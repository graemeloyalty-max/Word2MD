"""The terminal's own functions, exposed as tools the AI can call.

Every result includes `source` and `as_of`, so answers can cite where each number came
from (ground rule 2). Results are kept compact: tokens cost quota on free tiers.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from ..analysis import country as ctry
from ..analysis import dcf, multiples, quality
from ..core import fx, indicators
from ..core.statements import latest, series
from ..data import macro, news, registry
from ..data.base import DataSourceError


def _r(x, n=4):
    try:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)
    except (TypeError, ValueError):
        return None


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def get_quote(symbol: str) -> dict:
    q = registry.quote(symbol)
    return {"symbol": q.symbol, "name": q.name, "price": _r(q.price), "currency": q.currency,
            "change_pct": _r(q.change_pct, 2), "price_date": f"{q.price_date:%Y-%m-%d}", "source": q.source, "as_of": _now()}


def get_price_performance(symbol: str, period: str = "1y") -> dict:
    h = registry.prices(symbol, period, "1d")
    c = h.frame["Close"]
    return {"symbol": symbol, "period": period, "currency": h.currency, "last_close": _r(h.last_close),
            "total_return": _r(indicators.total_return(c)), "annualised_volatility": _r(indicators.annualised_volatility(c)),
            "max_drawdown": _r(indicators.max_drawdown(c)), "start": f"{c.index[0]:%Y-%m-%d}", "end": f"{c.index[-1]:%Y-%m-%d}",
            "source": h.source, "as_of": _now()}


def get_fundamentals(symbol: str) -> dict:
    f = registry.fundamentals(symbol)
    inc, bal, cf = f.statements["income"], f.statements["balance"], f.statements["cashflow"]
    lines = {"revenue": inc, "operating_income": inc, "net_income": inc, "operating_margin": inc,
             "free_cash_flow": cf, "net_debt": bal, "total_equity": bal}
    years = {}
    for line, df in lines.items():
        for d, v in series(df, line).items():
            years.setdefault(str(d.year), {})[line] = _r(v / 1e6 if line != "operating_margin" else v)
    return {"symbol": symbol, "name": f.name, "sector": f.sector, "country": f.country,
            "statement_currency": f.financial_currency, "units": "millions (margins as fractions)",
            "basis": "reported annual results (actuals) by fiscal year end; not forecasts",
            "fiscal_years": years, "adjustments": f.adjustments, "source": f.source, "as_of": _now()}


def run_dcf(symbol: str, growth_years_1_5: float | None = None, terminal_growth: float = 0.025,
            equity_risk_premium: float = ctry.MATURE_ERP) -> dict:
    """DCF with the same defaults as the Equity Research page; growth defaults to revenue CAGR (capped -5%..20%)."""
    f = registry.fundamentals(symbol)
    if f.sector == "Financial Services":
        return {"symbol": symbol, "error": "DCF not meaningful for banks/insurers; use P/B and P/E instead."}
    inc, bal, cf = f.statements["income"], f.statements["balance"], f.statements["cashflow"]
    fcf, rev = series(cf, "free_cash_flow"), series(inc, "revenue")
    cagr = (rev.iloc[-1] / rev.iloc[0]) ** (1 / (len(rev) - 1)) - 1 if len(rev) > 1 and rev.iloc[0] > 0 else 0.05
    g = float(np.clip(cagr, -0.05, 0.20)) if growth_years_1_5 is None else growth_years_1_5
    fin = f.financial_currency
    live = registry.risk_free_usd() if fin == "USD" else None
    rf, rf_src = live if live else (ctry.risk_free_default(fin), ctry.RF_SOURCE)
    tax = latest(inc, "tax_rate")
    tax = 0.21 if np.isnan(tax) else tax
    beta = f.beta if f.beta is not None and f.beta >= 0.3 else 1.0
    p2f = fx.latest_rate(f.price_currency, fin)[0]
    f2p = fx.latest_rate(fin, f.price_currency)[0]
    debt = latest(bal, "total_debt", 0.0)
    minority = latest(bal, "minority_interest", 0.0)
    i = dcf.DCFInputs(base_fcf=float(fcf.iloc[-1]), growth_high=g, terminal_growth=terminal_growth, risk_free=rf, beta=beta,
                      equity_risk_premium=equity_risk_premium, country_risk_premium=ctry.crp_for(f.country),
                      pre_tax_cost_of_debt=rf + 0.015, tax_rate=tax, equity_value_market=f.price * p2f * f.shares,
                      debt=debt, cash=latest(bal, "cash", 0.0), minority_interest=0.0 if np.isnan(minority) else minority,
                      shares=f.shares)
    r = dcf.run(i)
    vps = r.value_per_share * f2p
    return {"symbol": symbol, "price": _r(f.price), "price_currency": f.price_currency, "value_per_share": _r(vps, 2),
            "upside": _r(vps / f.price - 1), "wacc": _r(r.wacc), "cost_of_equity": _r(r.cost_of_equity),
            "assumptions": {"growth_years_1_5": _r(g), "terminal_growth": terminal_growth, "risk_free": _r(rf),
                            "risk_free_source": rf_src, "beta": _r(beta), "equity_risk_premium": equity_risk_premium,
                            "country_risk_premium": ctry.crp_for(f.country), "tax_rate": _r(tax)},
            "terminal_value_share": _r(r.terminal_share), "warnings": r.warnings,
            "implied_growth_at_market_price": _r(dcf.implied_growth(i, f.price * p2f)),
            "source": f"{f.source} statements; model gterm.analysis.dcf", "as_of": _now()}


def get_peer_multiples(symbol: str) -> dict:
    f = registry.fundamentals(symbol)
    target = registry.key_stats(symbol)
    peers = []
    for p in multiples.default_peers(symbol, f.sector, f.country):
        try:
            peers.append(registry.key_stats(p))
        except DataSourceError:
            pass
    imp = multiples.implied_prices(target, peers, f.price)
    return {"symbol": symbol, "price": _r(f.price), "currency": f.price_currency, "peers": [p.symbol for p in peers],
            "multiples": [{k: (_r(v) if not isinstance(v, str) else v) for k, v in row.items()} for row in imp.to_dict("records")],
            "source": target.source, "as_of": _now()}


def get_quality_score(symbol: str) -> dict:
    f = registry.fundamentals(symbol)
    q = quality.compute(f.statements, f.sector)
    return {"symbol": symbol, "overall_0_100": _r(q.overall, 1), "values": {k: _r(v) for k, v in q.values.items()},
            "scores": {k: _r(v, 1) for k, v in q.scores.items()}, "notes": q.notes, "source": f.source, "as_of": _now()}


def get_macro_indicator(indicator: str, country: str, source: str = "World Bank") -> dict:
    ms = macro.country_indicator(indicator, country, source)
    s = ms.series.tail(8)
    return {"indicator": indicator, "country": country, "units": ms.units, "values": {str(d.year): _r(v, 2) for d, v in s.items()},
            "includes_forecasts_from": f"{ms.forecast_from:%Y}" if ms.forecast_from is not None else None,
            "source": ms.source, "url": ms.url, "as_of": _now()}


def get_policy_rate(bank: str) -> dict:
    ms = macro.policy_rate(bank)
    d, v = ms.last
    return {"rate": bank, "level_pct": _r(v, 3), "date": f"{d:%Y-%m-%d}", "source": ms.source, "as_of": _now()}


def search_news(query: str, max_items: int = 8) -> dict:
    df = news.to_frame(news.google_news(query))
    items = [{"title": r.title, "publisher": r.publisher, "published": f"{r.published:%Y-%m-%d}" if r.published == r.published else None,
              "url": r.url} for r in df.head(max_items).itertuples()]
    return {"query": query, "articles": items, "source": "Google News", "as_of": _now()}


def _schema(props: dict, required: list) -> dict:
    return {"type": "object", "properties": props, "required": required}


SYM = {"type": "string", "description": "Yahoo-style ticker, e.g. AAPL, SAP.DE, SHEL.L, ^GSPC, EURUSD=X"}
TOOLS = {
    "get_quote": (get_quote, "Latest price and daily change for a ticker.", _schema({"symbol": SYM}, ["symbol"])),
    "get_price_performance": (get_price_performance, "Total return, volatility and max drawdown over a period.",
                              _schema({"symbol": SYM, "period": {"type": "string", "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y"]}}, ["symbol"])),
    "get_fundamentals": (get_fundamentals, "Normalised annual revenue, operating income, net income, margin, free cash flow, net debt, equity (millions).",
                         _schema({"symbol": SYM}, ["symbol"])),
    "run_dcf": (run_dcf, "Discounted cash flow valuation with the terminal's default assumptions; optionally override growth/terminal growth/ERP (fractions, e.g. 0.06).",
                _schema({"symbol": SYM, "growth_years_1_5": {"type": "number"}, "terminal_growth": {"type": "number"},
                         "equity_risk_premium": {"type": "number"}}, ["symbol"])),
    "get_peer_multiples": (get_peer_multiples, "P/E, EV/EBITDA, P/B, EV/Sales vs regional sector peers, with implied prices.",
                           _schema({"symbol": SYM}, ["symbol"])),
    "get_quality_score": (get_quality_score, "Quality score 0-100 from ROIC, margin stability, leverage, cash conversion.",
                          _schema({"symbol": SYM}, ["symbol"])),
    "get_macro_indicator": (get_macro_indicator, "Annual macro data for a country (World Bank actuals or IMF incl. forecasts).",
                            _schema({"indicator": {"type": "string", "enum": list(macro.INDICATORS)},
                                     "country": {"type": "string", "enum": list(macro.COUNTRIES)},
                                     "source": {"type": "string", "enum": ["World Bank", "IMF"]}}, ["indicator", "country"])),
    "get_policy_rate": (get_policy_rate, "Latest central bank policy rate.",
                        _schema({"bank": {"type": "string", "enum": list(macro.POLICY_RATES)}}, ["bank"])),
    "search_news": (search_news, "Recent news headlines for a company, country or topic.",
                    _schema({"query": {"type": "string"}}, ["query"])),
}


def definitions() -> list[dict]:
    return [{"name": n, "description": d, "parameters": p} for n, (_, d, p) in TOOLS.items()]


def execute(name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"Unknown tool {name}"}
    fn = TOOLS[name][0]
    try:
        return fn(**args)
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}
    except (DataSourceError, ValueError) as exc:
        return {"error": str(exc)}
