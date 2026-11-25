"""Visual workflows: run a graph of blocks drawn on the canvas.

Plain English: each block does one job (fetch tickers' quotes, run a DCF, filter rows, ask an AI
to write, save a report). Arrows carry each block's output into the next. The runner:
1. works out an order where every block runs after the blocks feeding it (cycles are refused);
2. runs each block with its inputs, using the same functions as the rest of the terminal;
3. records where every figure came from, so a saved report gets a Sources appendix;
4. if a block fails, blocks downstream of it are skipped and the error is shown on that block.

Data between blocks: a list of tickers, a table (rows with a "symbol" column where relevant), or text.
Workflows are saved in data/workflows/ as the canvas's JSON.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import reports
from .ai import agent, tools
from .ai.providers import PROVIDERS, ProviderError
from .core import config


class WorkflowError(Exception):
    pass


@dataclass
class NodeResult:
    id: str
    name: str
    output: object = None  # list[str] tickers | pd.DataFrame | str
    error: str | None = None
    skipped: bool = False
    note: str = ""


@dataclass
class RunResult:
    order: list
    results: dict  # id -> NodeResult
    provenance: list = field(default_factory=list)  # steps for the report's Sources appendix
    report_paths: list = field(default_factory=list)


# ---------------------------------------------------------------- graph handling

def parse(graph: dict) -> dict:
    """Drawflow export -> {id: {"name", "data", "inputs": {"input_1": [src ids], ...}}}"""
    try:
        data = graph["drawflow"]["Home"]["data"]
    except (KeyError, TypeError) as exc:
        raise WorkflowError("Empty or invalid workflow.") from exc
    nodes = {}
    for nid, n in data.items():
        ins = {k: [str(c["node"]) for c in v.get("connections", [])] for k, v in (n.get("inputs") or {}).items()}
        nodes[str(nid)] = {"name": n["name"], "data": n.get("data") or {}, "inputs": ins}
    return nodes


def topo_order(nodes: dict) -> list:
    deps = {nid: {s for srcs in n["inputs"].values() for s in srcs if s in nodes} for nid, n in nodes.items()}
    order, ready = [], sorted([n for n, d in deps.items() if not d], key=int)
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for other, d in deps.items():
            if nid in d:
                d.discard(nid)
                if not d and other not in order and other not in ready:
                    ready.append(other)
        ready.sort(key=int)
    if len(order) != len(nodes):
        raise WorkflowError("The workflow has a loop (a block feeds back into itself). Remove one arrow.")
    return order


# ---------------------------------------------------------------- helpers

def tickers_from(value) -> list[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, pd.DataFrame) and "symbol" in value.columns:
        return [str(s) for s in value["symbol"].dropna().unique()]
    raise WorkflowError("This block needs tickers (from a Tickers block or a table with a 'symbol' column).")


def md_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    """DataFrame -> Markdown table (no extra dependency). Floats get 4 significant digits."""
    def fmt(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v).replace("|", "/").replace("\n", " ")
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.head(max_rows).itertuples(index=False)]
    if len(df) > max_rows:
        lines.append(f"\n… {len(df) - max_rows} more rows")
    return "\n".join(lines)


def as_text(value) -> str:
    if isinstance(value, pd.DataFrame):
        return md_table(value)
    if isinstance(value, list):
        return ", ".join(value)
    return str(value)


class _Ctx:
    def __init__(self):
        self.provenance = []

    def tool(self, label, name, **args):
        out = tools.execute(name, args)
        self.provenance.append({"agent": label, "tool": name, "args": args, "result": out})
        return out


def _per_ticker(ctx, label, tickers, tool, row_fn, **extra):
    rows = []
    for t in tickers:
        out = ctx.tool(label, tool, symbol=t, **extra)
        if "error" in out:
            rows.append({"symbol": t, "error": out["error"][:120]})
        else:
            rows.append({"symbol": t, **row_fn(out), "source": out.get("source"), "as_of": out.get("as_of")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- blocks

def b_tickers(ctx, label, data, inputs):
    syms = [s.strip().upper() for s in re.split(r"[,\s]+", data.get("symbols", "")) if s.strip()]
    if not syms:
        raise WorkflowError("Enter at least one ticker.")
    return syms


def b_quotes(ctx, label, data, inputs):
    return _per_ticker(ctx, label, tickers_from(inputs[0]), "get_quote",
                       lambda o: {"name": o["name"], "price": o["price"], "currency": o["currency"], "change_pct": o["change_pct"]})


def b_performance(ctx, label, data, inputs):
    return _per_ticker(ctx, label, tickers_from(inputs[0]), "get_price_performance",
                       lambda o: {"total_return": o["total_return"], "volatility": o["annualised_volatility"],
                                  "max_drawdown": o["max_drawdown"], "period": o["period"]}, period=data.get("period", "1y"))


def b_fundamentals(ctx, label, data, inputs):
    def row(o):
        years = o.get("fiscal_years", {})
        if not years:
            return {}
        y = max(years)
        return {"fiscal_year": y, "currency": o["statement_currency"], **{f"{k} (m)" if k not in ("operating_margin",) else k: v
                                                                          for k, v in years[y].items()}}
    return _per_ticker(ctx, label, tickers_from(inputs[0]), "get_fundamentals", row)


def b_dcf(ctx, label, data, inputs):
    return _per_ticker(ctx, label, tickers_from(inputs[0]), "run_dcf",
                       lambda o: {"price": o["price"], "currency": o["price_currency"], "dcf_value": o["value_per_share"],
                                  "upside": o["upside"], "wacc": o["wacc"], "terminal_growth": o["assumptions"]["terminal_growth"],
                                  "warning": "; ".join(o.get("warnings", []))[:160]})


def b_quality(ctx, label, data, inputs):
    return _per_ticker(ctx, label, tickers_from(inputs[0]), "get_quality_score",
                       lambda o: {"quality": o["overall_0_100"], **{f"{k}_score": v for k, v in o["scores"].items()}})


def b_peers(ctx, label, data, inputs):
    def row(o):
        imp = [m["Implied price"] for m in o["multiples"] if m.get("Implied price") is not None]
        med = float(pd.Series(imp).median()) if imp else None
        return {"price": o["price"], "currency": o["currency"], "peer_implied_price": med,
                "peer_upside": (med / o["price"] - 1) if med and o["price"] else None, "peers": ", ".join(o["peers"])}
    return _per_ticker(ctx, label, tickers_from(inputs[0]), "get_peer_multiples", row)


def b_macro(ctx, label, data, inputs):
    rows = []
    for c in [x.strip() for x in data.get("countries", "").split(",") if x.strip()]:
        out = ctx.tool(label, "get_macro_indicator", indicator=data.get("indicator", "Real GDP growth"), country=c,
                       source=data.get("source", "World Bank"))
        if "error" in out:
            rows.append({"country": c, "error": out["error"][:120]})
        else:
            rows.append({"country": c, **out["values"], "source": out["source"]})
    return pd.DataFrame(rows)


def b_news(ctx, label, data, inputs):
    queries = [data["query"]] if data.get("query", "").strip() else tickers_from(inputs[0]) if inputs and inputs[0] is not None else []
    if not queries:
        raise WorkflowError("Enter a query or connect tickers.")
    rows = []
    for q in queries:
        out = ctx.tool(label, "search_news", query=q, max_items=5)
        for a in out.get("articles", []):
            rows.append({"query": q, **a, "source": out.get("source")})
        if "error" in out:
            rows.append({"query": q, "error": out["error"][:120]})
    return pd.DataFrame(rows)


def b_filter(ctx, label, data, inputs):
    df = inputs[0]
    if not isinstance(df, pd.DataFrame):
        raise WorkflowError("Filter needs a table.")
    cond = data.get("condition", "").strip()
    if not cond:
        return df
    try:
        return df.query(cond, engine="python")
    except Exception as exc:
        raise WorkflowError(f"Could not apply '{cond}': {exc}. Use column names like: upside > 0.1 and quality >= 70") from exc


def b_sort(ctx, label, data, inputs):
    df = inputs[0]
    col = data.get("column", "")
    if not isinstance(df, pd.DataFrame) or col not in df.columns:
        raise WorkflowError(f"Sort: column '{col}' not found. Available: {', '.join(df.columns) if isinstance(df, pd.DataFrame) else '-'}")
    return df.sort_values(col, ascending=data.get("order") == "ascending")


def b_merge(ctx, label, data, inputs):
    a, b = (inputs + [None, None])[:2]
    if not all(isinstance(x, pd.DataFrame) and "symbol" in x.columns for x in (a, b)):
        raise WorkflowError("Merge needs two tables with a 'symbol' column (connect both inputs).")
    return a.merge(b, on="symbol", how="outer", suffixes=("", "_2"))


def b_ai(ctx, label, data, inputs):
    pname = data.get("provider", "Google AI Studio (Gemini)")
    if pname not in PROVIDERS:
        raise WorkflowError(f"Unknown provider {pname}")
    prov = PROVIDERS[pname]()
    material = "\n\n".join(as_text(i) for i in inputs if i is not None)
    prompt = (f"{data.get('instructions', 'Summarise the data.')}\n\nData (with its source and as_of columns where available):\n"
              f"{material[:15000]}")
    try:
        turn = prov.chat(data.get("model") or prov.default_model([]), agent.system_prompt(), [{"role": "user", "text": prompt}], [])
    except ProviderError as exc:
        raise WorkflowError(str(exc)) from exc
    return turn.text


def b_report(ctx, label, data, inputs):
    body = "\n\n".join(as_text(i) for i in inputs if i is not None)
    title = data.get("title") or "Workflow report"
    path = reports.save({"question": f"Workflow: {title}", "title": title, "note": body, "steps": list(ctx.provenance),
                         "usage": {}, "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")})
    ctx.report_paths.append(path)
    return f"Saved report '{title}' (see AI Research → Reports)."


BLOCKS = {"tickers": b_tickers, "quotes": b_quotes, "performance": b_performance, "fundamentals": b_fundamentals,
          "macro": b_macro, "news": b_news, "dcf": b_dcf, "quality": b_quality, "peers": b_peers, "filter": b_filter,
          "sort": b_sort, "merge": b_merge, "ai": b_ai, "report": b_report}
NEEDS_INPUT = {"quotes", "performance", "fundamentals", "dcf", "quality", "peers", "filter", "sort", "merge", "ai", "report"}


def run(graph: dict, on_node=None) -> RunResult:
    nodes = parse(graph)
    if not nodes:
        raise WorkflowError("The canvas is empty: add some blocks.")
    order = topo_order(nodes)
    ctx = _Ctx()
    ctx.report_paths = []
    results = {}
    for nid in order:
        n = nodes[nid]
        label = f"{n['name']} #{nid}"
        srcs = [s for k in sorted(n["inputs"]) for s in n["inputs"][k]]
        r = NodeResult(nid, n["name"])
        failed_up = [s for s in srcs if results.get(s) and (results[s].error or results[s].skipped)]
        if n["name"] not in BLOCKS:
            r.error = f"Unknown block type {n['name']}"
        elif failed_up:
            r.skipped, r.note = True, f"skipped because block #{', #'.join(failed_up)} failed"
        elif n["name"] in NEEDS_INPUT and not srcs:
            r.error = "Nothing is connected to this block's input."
        else:
            try:
                r.output = BLOCKS[n["name"]](ctx, label, n["data"], [results[s].output for s in srcs])
            except WorkflowError as exc:
                r.error = str(exc)
            except Exception as exc:  # keep the run going; show the problem on the block
                r.error = f"{type(exc).__name__}: {exc}"
        results[nid] = r
        if on_node:
            on_node(r)
    return RunResult(order, results, ctx.provenance, ctx.report_paths)


# ---------------------------------------------------------------- library & examples

def _dir() -> Path:
    d = config.DATA_DIR / "workflows"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_graph(nodes: list[dict], links: list[tuple]) -> dict:
    """Small helper to write example graphs: nodes [{name, data, x, y}], links [(from_index, to_index, to_input)]."""
    data = {}
    for i, n in enumerate(nodes, 1):
        data[str(i)] = {"id": i, "name": n["name"], "data": n.get("data", {}), "class": n["name"], "html": "", "typenode": False,
                        "inputs": {}, "outputs": {}, "pos_x": n["x"], "pos_y": n["y"]}
    for a, b, inp in links:
        a, b = str(a), str(b)
        data[a]["outputs"].setdefault("output_1", {"connections": []})["connections"].append({"node": b, "output": inp})
        data[b]["inputs"].setdefault(inp, {"connections": []})["connections"].append({"node": a, "input": "output_1"})
    for n in data.values():  # every declared port needs an entry, even if unconnected
        ins = {"quotes": 1, "performance": 1, "fundamentals": 1, "dcf": 1, "quality": 1, "peers": 1, "filter": 1, "sort": 1,
               "merge": 2, "ai": 1, "report": 1, "news": 1}.get(n["name"], 0)
        outs = 0 if n["name"] == "report" else 1
        for k in range(1, ins + 1):
            n["inputs"].setdefault(f"input_{k}", {"connections": []})
        for k in range(1, outs + 1):
            n["outputs"].setdefault(f"output_{k}", {"connections": []})
    return {"drawflow": {"Home": {"data": data}}}


EXAMPLES = {
    "Example: DCF screen → AI summary → report": build_graph(
        [{"name": "tickers", "data": {"symbols": "AAPL, MSFT, SAP.DE, ASML.AS, SHEL.L"}, "x": 10, "y": 60},
         {"name": "dcf", "x": 255, "y": 10},
         {"name": "quality", "x": 255, "y": 130},
         {"name": "merge", "x": 500, "y": 60},
         {"name": "sort", "data": {"column": "upside", "order": "descending"}, "x": 10, "y": 290},
         {"name": "ai", "data": {"provider": "Google AI Studio (Gemini)", "model": "gemini-3.5-flash",
                                 "instructions": "Rank these companies on DCF upside and quality. 5 bullets, cite sources, flag any warnings."},
          "x": 255, "y": 270},
         {"name": "report", "data": {"title": "DCF and quality screen"}, "x": 500, "y": 300}],
        [(1, 2, "input_1"), (1, 3, "input_1"), (2, 4, "input_1"), (3, 4, "input_2"), (4, 5, "input_1"), (5, 6, "input_1"),
         (6, 7, "input_1")]),
    "Example: macro dashboard (no AI)": build_graph(
        [{"name": "macro", "data": {"indicator": "Inflation (CPI)", "countries": "United States, Euro area, United Kingdom, South Africa",
                                    "source": "World Bank"}, "x": 40, "y": 60},
         {"name": "report", "data": {"title": "Inflation snapshot"}, "x": 300, "y": 60}],
        [(1, 2, "input_1")]),
}


def list_workflows() -> dict:
    out = dict(EXAMPLES)
    for p in sorted(_dir().glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
    return out


def save_workflow(name: str, graph: dict) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")[:60] or "workflow"
    p = _dir() / f"{safe}.json"
    p.write_text(json.dumps(graph, indent=1))
    return p


def delete_workflow(name: str) -> None:
    (_dir() / f"{name}.json").unlink(missing_ok=True)
