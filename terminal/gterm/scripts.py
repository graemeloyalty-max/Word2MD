"""Code editor back end: run your own Python against the terminal's data.

Plain English: your script runs as a separate Python process (so a bug or an endless loop can't
freeze the app) with a time limit. It starts with the terminal's tools already imported:

    prices("SAP.DE", "1y")      -> DataFrame of Open/High/Low/Close/Volume (the active data source)
    quote("AAPL")               -> latest Quote (price, currency, source, ...)
    fundamentals("MSFT")        -> normalised statements: .statements["income"|"balance"|"cashflow"]
    to_ccy(series, "GBp", "ZAR") -> convert a price series at daily FX
    registry, fx, indicators, dcf, multiples, quality, pa (portfolio), backtest, strategies, factors, metrics
    pd, np, go (plotly.graph_objects), px (plotly.express)

    show(obj, "title")          -> display a DataFrame/Series (table), a plotly figure (chart), or any value

`print()` output appears below the editor. Scripts run on your machine with your permissions:
it's your code, but don't paste in code you don't understand.
Saved scripts live in data/scripts/; examples ship in gterm/script_examples/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .core import config

EXAMPLES_DIR = Path(__file__).parent / "script_examples"

PRELUDE = r'''
import json as _json, os as _os, sys as _sys
_sys.path.insert(0, {root!r})
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gterm.data import registry
from gterm.core import fx, indicators
from gterm.analysis import dcf, multiples, quality, portfolio as pa
from gterm.quant import backtest, strategies, factors, metrics

_OUT, _N = _os.environ["GTERM_SCRIPT_OUT"], [0]


def prices(symbol, period="1y", interval="1d"):
    return registry.prices(symbol, period, interval).frame


def quote(symbol):
    return registry.quote(symbol)


def fundamentals(symbol):
    return registry.fundamentals(symbol)


def to_ccy(series, from_ccy, to_ccy_):
    return fx.convert_series(series, from_ccy, to_ccy_)[0]


def show(obj, title=None):
    """Display a table (DataFrame/Series), a chart (plotly figure) or any value on the page."""
    _N[0] += 1
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        item = {{"kind": "table", "data": obj.head(5000).to_json(orient="split", date_format="iso", default_handler=str)}}
    elif hasattr(obj, "to_plotly_json"):
        item = {{"kind": "chart", "data": obj.to_json()}}
    else:
        try:
            item = {{"kind": "json", "data": _json.dumps(obj, default=str)}}
        except Exception:
            item = {{"kind": "text", "data": repr(obj)}}
    item["title"] = title
    with open(_os.path.join(_OUT, f"{{_N[0]:04d}}.json"), "w") as _f:
        _json.dump(item, _f)


_src = open(_os.environ["GTERM_SCRIPT_FILE"]).read()
exec(compile(_src, "your_script.py", "exec"), {{**globals(), "__name__": "__main__"}})
'''


@dataclass
class ScriptResult:
    stdout: str
    stderr: str
    returncode: int | None
    seconds: float
    timed_out: bool = False
    outputs: list = field(default_factory=list)  # [{"kind": table|chart|json|text, "data": str, "title": str|None}]

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def run(code: str, timeout: int = 60) -> ScriptResult:
    with tempfile.TemporaryDirectory(prefix="gterm_script_") as tmp:
        tmp = Path(tmp)
        out = tmp / "out"
        out.mkdir()
        (tmp / "user.py").write_text(code)
        (tmp / "runner.py").write_text(PRELUDE.format(root=str(config.ROOT)))
        env = {**os.environ, "GTERM_SCRIPT_OUT": str(out), "GTERM_SCRIPT_FILE": str(tmp / "user.py"),
               "GTERM_DATA_DIR": str(config.DATA_DIR), "GTERM_ENV_FILE": str(config.ENV_FILE), "MPLBACKEND": "Agg"}
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, str(tmp / "runner.py")], cwd=str(config.ROOT), env=env,
                               capture_output=True, text=True, timeout=timeout)
            res = ScriptResult(p.stdout[-50_000:], _clean_traceback(p.stderr)[-20_000:], p.returncode, time.time() - t0)
        except subprocess.TimeoutExpired as exc:
            res = ScriptResult((exc.stdout or b"").decode(errors="replace")[-50_000:] if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                               f"Stopped: the script ran longer than {timeout} seconds.", None, time.time() - t0, timed_out=True)
        for f in sorted(out.glob("*.json")):
            try:
                res.outputs.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                pass
        return res


def _clean_traceback(err: str) -> str:
    """Drop the runner's own frames so the traceback starts at the user's code."""
    lines = err.splitlines()
    keep, skipping = [], False
    for ln in lines:
        if 'File "' in ln and "runner.py" in ln:
            skipping = True
            continue
        if skipping and ln.startswith("    "):
            continue
        skipping = False
        keep.append(ln)
    return "\n".join(keep)


# ---------------------------------------------------------------- script library

def _dir() -> Path:
    d = config.DATA_DIR / "scripts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_name(name: str) -> str:
    import re

    n = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")[:60] or "script"
    return n


def list_scripts() -> dict[str, Path]:
    """name -> path; examples are prefixed 'Example: '."""
    out = {f"Example: {p.stem.replace('_', ' ')}": p for p in sorted(EXAMPLES_DIR.glob("*.py"))}
    out.update({p.stem: p for p in sorted(_dir().glob("*.py"))})
    return out


def save_script(name: str, code: str) -> Path:
    p = _dir() / f"{safe_name(name)}.py"
    p.write_text(code)
    return p


def delete_script(path: Path) -> None:
    if Path(path).parent == _dir():  # never delete the bundled examples
        Path(path).unlink(missing_ok=True)


def table(output: dict):
    """DataFrame from a 'table' output (pandas 3 needs a file-like object, not a JSON string)."""
    import io

    import pandas as pd

    return pd.read_json(io.StringIO(output["data"]), orient="split")
