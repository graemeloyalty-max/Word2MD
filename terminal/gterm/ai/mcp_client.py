"""MCP (Model Context Protocol) client: let the AI agents use tools from outside MCP servers.

Plain English: an MCP server is a small program (local command or web URL) that offers tools in a
standard format. Register one here and its tools appear to the AI next to the terminal's own,
named `mcp__<server>__<tool>`. Results come back labelled with the server they came from.

Security: a stdio server is a program on your machine started with your permissions, so only add
servers you trust. Text returned by any server is treated as data, not instructions.
Servers are saved in data/mcp_servers.json. Built on the official `mcp` Python SDK (v2).
"""
from __future__ import annotations

import asyncio
import json
import re
import shlex
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from ..core import config

LIST_TTL_S = 600
CALL_TIMEOUT_S = 60
MAX_RESULT_CHARS = 6000
_tool_cache: dict[str, tuple[float, list]] = {}


class MCPError(Exception):
    pass


@dataclass
class MCPServerConfig:
    name: str
    transport: str = "stdio"  # "stdio" (local command) or "http" (Streamable HTTP URL)
    command: str = ""  # stdio: full command line, e.g. "python -m gterm.mcp_server"
    url: str = ""  # http: e.g. "http://localhost:8000/mcp"
    env: dict = field(default_factory=dict)
    enabled: bool = True


def _path():
    return config.DATA_DIR / "mcp_servers.json"


def load_servers() -> list[MCPServerConfig]:
    if not _path().exists():
        return []
    try:
        return [MCPServerConfig(**{k: v for k, v in d.items() if k in MCPServerConfig.__dataclass_fields__})
                for d in json.loads(_path().read_text())]
    except (json.JSONDecodeError, TypeError):
        return []


def save_servers(servers: list[MCPServerConfig]) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _path().write_text(json.dumps([asdict(s) for s in servers], indent=2))
    _tool_cache.clear()


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()[:20] or "server"


def tool_name(server: str, tool: str) -> str:
    """Provider-safe name: letters, digits, _ and -, at most 64 characters."""
    return f"mcp__{slug(server)}__{re.sub(r'[^A-Za-z0-9_-]+', '_', tool)}"[:64]


KEEP = {"type", "properties", "required", "items", "enum", "description", "format", "nullable", "minimum", "maximum", "anyOf"}


def clean_schema(schema) -> dict:
    """Keep the JSON-schema subset every provider accepts (Gemini rejects e.g. 'title', 'additionalProperties', '$schema')."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    out = {}
    for k, v in schema.items():
        if k not in KEEP:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: clean_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = clean_schema(v)
        elif k == "anyOf" and isinstance(v, list):
            # Optional[X] arrives as anyOf [X, null]: collapse to X so simple providers accept it
            non_null = [clean_schema(x) for x in v if not (isinstance(x, dict) and x.get("type") == "null")]
            if len(non_null) == 1:
                out.update(non_null[0])
            else:
                out[k] = non_null
        else:
            out[k] = v
    if out.get("type") == "object" and "properties" not in out:
        out["properties"] = {}
    return out


def _target(s: MCPServerConfig):
    from mcp import StdioServerParameters

    if s.transport == "http":
        if not s.url:
            raise MCPError(f"{s.name}: no URL set")
        return s.url
    parts = shlex.split(s.command)
    if not parts:
        raise MCPError(f"{s.name}: no command set")
    return StdioServerParameters(command=parts[0], args=parts[1:], env={**passthrough_env(), **s.env}, cwd=str(config.ROOT))


# The MCP SDK starts stdio servers with a minimal environment (PATH, HOME, ...). Without the proxy and certificate
# settings a server behind a proxy can't reach the internet, and the terminal's "auto" data source would silently
# fall back to demo data. Pass those through.
PASSTHROUGH = ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "no_proxy", "all_proxy",
               "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
               "GTERM_DATA_DIR", "GTERM_ENV_FILE")


def passthrough_env() -> dict:
    import os

    return {k: os.environ[k] for k in PASSTHROUGH if os.environ.get(k)}


async def _list(s):
    from mcp import Client

    async with Client(_target(s), read_timeout_seconds=CALL_TIMEOUT_S) as c:
        return (await c.list_tools()).tools


async def _call(s, tool, args):
    from mcp import Client

    async with Client(_target(s), read_timeout_seconds=CALL_TIMEOUT_S) as c:
        return await c.call_tool(tool, args)


def _run(coro):
    try:
        return asyncio.run(asyncio.wait_for(coro, CALL_TIMEOUT_S + 10))
    except MCPError:
        raise
    except Exception as exc:  # connection refused, bad command, timeout, protocol errors…
        msg = str(exc) or type(exc).__name__
        if hasattr(exc, "exceptions"):  # ExceptionGroup from the async task group: show the first real cause
            msg = "; ".join(str(e) or type(e).__name__ for e in exc.exceptions)[:300]
        raise MCPError(msg) from exc


def list_tools(s: MCPServerConfig, use_cache: bool = True) -> list:
    key = json.dumps(asdict(s), sort_keys=True)
    hit = _tool_cache.get(key)
    if use_cache and hit and time.time() - hit[0] < LIST_TTL_S:
        return hit[1]
    tools = _run(_list(s))
    _tool_cache[key] = (time.time(), tools)
    return tools


def definitions() -> tuple[list[dict], dict, list[str]]:
    """Tool definitions for all enabled servers, a name -> (server, tool) map, and any connection errors."""
    defs, route, errors = [], {}, []
    for s in load_servers():
        if not s.enabled:
            continue
        try:
            for t in list_tools(s):
                n = tool_name(s.name, t.name)
                desc = f"[MCP server '{s.name}'] {t.description or t.name}"[:1000]
                defs.append({"name": n, "description": desc, "parameters": clean_schema(t.input_schema)})
                route[n] = (s, t.name)
        except MCPError as exc:
            errors.append(f"{s.name}: {exc}")
    return defs, route, errors


def format_result(server: str, tool: str, res) -> dict:
    texts = [getattr(c, "text", None) or f"[{getattr(c, 'type', 'content')} omitted]" for c in (res.content or [])]
    out = {"server": server, "tool": tool, "is_error": bool(res.is_error)}
    if res.structured_content is not None:
        out["result"] = res.structured_content
    else:
        joined = "\n".join(texts)
        try:  # many servers return JSON as text: hand the AI real structure when it parses
            out["result"] = json.loads(joined) if len(texts) == 1 else joined
        except (json.JSONDecodeError, TypeError):
            out["result"] = joined
    s = json.dumps(out["result"], default=str)
    if len(s) > MAX_RESULT_CHARS:
        out["result"] = s[:MAX_RESULT_CHARS] + "… (truncated)"
    if res.is_error:
        out["error"] = "\n".join(texts)[:500] or "tool reported an error"
    inner = out["result"].get("source") if isinstance(out["result"], dict) else None
    # Never hide the underlying data source (e.g. "demo (synthetic)") behind the server's name.
    out["source"] = f"{inner} via MCP server '{server}'" if inner else f"MCP server '{server}'"
    out["as_of"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return out


def call(name: str, args: dict, route: dict | None = None) -> dict:
    route = route if route is not None else definitions()[1]
    if name not in route:
        return {"error": f"MCP tool {name} is not available (server disabled or unreachable)."}
    s, tool = route[name]
    try:
        return format_result(s.name, tool, _run(_call(s, tool, args)))
    except MCPError as exc:
        return {"error": f"MCP server '{s.name}' failed: {exc}", "source": f"MCP server '{s.name}'"}
