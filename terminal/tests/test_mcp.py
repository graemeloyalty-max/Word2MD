"""MCP client and server, tested against real stdio servers (subprocesses), no network."""
import json
import sys
from pathlib import Path

import pytest

from gterm.ai import agent, mcp_client, tools
from gterm.ai.providers import Provider, ToolCall, Turn

FIXTURE = Path(__file__).parent / "fixtures" / "mcp_demo_server.py"


@pytest.fixture
def demo_server():
    s = mcp_client.MCPServerConfig("Demo Server", "stdio", f"{sys.executable} {FIXTURE}")
    mcp_client.save_servers([s])
    yield s
    mcp_client.save_servers([])


def test_tool_names_and_schema_cleaning():
    assert mcp_client.tool_name("My Server!", "get.data") == "mcp__my_server__get_data"
    assert len(mcp_client.tool_name("x" * 50, "y" * 80)) <= 64
    raw = {"title": "T", "type": "object", "additionalProperties": False, "$schema": "x",
           "properties": {"a": {"title": "A", "type": "number"},
                          "t": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "title": "Title"}},
           "required": ["a"]}
    assert mcp_client.clean_schema(raw) == {"type": "object", "properties": {"a": {"type": "number"}, "t": {"type": "string"}},
                                            "required": ["a"]}
    assert mcp_client.clean_schema(None) == {"type": "object", "properties": {}}


def test_list_and_call_real_stdio_server(demo_server):
    defs, route, errors = mcp_client.definitions()
    assert not errors
    names = {d["name"] for d in defs}
    assert {"mcp__demo_server__add", "mcp__demo_server__greet", "mcp__demo_server__boom"} <= names
    greet = next(d for d in defs if d["name"] == "mcp__demo_server__greet")
    assert greet["parameters"]["properties"]["title"] == {"type": "string"}  # Optional collapsed for providers
    r = mcp_client.call("mcp__demo_server__add", {"a": 2, "b": 3}, route)
    assert r["result"] == {"result": 5.0} and not r["is_error"] and r["source"] == "MCP server 'Demo Server'"
    assert "via" not in r["source"]  # no inner source in this result
    bad = mcp_client.call("mcp__demo_server__boom", {}, route)
    assert bad["is_error"] and "error" in bad
    assert "not available" in mcp_client.call("mcp__nope__x", {}, route)["error"]


def test_unreachable_server_reports_error_not_crash():
    mcp_client.save_servers([mcp_client.MCPServerConfig("Broken", "stdio", "definitely-not-a-command-xyz")])
    defs, route, errors = mcp_client.definitions()
    assert defs == [] and errors and errors[0].startswith("Broken:")
    mcp_client.save_servers([mcp_client.MCPServerConfig("NoURL", "http", url="")])
    assert "no URL" in mcp_client.definitions()[2][0]
    mcp_client.save_servers([])


def test_disabled_servers_are_skipped(demo_server):
    demo_server.enabled = False
    mcp_client.save_servers([demo_server])
    assert mcp_client.definitions()[0] == []


def test_agent_uses_mcp_tool_end_to_end(demo_server):
    class Fake(Provider):
        def chat(self, model, system, messages, tools_):
            assert any(t["name"] == "mcp__demo_server__add" for t in tools_)
            if messages[-1]["role"] == "user":
                return Turn("", [ToolCall("1", "mcp__demo_server__add", {"a": 40, "b": 2})])
            return Turn(f"The answer is {messages[-1]['result']['result']['result']} [MCP server 'Demo Server'].")

    r = agent.run(Fake(), "m", "add 40 and 2", tool_defs=tools.definitions(include_mcp=True))
    assert r.answer.startswith("The answer is 42.0")


def test_terminal_runs_as_mcp_server(tmp_path):
    """Start `python -m gterm.mcp_server` exactly as Claude Code would, on offline demo data."""
    data = tmp_path / "srvdata"
    data.mkdir()
    (data / "settings.json").write_text(json.dumps({"data_source": "demo", "base_currency": "USD"}))
    s = mcp_client.MCPServerConfig("Terminal", "stdio", f"{sys.executable} -m gterm.mcp_server",
                                   env={"GTERM_DATA_DIR": str(data), "GTERM_ENV_FILE": str(tmp_path / ".env")})
    names = {t.name for t in mcp_client.list_tools(s, use_cache=False)}
    assert set(tools.TOOLS) <= names
    mcp_client.save_servers([s])
    out = mcp_client.call("mcp__terminal__get_quote", {"symbol": "SAP.DE"})
    assert not out["is_error"], out
    assert out["result"]["source"].startswith("demo") and out["result"]["currency"] == "EUR"
    assert out["source"] == "demo (synthetic) via MCP server 'Terminal'"  # the real data source stays visible
    mcp_client.save_servers([])


def test_proxy_settings_reach_stdio_servers(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ca.pem")
    params = mcp_client._target(mcp_client.MCPServerConfig("x", "stdio", "python -m something", env={"FOO": "1"}))
    assert params.env["HTTPS_PROXY"] == "http://proxy:8080" and params.env["SSL_CERT_FILE"] == "/etc/ca.pem" and params.env["FOO"] == "1"
