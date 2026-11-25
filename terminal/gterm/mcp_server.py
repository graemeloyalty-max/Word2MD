"""Run the terminal as an MCP server, so other AI apps can use its research tools.

Example: let Claude Code (on your Claude Pro plan, no API key) use the terminal:

    cd terminal
    claude mcp add equity-terminal -- python -m gterm.mcp_server

Claude Desktop: add to claude_desktop_config.json
    {"mcpServers": {"equity-terminal": {"command": "python", "args": ["-m", "gterm.mcp_server"],
                                        "cwd": "/path/to/Word2MD/terminal"}}}

It exposes the same tools as the AI Research page (quotes, fundamentals, DCF, peers, quality,
macro, policy rates, news), each result carrying its source and date. It uses the data source
and keys configured in the terminal's Settings.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python gterm/mcp_server.py` too

from mcp.server.mcpserver import MCPServer  # noqa: E402

from gterm.ai import tools  # noqa: E402

INSTRUCTIONS = ("Research tools from a personal global equity terminal. Every result includes 'source' and 'as_of': "
                "cite them. Valuation outputs depend on editable assumptions; state them. Research, not investment advice.")


def build() -> MCPServer:
    app = MCPServer("global-equity-terminal", instructions=INSTRUCTIONS)
    for name, (fn, description, _) in tools.TOOLS.items():
        app.add_tool(fn, name=name, description=description)
    return app


if __name__ == "__main__":
    build().run("stdio")
