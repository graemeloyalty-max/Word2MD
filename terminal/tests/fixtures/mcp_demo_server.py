"""Tiny MCP server used by the tests (stdio)."""
from typing import Optional

from mcp.server.mcpserver import MCPServer

app = MCPServer("demo")


@app.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@app.tool()
def greet(name: str, title: Optional[str] = None) -> str:
    """Greet someone."""
    return f"Hello {title + ' ' if title else ''}{name}"


@app.tool()
def boom() -> str:
    """Always fails."""
    raise ValueError("kaboom")


if __name__ == "__main__":
    app.run("stdio")
