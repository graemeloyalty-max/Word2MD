"""Drag-and-drop workflow canvas: a small custom Streamlit component around Drawflow (MIT, bundled here).

The browser side (index.html) sends the graph back as Drawflow's JSON export whenever it changes.
It only reloads a graph from Python when `graph_id` changes, so edits survive Streamlit reruns.
"""
from pathlib import Path

import streamlit.components.v1 as components

_component = components.declare_component("gterm_flow_editor", path=str(Path(__file__).parent))


def flow_editor(graph: dict | None, graph_id: str, key: str = "flow"):
    """Render the canvas. Returns the current graph (Drawflow export) or None before the first change."""
    return _component(graph=graph or {}, graph_id=graph_id, key=key, default=None)
