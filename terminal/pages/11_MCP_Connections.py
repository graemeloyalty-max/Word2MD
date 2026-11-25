import sys

import pandas as pd
import streamlit as st

from gterm import ui
from gterm.ai import mcp_client
from gterm.core import config

ui.page_setup("MCP Connections", "🔌")

st.markdown("Connect **MCP servers** so the AI agents can use outside tools and data. Their tools appear in the AI Research "
            "chat and can be given to research-team specialists in the Agents tab, named `mcp__<server>__<tool>`.")
st.warning("A **local command** server is a program on your machine that runs with your permissions: only add servers you trust. "
           "Whatever a server returns is treated as data, never as instructions.", icon="🔐")

servers = mcp_client.load_servers()

st.subheader("Your servers")
if not servers:
    st.info("No servers yet. Add one below. To try it out, add this terminal itself (the example under 'Add a server').")
for i, s in enumerate(servers):
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        c1.markdown(f"**{s.name}** · {'local command' if s.transport == 'stdio' else 'URL'}  \n"
                    f"`{s.command if s.transport == 'stdio' else s.url}`")
        enabled = c2.toggle("Enabled", s.enabled, key=f"en{i}")
        if enabled != s.enabled:
            s.enabled = enabled
            mcp_client.save_servers(servers)
            st.rerun()
        if c3.button("Test", key=f"t{i}"):
            try:
                with st.spinner("Connecting…"):
                    tl = mcp_client.list_tools(s, use_cache=False)
                st.success(f"Connected: {len(tl)} tool(s).")
                st.dataframe(pd.DataFrame([{"Tool (as the AI sees it)": mcp_client.tool_name(s.name, t.name),
                                            "Description": (t.description or "")[:140]} for t in tl]),
                             hide_index=True, use_container_width=True)
            except mcp_client.MCPError as exc:
                st.error(f"Could not connect: {exc}")
        if c4.button("Remove", key=f"r{i}"):
            servers.pop(i)
            mcp_client.save_servers(servers)
            st.rerun()

st.subheader("Add a server")
with st.form("add", clear_on_submit=True):
    a, b = st.columns([2, 1])
    name = a.text_input("Name", placeholder="e.g. files, github, my-data")
    transport = b.radio("Type", ["Local command (stdio)", "URL (Streamable HTTP)"])
    target = st.text_input("Command or URL", placeholder="e.g. npx -y @modelcontextprotocol/server-filesystem /path/to/folder   or   http://localhost:8000/mcp")
    env_txt = st.text_area("Environment variables for a local command (optional, one KEY=value per line)", height=68,
                           help="Stored in data/mcp_servers.json on this machine. Prefer servers that read keys from their own config.")
    if st.form_submit_button("Add server", type="primary"):
        if not name.strip() or not target.strip():
            st.error("Name and command/URL are required.")
        elif any(x.name.lower() == name.strip().lower() for x in servers):
            st.error("A server with that name already exists.")
        else:
            env = dict(line.split("=", 1) for line in env_txt.splitlines() if "=" in line)
            stdio = transport.startswith("Local")
            servers.append(mcp_client.MCPServerConfig(name.strip(), "stdio" if stdio else "http",
                                                      command=target.strip() if stdio else "", url="" if stdio else target.strip(),
                                                      env={k.strip(): v.strip() for k, v in env.items()}))
            mcp_client.save_servers(servers)
            st.success(f"Added {name}. Press Test to check it.")
            st.rerun()

with st.expander("Example: connect this terminal to itself (a quick way to test MCP)"):
    st.markdown("Name `terminal`, type *Local command*, command:")
    st.code(f"{sys.executable} -m gterm.mcp_server", language="bash")
    st.caption("Its tools then appear twice (built-in and via MCP): fine for testing, then remove it.")

st.subheader("Use this terminal from Claude (no API key)")
st.markdown("The terminal can itself act as an MCP server, so **Claude Code or Claude Desktop on your Claude Pro plan** can call its "
            "research tools (quotes, fundamentals, DCF, peers, quality, macro, news). Usage counts against your Pro allowance, "
            "not an API bill.")
root = config.ROOT
st.markdown("**Claude Code**: run once inside the `terminal` folder:")
st.code("claude mcp add equity-terminal -- python -m gterm.mcp_server", language="bash")
st.markdown("**Claude Desktop**: Settings → Developer → Edit config, then add:")
st.code('{\n  "mcpServers": {\n    "equity-terminal": {\n      "command": "python",\n      "args": ["-m", "gterm.mcp_server"],\n'
        f'      "cwd": "{root}"\n    }}\n  }}\n}}'.replace("}}", "}"), language="json")
st.caption("Then ask Claude e.g. “Use equity-terminal to value SAP.DE and compare it with peers.” It uses the data source and keys from "
           "this terminal's Settings.")
