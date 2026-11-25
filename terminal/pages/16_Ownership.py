import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gterm import ui
from gterm.analysis import ownership as ow
from gterm.data import registry
from gterm.data.base import DataSourceError

ui.page_setup("Ownership & Relationships", "🕸️")
COLORS = {"company": "#111827", "institution": "#2563eb", "fund": "#7c3aed", "shared": "#d97706"}


def draw(nodes: pd.DataFrame, edges: list, height=560):
    pos = nodes.set_index("id")[["x", "y"]]
    fig = go.Figure()
    ex, ey = [], []
    for a, b in edges:
        if a in pos.index and b in pos.index:
            ex += [pos.loc[a, "x"], pos.loc[b, "x"], None]
            ey += [pos.loc[a, "y"], pos.loc[b, "y"], None]
    fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(color="#d1d5db", width=1), hoverinfo="skip", showlegend=False)
    for kind, g in nodes.groupby("kind"):
        if kind == "insider":
            color = ["#16a34a" if v > 0 else "#dc2626" if v < 0 else "#9ca3af" for v in g["value"]]
            name = "insider (green = net buyer, red = net seller)"
        else:
            color, name = COLORS.get(kind, "#6b7280"), kind
        fig.add_scatter(x=g["x"], y=g["y"], mode="markers+text", name=name, text=g["label"].str.slice(0, 32),
                        textposition="top center", textfont=dict(size=10),
                        marker=dict(size=g["size"].clip(8, 60), color=color, line=dict(color="white", width=1)),
                        hovertext=g["label"], hoverinfo="text")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      legend=dict(orientation="h"), plot_bgcolor="rgba(0,0,0,0)")
    return fig


t_one, t_many = st.tabs(["🏢 One company", "🔗 Common ownership"])

with t_one:
    c1, c2 = st.columns([2, 1])
    sym = c1.text_input("Ticker", st.query_params.get("symbol", "AAPL")).strip().upper()
    months = c2.selectbox("Insider window", [3, 6, 12, 24], index=2, format_func=lambda m: f"last {m} months")
    try:
        o = registry.ownership(sym)
    except DataSourceError as exc:
        st.error(str(exc))
        st.stop()
    since = pd.Timestamp.today() - pd.DateOffset(months=months)
    act = ow.insider_activity(o.insider_transactions, since)
    m = st.columns(4)
    m[0].metric("Held by institutions", f"{o.summary['institutions_pct']:.1%}" if "institutions_pct" in o.summary else "–")
    m[1].metric("Held by insiders", f"{o.summary['insiders_pct']:.2%}" if "insiders_pct" in o.summary else "–")
    net = act["net_value"].sum() if not act.empty else 0
    m[2].metric(f"Insider net buying ({months}m)", f"{net / 1e6:+,.1f} m", "net buyers" if net > 0 else "net sellers" if net < 0 else "flat",
                delta_color="normal" if net > 0 else "inverse" if net < 0 else "off")
    m[3].metric("Institutions listed", f"{len(o.institutions)}")
    rep = pd.to_datetime(o.institutions.get("Date Reported"), errors="coerce").max() if not o.institutions.empty else None
    st.caption(f"Source: **{o.source}** · retrieved {o.as_of:%Y-%m-%d %H:%M} UTC" +
               (f" · holdings as reported on {rep:%Y-%m-%d} (13F filings lag up to ~45 days)" if rep is not None and pd.notna(rep) else ""))
    nodes, edges = ow.company_network(sym, o.institutions, o.funds, act)
    st.plotly_chart(draw(nodes, edges), use_container_width=True)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Top institutional holders**")
        st.dataframe(o.institutions.style.format({"pctHeld": "{:.2%}", "pctChange": "{:+.2%}", "Shares": "{:,.0f}", "Value": "{:,.0f}"}),
                     use_container_width=True, hide_index=True)
        if not o.funds.empty:
            st.markdown("**Top fund holders**")
            st.dataframe(o.funds.style.format({"pctHeld": "{:.2%}", "pctChange": "{:+.2%}", "Shares": "{:,.0f}", "Value": "{:,.0f}"}),
                         use_container_width=True, hide_index=True)
    with g2:
        st.markdown(f"**Insider activity, last {months} months**")
        st.dataframe(act.style.format({"bought_shares": "{:,.0f}", "sold_shares": "{:,.0f}", "net_shares": "{:+,.0f}", "net_value": "{:+,.0f}"}),
                     use_container_width=True, hide_index=True)
        with st.expander("All insider transactions"):
            cols = [c for c in ["Start Date", "Insider", "Position", "Text", "Shares", "Value"] if c in o.insider_transactions]
            st.dataframe(o.insider_transactions[cols], use_container_width=True, hide_index=True)

with t_many:
    txt = st.text_input("Tickers (comma-separated)", "AAPL, MSFT, NVDA, JPM, XOM")
    syms = [s.strip().upper() for s in txt.split(",") if s.strip()]
    top = st.slider("Top holders per company", 5, 10, 10)
    holdings, missing = {}, []
    for s in syms:
        try:
            holdings[s] = registry.ownership(s).institutions
        except DataSourceError:
            missing.append(s)
    if missing:
        st.caption("No ownership data for: " + ", ".join(missing))
    if len(holdings) >= 2:
        nodes, edges, shared = ow.common_ownership(holdings, top)
        st.plotly_chart(draw(nodes, edges, 620), use_container_width=True)
        st.markdown("**Holders shared by two or more of these companies** (% of each company's shares held)")
        st.dataframe(shared.style.format({s: "{:.2%}" for s in holdings}, na_rep="–"), use_container_width=True, hide_index=True)
        st.caption(f"Source: **{registry.active_source_name()}** · top {top} institutional holders per company from the latest filings.")
    else:
        st.info("Enter at least two tickers that have ownership data (mostly US-listed companies).")
with st.expander("How to read these maps"):
    st.markdown(ow.__doc__)
