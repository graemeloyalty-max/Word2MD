import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gterm import ui
from gterm.analysis import options as op
from gterm.data import registry
from gterm.data.base import DataSourceError

ui.page_setup("Volatility Surface", "🌋")

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
sym = c1.text_input("Underlying", st.query_params.get("symbol", "SPY"),
                    help="US-listed stocks/ETFs via Yahoo (e.g. SPY, AAPL, NVDA). BTC or ETH via Deribit.").strip().upper()
crypto = sym in ("BTC", "ETH")
live_rf = registry.risk_free_usd_short()
r = c2.number_input("Risk-free rate (%)", 0.0, 20.0, round((live_rf[0] if live_rf else 0.042) * 100, 2), 0.05) / 100
q = c3.number_input("Dividend yield (%)", 0.0, 15.0, 0.0, 0.1, disabled=crypto,
                    help="Set the underlying's dividend yield (SPY ≈ 1.2%). Ignoring it raises put IVs slightly.") / 100
n_exp = c4.slider("Expiries", 4, 20, 12, disabled=crypto)
st.caption(f"Risk-free default: {live_rf[1] if live_rf else 'placeholder 4.2%'}")

try:
    with st.spinner("Loading option chain…"):
        chain, spot, ccy = registry.options_chain(sym, n_exp)
except DataSourceError as exc:
    st.error(str(exc))
    st.stop()

pts = op.surface_points(chain, spot, 0.0 if crypto else r, 0.0 if crypto else q)
if pts.empty:
    st.error("No usable option prices after filtering.")
    st.stop()
src = "Deribit" if crypto else registry.active_source_name()
m = st.columns(4)
m[0].metric(f"Spot ({ccy})", f"{spot:,.2f}")
ts = op.term_structure(pts, spot)
m[1].metric("Near-term ATM IV", f"{ts.iloc[0]:.1%}", f"expiry {ts.index[0]:%Y-%m-%d}", delta_color="off")
m[2].metric("Longest ATM IV", f"{ts.iloc[-1]:.1%}", f"expiry {ts.index[-1]:%Y-%m-%d}", delta_color="off")
m[3].metric("Points on surface", f"{len(pts):,}")
how = "Deribit mark IV where prices are unusable; otherwise computed from mid prices" if crypto else "computed by us from bid/ask mid prices (Black-Scholes, out-of-the-money options)"
st.caption(f"Source: **{src}** · IV {how} · retrieved {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC")

view = st.radio("View", ["3D surface", "Smile by expiry", "Term structure"], horizontal=True)
if view == "3D surface":
    # Bin moneyness so each expiry contributes one row of the grid.
    pts["m_bin"] = (pts["moneyness"] / 0.025).round() * 0.025
    grid = pts.pivot_table(index="days", columns="m_bin", values="iv", aggfunc="mean").sort_index()
    grid = grid.interpolate(axis=1, limit_area="inside")
    fig = go.Figure(go.Surface(x=grid.columns.values, y=grid.index.values, z=grid.values * 100, colorscale="Viridis",
                               colorbar=dict(title="IV %")))
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0),
                      scene=dict(xaxis_title="strike / spot", yaxis_title="days to expiry", zaxis_title="IV %",
                                 camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8))))
    st.plotly_chart(fig, use_container_width=True)
elif view == "Smile by expiry":
    fig = go.Figure()
    for exp, g in pts.groupby("expiry"):
        fig.add_scatter(x=g["moneyness"], y=g["iv"] * 100, mode="lines+markers", name=f"{pd.Timestamp(exp):%Y-%m-%d}", marker=dict(size=4))
    fig.add_vline(x=1.0, line=dict(dash="dot", color="gray"))
    fig.update_layout(height=520, xaxis_title="strike / spot (left of 1 = puts, right = calls)", yaxis_title="IV %",
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Higher IV on the left is the 'skew': investors pay more for protection against falls.")
else:
    fig = go.Figure(go.Scatter(x=ts.index, y=ts.values * 100, mode="lines+markers"))
    fig.update_layout(height=420, xaxis_title="expiry", yaxis_title="ATM IV %", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Upward slope = calm now, uncertainty later (normal). Downward slope = stress now (the market expects it to fade).")

with st.expander("Data table"):
    cols = ["expiry", "strike", "type", "bid", "ask", "price", "iv", "iv_source"] + [c for c in ("yahoo_iv", "quoted_iv") if c in pts]
    st.dataframe(pts[cols].round(4), use_container_width=True, hide_index=True)
with st.expander("How the surface is built"):
    st.markdown(op.__doc__)
