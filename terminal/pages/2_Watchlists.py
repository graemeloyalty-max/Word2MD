import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gterm import ui
from gterm.core import watchlists
from gterm.data import registry
from gterm.data.base import DataSourceError

ui.page_setup("Watchlists", "👀")
watchlists.seed_defaults()

lists = watchlists.list_watchlists()
c1, c2 = st.columns([3, 2])
with c2:
    with st.popover("➕ New watchlist"):
        new = st.text_input("Name")
        if st.button("Create") and new:
            watchlists.create_watchlist(new)
            st.rerun()
if not lists:
    st.info("No watchlists yet. Create one.")
    st.stop()
current = c1.selectbox("Watchlist", lists)

auto = st.toggle("Auto-refresh every 60 s", value=False, help="Re-fetches quotes on a timer (cache lifetime is 5 minutes).")


@st.fragment(run_every=60 if auto else None)
def quote_table():
    syms = watchlists.symbols(current)
    if not syms:
        st.info("This watchlist is empty. Add a symbol below.")
        return
    rows, sparks, errors = [], {}, []
    for s in syms:
        try:
            q = registry.quote(s)
            h = registry.prices(s, "3mo", "1d")
            rows.append(
                {
                    "Symbol": s,
                    "Name": q.name,
                    "Last": q.price,
                    "Chg %": q.change_pct,
                    "Ccy": q.currency,
                    "3m trend": h.frame["Close"].round(4).tolist(),
                    "Price date": q.price_date.strftime("%Y-%m-%d"),
                    "Source": q.source,
                    "Retrieved (UTC)": q.as_of.strftime("%H:%M:%S"),
                }
            )
        except DataSourceError as exc:
            errors.append(f"{s}: {exc}")
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Last": st.column_config.NumberColumn(format="%.2f"),
                "Chg %": st.column_config.NumberColumn(format="%+.2f%%"),
                "3m trend": st.column_config.LineChartColumn(width="medium"),
            },
        )
        # Alerts check
        prices = {r["Symbol"]: r["Last"] for r in rows}
        for a in watchlists.list_alerts():
            if a.symbol in prices and a.triggered(prices[a.symbol]):
                st.warning(f"🔔 Alert: {a.symbol} is {a.direction} {a.level:,.2f} (last {prices[a.symbol]:,.2f})")
    for e in errors:
        st.error(e)


quote_table()

st.divider()
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Add / remove symbols**")
    add = st.text_input("Symbol to add", key="add").strip().upper()
    if st.button("Add") and add:
        try:
            registry.quote(add)  # validate before saving
            watchlists.add_symbol(current, add)
            st.rerun()
        except DataSourceError as exc:
            st.error(str(exc))
    syms = watchlists.symbols(current)
    if syms:
        rm = st.selectbox("Remove", syms)
        if st.button("Remove"):
            watchlists.remove_symbol(current, rm)
            st.rerun()
with c2:
    st.markdown("**Price alerts**")
    a_sym = st.text_input("Symbol", key="asym").strip().upper()
    a_dir = st.radio("When price is", ["above", "below"], horizontal=True)
    a_lvl = st.number_input("Level", min_value=0.0, step=1.0)
    if st.button("Add alert") and a_sym and a_lvl > 0:
        watchlists.add_alert(a_sym, a_dir, a_lvl)
        st.rerun()
    for a in watchlists.list_alerts():
        cc1, cc2 = st.columns([4, 1])
        cc1.write(f"{a.symbol} {a.direction} {a.level:,.2f}")
        if cc2.button("✕", key=f"del{a.id}"):
            watchlists.delete_alert(a.id)
            st.rerun()
with c3:
    st.markdown("**Compare (rebased to 100)**")
    syms = watchlists.symbols(current)
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    fig = go.Figure()
    for s in syms:
        try:
            c = registry.prices(s, period).frame["Close"]
            fig.add_trace(go.Scatter(x=c.index, y=c / c.iloc[0] * 100, name=s, line=dict(width=1.3)))
        except DataSourceError:
            pass
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Source: **{registry.active_source_name()}** · local-currency total price change, not FX-adjusted")

if st.button("🗑 Delete this watchlist", type="secondary"):
    watchlists.delete_watchlist(current)
    st.rerun()
