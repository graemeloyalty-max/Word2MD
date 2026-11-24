import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gterm import ui
from gterm.analysis import portfolio as pa
from gterm.core import config, fx, portfolio_loader, watchlists
from gterm.data.base import DataSourceError

ui.page_setup("Portfolio", "💼")
settings = config.load_settings()
watchlists.seed_portfolio()

# ---- choose / edit portfolio ---------------------------------------------
c1, c2, c3, c4 = st.columns([3, 1.2, 1, 1.4])
names = watchlists.list_portfolios()
name = c1.selectbox("Portfolio", names) if names else None
base = c2.selectbox("Report in", config.BASE_CURRENCIES, index=config.BASE_CURRENCIES.index(settings["base_currency"]))
period = c3.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
bench_sym = c4.selectbox("Benchmark", ["URTH", "^GSPC", "^STOXX50E", "SPY", "EZU"], index=0,
                         help="URTH = iShares MSCI World ETF (global developed markets). All converted to your base currency.")
with st.popover("➕ New portfolio"):
    new = st.text_input("Name", key="newpf")
    if st.button("Create") and new:
        watchlists.save_portfolio(new, [])
        st.rerun()
if not name:
    st.stop()

with st.expander("✏️ Edit holdings", expanded=False):
    st.caption("Quantity = number of shares/units. Cost price is optional, per share in the share's own trading currency "
               "(e.g. pence for .L shares). Add rows at the bottom; select a row and press Delete to remove it.")
    edited = st.data_editor(pd.DataFrame(watchlists.get_holdings(name), columns=["symbol", "quantity", "cost_price"]),
                            num_rows="dynamic", use_container_width=True, key=f"ed_{name}",
                            column_config={"symbol": st.column_config.TextColumn("Symbol", required=True),
                                           "quantity": st.column_config.NumberColumn("Quantity", required=True),
                                           "cost_price": st.column_config.NumberColumn("Cost price (optional)")})
    b1, b2 = st.columns([1, 5])
    if b1.button("💾 Save", type="primary"):
        watchlists.save_portfolio(name, edited.to_dict("records"))
        st.success("Saved.")
        st.rerun()
    if b2.button("🗑 Delete portfolio"):
        watchlists.delete_portfolio(name)
        st.rerun()

holdings = watchlists.get_holdings(name)
if not holdings:
    st.info("This portfolio is empty. Add holdings above.")
    st.stop()

try:
    with st.spinner("Pricing holdings and converting currencies…"):
        lp = portfolio_loader.load(holdings, base, period)
        bench = portfolio_loader.benchmark(bench_sym, base, period)
except DataSourceError as exc:
    st.error(str(exc))
    st.stop()
for e in lp.errors:
    st.warning(f"Skipped {e}")

pos = pa.position_values(lp.prices_base, lp.quantities)
total = pos.sum(axis=1)
bench_aligned = bench.reindex(total.index).ffill().dropna()
meta = lp.meta

# ---- headline metrics ------------------------------------------------------
period_ret = total.iloc[-1] / total.iloc[0] - 1
bench_ret = bench_aligned.iloc[-1] / bench_aligned.iloc[0] - 1
m = st.columns(6)
m[0].metric(f"Value ({base})", f"{total.iloc[-1]:,.0f}")
m[1].metric(f"Return ({period})", f"{period_ret:+.1%}", f"{period_ret - bench_ret:+.1%} vs {bench_sym}")
m[2].metric("Volatility (ann.)", f"{pa.volatility(total):.1%}")
m[3].metric(f"Beta vs {bench_sym}", f"{pa.beta(total, bench_aligned):.2f}")
m[4].metric("Max drawdown", f"{pa.drawdown(total).min():.1%}")
rf = m[5].number_input("Risk-free for Sharpe (%)", 0.0, 20.0, 0.0, 0.25) / 100
m[5].caption(f"Sharpe: **{pa.sharpe(total, rf):.2f}**")
st.caption(f"Source: **{lp.source}** · prices to {total.index[-1]:%Y-%m-%d} · {len(lp.quantities)} holdings priced · "
           f"{'; '.join(n for n in lp.fx_notes if n != 'no conversion') or 'no FX conversion needed'} · "
           "current quantities held constant over the period")

# ---- holdings table ---------------------------------------------------------
last_native = meta["last_price"].astype(float)
tbl = pd.DataFrame({
    "Name": meta["name"], "Sector": meta["sector"], "Region": meta["region"], "Ccy": meta["currency"],
    "Quantity": pd.Series(lp.quantities), "Price": last_native,
    f"Value ({base})": pos.iloc[-1], "Weight": pos.iloc[-1] / total.iloc[-1],
    f"Return ({base})": pos.iloc[-1] / pos.iloc[0] - 1,
    "Contribution": pa.contributions(pos),
})
cost = pd.to_numeric(meta["cost_price"], errors="coerce")
if cost.notna().any():
    # P&L vs cost: compare in each share's trading currency, then convert the gain at today's rate.
    gains = {}
    for s in tbl.index:
        if pd.notna(cost.get(s)):
            rate = fx.latest_rate(meta.loc[s, "currency"], base)[0]
            gains[s] = (last_native[s] - cost[s]) * lp.quantities[s] * rate
    tbl[f"Gain vs cost ({base})"] = pd.Series(gains)
st.dataframe(tbl.style.format({"Price": "{:,.2f}", f"Value ({base})": "{:,.0f}", "Weight": "{:.1%}", "Quantity": "{:,.4g}",
                               f"Return ({base})": "{:+.1%}", "Contribution": "{:+.2%}", f"Gain vs cost ({base})": "{:+,.0f}"},
                              na_rep="–"), use_container_width=True)

# ---- charts ---------------------------------------------------------------------
g1, g2 = st.columns([3, 2])
with g1:
    fig = go.Figure()
    fig.add_scatter(x=total.index, y=total / total.iloc[0] * 100, name=name)
    fig.add_scatter(x=bench_aligned.index, y=bench_aligned / bench_aligned.iloc[0] * 100, name=bench_sym, line=dict(dash="dot"))
    fig.update_layout(height=320, title=f"Growth of 100 ({base})", margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
    dd = pa.drawdown(total)
    fig = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", line=dict(color="crimson", width=1)))
    fig.update_layout(height=200, title="Drawdown (%)", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
with g2:
    by = st.radio("Allocation by", ["sector", "region", "currency"], horizontal=True)
    w = pd.DataFrame({"group": meta[by], "value": pos.iloc[-1]}).groupby("group")["value"].sum()
    fig = px.pie(values=w.values, names=w.index, hole=0.45)
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Performance attribution")
a1, a2 = st.columns(2)
for col, key, label in [(a1, "sector", "Sector"), (a2, "region", "Region")]:
    with col:
        att = pa.attribution(pos, meta[key].to_dict())
        fig = go.Figure(go.Bar(x=att["contribution"] * 100, y=att.index, orientation="h",
                               marker_color=["seagreen" if v >= 0 else "crimson" for v in att["contribution"]]))
        fig.update_layout(height=60 + 34 * len(att), title=f"Contribution to return by {label.lower()} (percentage points)",
                          margin=dict(l=0, r=0, t=40, b=0), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(att.rename(columns={"start_weight": "Start weight", "end_weight": "End weight", "contribution": "Contribution"})
                     .style.format("{:.1%}"), use_container_width=True)
st.caption(f"Contributions add up to the portfolio return of {period_ret:+.1%}. Returns are in {base}, so currency moves are included.")

st.subheader("Correlation of daily returns")
corr = pa.correlation(lp.prices_base)
fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
fig.update_layout(height=80 + 40 * len(corr), margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Computed on daily returns in {base}. Values near +1 mean two holdings move together and add little diversification.")
with st.expander("How these numbers are calculated"):
    st.markdown(pa.__doc__)
