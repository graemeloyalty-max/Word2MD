import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gterm import ui
from gterm.analysis import alternatives as alt
from gterm.core import config
from gterm.data import registry
from gterm.data.base import DataSourceError
from gterm.quant import universe

ui.page_setup("Alternative Investments", "🪙")
settings = config.load_settings()

c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
classes = c1.multiselect("Asset classes", list(alt.UNIVERSE), default=["Crypto", "Commodities", "REITs", "Bonds (ETFs)", "Equities (reference)"])
base = c2.selectbox("Currency", config.BASE_CURRENCIES, index=config.BASE_CURRENCIES.index(settings["base_currency"]))
period = c3.selectbox("History", ["1y", "2y", "5y"], index=1)
bench = c4.selectbox("Equity benchmark", ["URTH", "^GSPC", "^STOXX50E"], help="URTH = MSCI World ETF")

labels, cls = {}, {}
for k in classes:
    for s, n in alt.UNIVERSE[k].items():
        labels[s], cls[s] = n, k
extra = st.text_input("Add your own symbols (comma-separated, Yahoo format)", "", placeholder="e.g. PAXG-USD, URA, NPN.JO")
for s in [x.strip().upper() for x in extra.split(",") if x.strip()]:
    labels.setdefault(s, s)
    cls.setdefault(s, "Custom")
if not labels:
    st.stop()


@st.cache_data(ttl=6 * 3600, show_spinner="Loading prices and converting currencies…")
def load(symbols, base, period):
    return universe.load_prices(list(symbols), base, period)


try:
    u = load(tuple(sorted(set(labels) | {bench})), base, period)
except DataSourceError as exc:
    st.error(str(exc))
    st.stop()
if u.errors:
    st.caption("No data for: " + ", ".join(e.split(":")[0] for e in u.errors))
prices = u.prices
table = alt.summary(prices.drop(columns=[bench], errors="ignore") if bench not in labels else prices, prices[bench], labels, cls)

st.dataframe(table.style.format({"Last": "{:,.2f}", "1 month": "{:+.1%}", "1 year": "{:+.1%}", "Period": "{:+.1%}", "Volatility": "{:.1%}",
                                 "Max drawdown": "{:.1%}", "Corr. with equities": "{:+.2f}"}, na_rep="–")
             .background_gradient(subset=["Corr. with equities"], cmap="RdYlGn_r", vmin=-1, vmax=1),
             use_container_width=True, hide_index=True, height=min(38 * (len(table) + 1), 600))
st.caption(f"Source: **{u.source}** · prices to {prices.index[-1]:%Y-%m-%d} · in {base} at daily FX · correlation with {bench} on "
           "daily weekday returns. Green = diversifies equities, red = moves with them.")

g1, g2 = st.columns([3, 2])
with g1:
    show = st.multiselect("Chart", list(table["Symbol"]), default=list(table.groupby("Class").head(1)["Symbol"]))
    fig = go.Figure()
    for s in show:
        ser = prices[s].dropna()
        fig.add_scatter(x=ser.index, y=ser / ser.iloc[0] * 100, name=labels.get(s, s))
    fig.update_layout(height=420, title=f"Growth of 100 ({base})", margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
with g2:
    fig = px.scatter(table, x="Volatility", y="1 year", color="Class", hover_name="Name", text="Symbol",
                     title="Risk vs 1-year return")
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0), xaxis_tickformat=".0%", yaxis_tickformat=".0%", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Correlation matrix"):
    cm = alt.correlation(prices[[s for s in table["Symbol"]]])
    fig = px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=60 + 28 * len(cm), margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("US Treasury yields")
rows = []
for s, n in alt.YIELDS.items():
    try:
        h = registry.prices(s, "1y")
        c = h.frame["Close"]
        rows.append({"Yield": n, "Level %": c.iloc[-1], "1y ago %": c.iloc[0], "Change (pp)": c.iloc[-1] - c.iloc[0],
                     "Date": f"{h.last_date:%Y-%m-%d}", "Source": h.source})
    except DataSourceError:
        pass
if rows:
    st.dataframe(pd.DataFrame(rows).style.format({"Level %": "{:.2f}", "1y ago %": "{:.2f}", "Change (pp)": "{:+.2f}"}),
                 use_container_width=True, hide_index=True)
with st.expander("Notes on these instruments"):
    st.markdown(alt.__doc__)
