import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from gterm import ui
from gterm.core import config, fx, indicators, watchlists
from gterm.data import registry
from gterm.data.base import INTERVALS, PERIODS, DataSourceError

ui.page_setup("Quotes & Charts", "📈")
settings = config.load_settings()

# ---- symbol picker -------------------------------------------------------
c1, c2, c3, c4 = st.columns([3, 1, 1, 1.4])
symbol = c1.text_input(
    "Symbol",
    value=st.query_params.get("symbol", "AAPL"),
    help="Yahoo-style symbols: AAPL, SAP.DE, ASML.AS, SHEL.L, ^GSPC, EURUSD=X, BTC-USD",
).strip().upper()
period = c2.selectbox("Period", PERIODS, index=PERIODS.index("1y"))
interval = c3.selectbox("Interval", INTERVALS)
view_ccy = c4.radio("Show in", ["Native", settings["base_currency"]], horizontal=True)

with st.expander("Search for a symbol"):
    q = st.text_input("Company or ticker name", key="search")
    if q:
        results = registry.search(q)
        st.dataframe(results or [{"result": "no matches"}], use_container_width=True, hide_index=True)

if not symbol:
    st.stop()

try:
    quote = registry.quote(symbol)
    hist = registry.prices(symbol, period, interval)
except DataSourceError as exc:
    st.error(str(exc))
    st.stop()

df = hist.frame.copy()
ccy = hist.currency
fx_note = None
if view_ccy != "Native":
    try:
        for col in ["Open", "High", "Low", "Close"]:
            df[col], fx_note = fx.convert_series(df[col], hist.currency, view_ccy, period="max")
        ccy = view_ccy
    except (DataSourceError, ValueError) as exc:
        st.warning(f"Could not convert to {view_ccy}: {exc}. Showing native currency.")

# ---- headline ------------------------------------------------------------
st.subheader(f"{quote.name}  ·  {quote.symbol}")
m = st.columns(5)
m[0].metric(f"Last ({quote.currency})", ui.fmt_price(quote.price), f"{quote.change:+,.2f} ({quote.change_pct:+.2f}%)")
m[1].metric(f"Return ({period})", f"{indicators.total_return(df['Close']) * 100:+.1f}%")
per_year = {"1d": 252, "1wk": 52, "1mo": 12}[interval]
m[2].metric("Volatility (ann.)", f"{indicators.annualised_volatility(df['Close'], per_year) * 100:.1f}%")
m[3].metric("Max drawdown", f"{indicators.max_drawdown(df['Close']) * 100:.1f}%")
m[4].metric("Type / exchange", quote.asset_type.title(), quote.exchange or "–", delta_color="off")
st.caption(ui.provenance(quote.source, quote.as_of, quote.price_date))

# ---- chart ---------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Indicators")
    show_sma = st.multiselect("SMA", [20, 50, 100, 200], default=[50, 200])
    show_ema = st.multiselect("EMA", [9, 12, 26, 50], default=[])
    show_bb = st.checkbox("Bollinger bands (20, 2σ)")
    show_rsi = st.checkbox("RSI (14)", value=True)
    show_macd = st.checkbox("MACD (12, 26, 9)")
    chart_type = st.radio("Chart", ["Candles", "Line"], horizontal=True)

rows = 2 + show_rsi + show_macd
heights = [0.55, 0.15] + [0.15] * (rows - 2)
fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights)
close = df["Close"]
if chart_type == "Candles":
    fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=close, name=symbol), 1, 1)
else:
    fig.add_trace(go.Scatter(x=df.index, y=close, name=symbol, line=dict(width=1.6)), 1, 1)
for w in show_sma:
    fig.add_trace(go.Scatter(x=df.index, y=indicators.sma(close, w), name=f"SMA {w}", line=dict(width=1)), 1, 1)
for w in show_ema:
    fig.add_trace(go.Scatter(x=df.index, y=indicators.ema(close, w), name=f"EMA {w}", line=dict(width=1, dash="dot")), 1, 1)
if show_bb:
    bb = indicators.bollinger(close)
    fig.add_trace(go.Scatter(x=df.index, y=bb.upper, name="BB upper", line=dict(width=0.8, color="gray")), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=bb.lower, name="BB lower", line=dict(width=0.8, color="gray"), fill="tonexty", fillcolor="rgba(128,128,128,0.1)"), 1, 1)
fig.add_trace(go.Bar(x=df.index, y=df.Volume, name="Volume", marker_color="rgba(100,120,160,0.5)"), 2, 1)
r = 3
if show_rsi:
    fig.add_trace(go.Scatter(x=df.index, y=indicators.rsi(close), name="RSI 14", line=dict(width=1.2)), r, 1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line=dict(dash="dash", width=0.7, color="gray"), row=r, col=1)
    fig.update_yaxes(range=[0, 100], title_text="RSI", row=r, col=1)
    r += 1
if show_macd:
    md = indicators.macd(close)
    fig.add_trace(go.Bar(x=df.index, y=md["hist"], name="MACD hist", marker_color="rgba(150,150,150,0.6)"), r, 1)
    fig.add_trace(go.Scatter(x=df.index, y=md["macd"], name="MACD", line=dict(width=1.2)), r, 1)
    fig.add_trace(go.Scatter(x=df.index, y=md["signal"], name="Signal", line=dict(width=1.2)), r, 1)
    fig.update_yaxes(title_text="MACD", row=r, col=1)
fig.update_yaxes(title_text=f"Price ({ccy})", row=1, col=1)
fig.update_yaxes(title_text="Vol", row=2, col=1)
fig.update_layout(height=520 + 140 * (rows - 2), xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.04))
st.plotly_chart(fig, use_container_width=True)
cap = ui.provenance(hist.source, hist.as_of, hist.last_date, hist.from_cache) + f" · {len(df)} bars, {interval}"
if fx_note:
    cap += f" · {fx_note}"
st.caption(cap)

# ---- add to watchlist / data table --------------------------------------
c1, c2 = st.columns([2, 3])
with c1:
    lists = watchlists.list_watchlists()
    if lists:
        target = st.selectbox("Add to watchlist", lists)
        if st.button(f"➕ Add {symbol}"):
            watchlists.add_symbol(target, symbol)
            st.toast(f"Added {symbol} to {target}")
with st.expander("Price data table"):
    st.dataframe(df.sort_index(ascending=False).round(4), use_container_width=True)
    st.download_button("Download CSV", df.to_csv().encode(), f"{symbol}_{period}_{interval}.csv", "text/csv")
