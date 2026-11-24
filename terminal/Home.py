"""Global Equity Terminal: home page. Run with:  streamlit run Home.py"""
import streamlit as st

from gterm import ui
from gterm.core import watchlists
from gterm.data import registry
from gterm.data.base import DataSourceError

ui.page_setup("Global Equity Terminal", "🌍")
watchlists.seed_defaults()

st.markdown(
    "A personal research terminal for global equity evaluation. "
    "**Phase 1** built the foundation; **Phase 2** adds the equity research core."
)

st.subheader("Market snapshot")
snapshot = ["^GSPC", "^STOXX50E", "EURUSD=X", "USDZAR=X", "BTC-USD"]
cols = st.columns(len(snapshot))
for col, sym in zip(cols, snapshot):
    with col:
        try:
            q = registry.quote(sym)
            st.metric(q.name, ui.fmt_price(q.price), f"{q.change_pct:+.2f}%")
            st.caption(ui.provenance(q.source, q.as_of, q.price_date))
        except DataSourceError as exc:
            st.error(f"{sym}: {exc}")

st.subheader("What's built so far")
st.markdown(
    """
| Area | Page | Status |
|---|---|---|
| Markets | **Quotes & Charts**: any equity, ETF, index, FX pair or crypto; candlesticks, SMA/EMA/Bollinger, RSI, MACD, base-currency view | ✅ Phase 1 |
| Markets | **Watchlists**: saved lists, quote table, price alerts | ✅ Phase 1 |
| Workspace | **Settings**: API keys (.env), base currency, data source, cache | ✅ Phase 1 |
| Research | **Equity Research**: normalised statements, DCF with reverse DCF and sensitivity, peer multiples, quality score, country & currency overlay | ✅ Phase 2 |
| Portfolio & Macro | Portfolio tools, FRED/World Bank/IMF/ECB/SARB, news, GDELT, dashboard | Phase 3 |
| Quant lab | Signals, backtests, sweeps, vol surface | Phase 4 |
| Agentic research | AI chat, agents, reports | Phase 5 |
"""
)
st.info("Use the sidebar to open a page.", icon="👈")
