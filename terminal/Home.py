"""Global Equity Terminal: home dashboard. Run with:  streamlit run Home.py"""
import pandas as pd
import streamlit as st

from gterm import ui
from gterm.analysis import portfolio as pa
from gterm.core import config, portfolio_loader, watchlists
from gterm.data import macro, news, registry
from gterm.data.base import DataSourceError

ui.page_setup("Dashboard", "🌍")
watchlists.seed_defaults()
watchlists.seed_portfolio()
settings = config.load_settings()

WIDGETS = ["Market snapshot", "Portfolio", "Watchlist", "Key rates", "Headlines", "Build progress"]
with st.expander("⚙️ Customise dashboard"):
    chosen = st.multiselect("Widgets, in display order", WIDGETS, default=[w for w in settings["dashboard_widgets"] if w in WIDGETS])
    if st.button("Save layout"):
        config.save_settings({**settings, "dashboard_widgets": chosen})
        st.rerun()
base = settings["base_currency"]


def market_snapshot():
    st.markdown("##### Market snapshot")
    labels = {"^GSPC": "S&P 500", "^STOXX50E": "EURO STOXX 50", "URTH": "MSCI World (ETF)", "EURUSD=X": "EUR/USD",
              "USDZAR=X": "USD/ZAR", "BTC-USD": "Bitcoin (USD)"}
    cols = st.columns(3)
    for i, (sym, label) in enumerate(labels.items()):
        with cols[i % 3]:
            try:
                q = registry.quote(sym)
                st.metric(label, ui.fmt_price(q.price), f"{q.change_pct:+.2f}%")
            except DataSourceError as exc:
                st.caption(f"{sym}: {exc}")
    st.caption(f"Source: **{registry.active_source_name()}** · last close or live price")


def portfolio_widget():
    names = watchlists.list_portfolios()
    if not names:
        st.info("No portfolio yet: create one on the Portfolio page.")
        return
    name = st.selectbox("Portfolio", names, key="dash_pf", label_visibility="collapsed")
    try:
        lp = portfolio_loader.load(watchlists.get_holdings(name), base, "1y")
    except DataSourceError as exc:
        st.warning(str(exc))
        return
    total = pa.position_values(lp.prices_base, lp.quantities).sum(axis=1)
    c = st.columns(3)
    c[0].metric(f"Value ({base})", f"{total.iloc[-1]:,.0f}")
    day = total.iloc[-1] / total.iloc[-2] - 1 if len(total) > 1 else 0
    c[1].metric("Last day", f"{day:+.2%}")
    c[2].metric("1 year", f"{total.iloc[-1] / total.iloc[0] - 1:+.1%}")
    st.line_chart(total / total.iloc[0] * 100, height=150)
    st.caption(f"{name} · in {base} · Source: **{lp.source}** · details on the Portfolio page")


def watchlist_widget():
    lists = watchlists.list_watchlists()
    if not lists:
        return
    wl = st.selectbox("Watchlist", lists, key="dash_wl", label_visibility="collapsed")
    rows = []
    for s in watchlists.symbols(wl)[:10]:
        try:
            q = registry.quote(s)
            rows.append({"Symbol": s, "Last": q.price, "Chg %": q.change_pct, "Ccy": q.currency})
        except DataSourceError:
            pass
    if rows:
        st.dataframe(pd.DataFrame(rows).style.format({"Last": "{:,.2f}", "Chg %": "{:+.2f}%"}), hide_index=True, use_container_width=True)
    st.caption(f"Source: **{registry.active_source_name()}**")


def key_rates():
    st.markdown("##### Key rates")
    rows = []
    for sym, label in [("^IRX", "US 13-week T-bill"), ("^FVX", "US 5-year Treasury"), ("^TNX", "US 10-year Treasury")]:
        try:
            h = registry.prices(sym, "1mo")
            rows.append({"Rate": label, "Level %": h.last_close, "Date": f"{h.last_date:%Y-%m-%d}", "Source": h.source})
        except DataSourceError:
            pass
    for label in ["US Fed funds (effective)", "ECB deposit facility"]:
        try:
            ms = macro.policy_rate(label)
            d, v = ms.last
            rows.append({"Rate": label, "Level %": v, "Date": f"{d:%Y-%m-%d}", "Source": ms.source})
        except DataSourceError:
            rows.append({"Rate": label, "Level %": None, "Date": "–", "Source": "unavailable (host blocked?)"})
    st.dataframe(pd.DataFrame(rows).style.format({"Level %": "{:.2f}"}, na_rep="–"), hide_index=True, use_container_width=True)


def headlines():
    st.markdown("##### Headlines")
    topic = st.selectbox("Topic", ["markets", "South Africa economy", "Eurozone economy", "Federal Reserve"], key="dash_news",
                         label_visibility="collapsed")
    try:
        df = news.to_frame(news.google_news(topic))
    except DataSourceError as exc:
        st.caption(str(exc))
        return
    for _, a in df.head(8).iterrows():
        when = f"{a['published']:%d %b %H:%M}" if pd.notna(a["published"]) else ""
        st.markdown(f"[{a['title']}]({a['url']})  \n<small>{a['publisher']} · {when}</small>", unsafe_allow_html=True)
    st.caption("Source: **Google News** · more on the News & Events page")


def build_progress():
    st.markdown("##### Build progress")
    st.markdown("""
| Phase | Area | Status |
|---|---|---|
| 1 | Shell, data layer, cache, Settings, quotes & charts, watchlists | ✅ |
| 2 | Equity research: statements, DCF, multiples, quality, country overlay | ✅ |
| 3 | Portfolio, macro & central banks, news & GDELT, this dashboard | ✅ |
| 4 | Quant lab: signals, backtests, sweeps, vol surface | next |
| 5 | Agentic research (Anthropic API key) | |
| 6 | Everything else | |
""")


RENDER = {"Market snapshot": market_snapshot, "Portfolio": portfolio_widget, "Watchlist": watchlist_widget,
          "Key rates": key_rates, "Headlines": headlines, "Build progress": build_progress}
TITLES = {"Portfolio": "##### Portfolio", "Watchlist": "##### Watchlist"}

left, right = st.columns(2)
for i, w in enumerate(chosen):
    with (left if i % 2 == 0 else right):
        with st.container(border=True):
            if w in TITLES:
                st.markdown(TITLES[w])
            RENDER[w]()
