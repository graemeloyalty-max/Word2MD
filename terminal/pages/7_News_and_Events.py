import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gterm import ui
from gterm.data import news, registry
from gterm.data.base import DataSourceError

ui.page_setup("News & Events", "📰")


def show_articles(df: pd.DataFrame, limit: int = 40):
    if df.empty:
        st.info("No articles found.")
        return
    for _, a in df.head(limit).iterrows():
        when = f"{a['published']:%Y-%m-%d %H:%M} UTC" if pd.notna(a["published"]) else "date unknown"
        extra = f" · {a['country']}" if a.get("country") else ""
        st.markdown(f"**[{a['title']}]({a['url']})**  \n<small>{a['publisher'] or '–'} · {when}{extra} · via {a['source']}</small>",
                    unsafe_allow_html=True)


def safe(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except DataSourceError as exc:
        st.warning(str(exc))
        return []


t_co, t_topic, t_geo = st.tabs(["🏢 Company", "🌍 Country / topic", "⚠️ Geopolitical events (GDELT)"])

with t_co:
    sym = st.text_input("Ticker", st.query_params.get("symbol", "SAP.DE")).strip().upper()
    if sym:
        try:
            name = registry.quote(sym).name
        except DataSourceError:
            name = sym
        arts = safe(news.company_news, sym) + safe(news.google_news, f'"{name}"')
        st.caption(f"News for **{name}** from Yahoo Finance and a Google News search for the company name.")
        show_articles(news.to_frame(arts))

with t_topic:
    c1, c2 = st.columns([3, 1])
    presets = ["South Africa economy", "Eurozone inflation", "Federal Reserve", "ECB interest rates", "SARB repo rate",
               "oil prices OPEC", "China economy"]
    q = c1.selectbox("Pick a topic, or type your own below", presets)
    custom = c1.text_input("Custom search (optional)")
    region = c2.selectbox("Edition", ["US", "GB", "ZA", "DE", "FR"], help="Google News regional edition")
    query = custom.strip() or q
    show_articles(news.to_frame(safe(news.google_news, query, "en", region)))

with t_geo:
    st.caption("GDELT monitors broadcast, print and web news worldwide in 100+ languages, updated every 15 minutes. "
               "Queries use GDELT syntax: words, \"exact phrases\", OR, and filters like sourcecountry:southafrica.")
    c1, c2 = st.columns([3, 1])
    presets = {"Conflict & sanctions": "(conflict OR sanctions OR military OR missile)",
               "Elections": "(election OR referendum)",
               "Central banks": '("central bank" OR "interest rate decision")',
               "Trade & tariffs": "(tariff OR \"trade war\" OR \"export controls\")",
               "Energy supply": "(pipeline OR \"oil supply\" OR LNG) (disruption OR outage OR attack)",
               "South Africa": "sourcecountry:southafrica (economy OR rand OR eskom OR government)"}
    pick = c1.selectbox("Theme", list(presets))
    q = c1.text_input("GDELT query", presets[pick], key=f"gq_{pick}")
    span = c2.selectbox("Window", ["24h", "3d", "7d", "1m"], index=1)
    try:
        arts, gdelt_ok = news.gdelt_articles(q, span, 250), True
    except DataSourceError as exc:
        gdelt_ok = False
        st.warning(f"GDELT is unavailable right now ({exc}). Showing Google News for the same theme instead; "
                   "the coverage map and tone chart need GDELT.")
        arts = safe(news.google_news, news.plain_query(q))
    df = news.to_frame(arts)
    if not df.empty and not gdelt_ok:
        show_articles(df, 60)
    elif not df.empty:
        g1, g2 = st.columns([3, 2])
        with g1:
            counts = df[df["country"] != ""].groupby("country").size().reset_index(name="articles")
            fig = px.choropleth(counts, locations="country", locationmode="country names", color="articles",
                                color_continuous_scale="Reds", title="Where the coverage comes from (articles by source country)")
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            tl = safe(news.gdelt_timeline, q, "timelinetone", "3m")
            if isinstance(tl, pd.DataFrame) and not tl.empty:
                daily = tl.iloc[:, 0].resample("D").mean()
                fig = go.Figure(go.Bar(x=daily.index, y=daily.values,
                                       marker_color=["crimson" if v < 0 else "seagreen" for v in daily.values]))
                fig.update_layout(height=380, title="Average tone of coverage, last 3 months (below 0 = negative)",
                                  margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Source: **GDELT 2.0 DOC API** · {len(df)} articles in the last {span} · map counts the publisher's country, "
                   "which is where the story was covered, not necessarily where the event happened.")
        show_articles(df, 60)
