import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gterm import ui
from gterm.data import macro
from gterm.data.base import DataSourceError

ui.page_setup("Macro & Central Banks", "🏛️")


def fetch(fn, *args, **kw):
    """Run one fetch; show the reason (e.g. a blocked host) instead of failing the page."""
    try:
        return fn(*args, **kw)
    except DataSourceError as exc:
        st.warning(str(exc))
        return None


def source_caption(items):
    srcs = {}
    for ms in items:
        if ms is not None:
            srcs.setdefault(ms.source, ms)
    return " · ".join(f"Source: **{k}** ([link]({v.url.split('?')[0]})) retrieved {v.as_of:%Y-%m-%d %H:%M} UTC" for k, v in srcs.items())


t_cty, t_cb, t_curve, t_sarb, t_explore = st.tabs(
    ["🌐 Countries", "🏦 Policy rates", "📈 Yield curves", "🇿🇦 SARB", "🔎 Series explorer"])

# ---------------------------------------------------------------- countries
with t_cty:
    c1, c2, c3 = st.columns([2, 3, 1])
    label = c1.selectbox("Indicator", list(macro.INDICATORS))
    countries = c2.multiselect("Countries", list(macro.COUNTRIES), default=["United States", "Euro area", "United Kingdom", "South Africa"])
    has_imf = macro.INDICATORS[label][1] is not None
    src = c3.radio("Source", ["World Bank", "IMF"] if has_imf else ["World Bank"],
                   help="IMF World Economic Outlook includes forecasts (dashed). World Bank is actuals only.")
    years = st.slider("Years", 2000, 2031, (2010, 2030 if src == "IMF" else 2025))
    fig, latest_rows, got = go.Figure(), [], []
    for ctry in countries:
        ms = fetch(macro.country_indicator, label, ctry, src)
        if ms is None:
            continue
        got.append(ms)
        sr = ms.series[(ms.series.index.year >= years[0]) & (ms.series.index.year <= years[1])]
        if ms.forecast_from is not None:
            act, fc = sr[sr.index < ms.forecast_from], sr[sr.index >= pd.Timestamp(ms.forecast_from) - pd.offsets.YearEnd(1)]
            line = fig.add_scatter(x=act.index.year, y=act.values, name=ctry, mode="lines+markers")
            fig.add_scatter(x=fc.index.year, y=fc.values, name=f"{ctry} (forecast)", mode="lines", line=dict(dash="dash"))
        else:
            fig.add_scatter(x=sr.index.year, y=sr.values, name=ctry, mode="lines+markers")
        actual = ms.series[ms.series.index < ms.forecast_from] if ms.forecast_from is not None else ms.series
        d, v = actual.index[-1], actual.iloc[-1]
        latest_rows.append({"Country": ctry, "Latest actual": v, "Year": d.year, "Source": ms.source})
    fig.update_layout(height=420, yaxis_title=macro.INDICATORS[label][2], margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    if latest_rows:
        st.dataframe(pd.DataFrame(latest_rows).style.format({"Latest actual": "{:.2f}"}), hide_index=True, use_container_width=True)
        st.caption(source_caption(got))

# ---------------------------------------------------------------- policy rates
with t_cb:
    sel = st.multiselect("Central bank rates", list(macro.POLICY_RATES), default=list(macro.POLICY_RATES)[:3])
    start = st.date_input("From", pd.Timestamp("2015-01-01"))
    fig, got = go.Figure(), []
    cols = st.columns(max(len(sel), 1))
    for col, lab in zip(cols, sel):
        ms = fetch(macro.policy_rate, lab)
        if ms is None:
            continue
        got.append(ms)
        sr = ms.series[ms.series.index >= pd.Timestamp(start)]
        fig.add_scatter(x=sr.index, y=sr.values, name=lab, line=dict(shape="hv"))
        d, v = ms.last
        col.metric(lab, f"{v:.2f}%", f"as of {d:%Y-%m-%d}", delta_color="off")
    fig.update_layout(height=420, yaxis_title="%", margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(source_caption(got))

# ---------------------------------------------------------------- yield curves
with t_curve:
    which = st.radio("Curve", ["US Treasury", "Euro area AAA government"], horizontal=True)
    ago = st.select_slider("Compare with", ["1 month ago", "6 months ago", "1 year ago", "2 years ago"], value="1 year ago")
    offs = {"1 month ago": 30, "6 months ago": 182, "1 year ago": 365, "2 years ago": 730}[ago]
    now = fetch(macro.yield_curve, which)
    if now is not None:
        curve, src, d = now
        then = fetch(macro.yield_curve, which, d - pd.Timedelta(days=offs))
        fig = go.Figure()
        fig.add_scatter(x=curve.index, y=curve.values, name=f"{d:%Y-%m-%d}", mode="lines+markers")
        if then is not None:
            fig.add_scatter(x=then[0].index, y=then[0].values, name=f"{then[2]:%Y-%m-%d}", mode="lines+markers", line=dict(dash="dot"))
        fig.update_layout(height=420, xaxis_title="maturity (years)", yaxis_title="yield %", xaxis_type="log",
                          margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        spread = curve.get(10.0, float("nan")) - curve.get(2.0, float("nan"))
        st.metric("10y − 2y spread", f"{spread * 100:+.0f} bp", "inverted: often precedes recessions" if spread < 0 else "normal (upward sloping)",
                  delta_color="off")
        st.caption(f"Source: **{src}** · curve as of {d:%Y-%m-%d}" + (" · euro curve = ECB AAA-rated euro-area government bonds, spot rates" if src.startswith("ECB") else ""))

# ---------------------------------------------------------------- SARB
with t_sarb:
    res = fetch(macro.sarb_rates)
    if res is not None:
        df, url = res
        st.dataframe(df.style.format({"Value": "{:.3f}", "Date": lambda x: f"{x:%Y-%m-%d}" if pd.notna(x) else "–"}),
                     hide_index=True, use_container_width=True)
        st.caption(f"Source: **South African Reserve Bank** web indicators ([link]({url})) · current values")
    st.markdown("**South Africa over time** (OECD series via FRED)")
    fig, got = go.Figure(), []
    for sid, lab in [("IR3TIB01ZAM156N", "3-month interbank rate"), ("IRLTLT01ZAM156N", "10-year government bond yield")]:
        ms = fetch(macro.fred, sid, lab, "%")
        if ms is not None:
            got.append(ms)
            fig.add_scatter(x=ms.series.index, y=ms.series.values, name=lab)
    fig.update_layout(height=360, yaxis_title="%", margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(source_caption(got))

# ---------------------------------------------------------------- explorer
with t_explore:
    st.markdown("Fetch any series by its code. Examples: FRED `CPIAUCSL`, `UNRATE`, `T10YIE`; World Bank `NY.GDP.PCAP.CD`; "
                "ECB `ICP.M.U2.N.000000.4.ANR` (euro-area HICP inflation).")
    c1, c2, c3 = st.columns([1, 2, 1])
    kind = c1.selectbox("Source", ["FRED", "World Bank", "ECB"])
    code = c2.text_input("Series code", "CPIAUCSL" if kind == "FRED" else "NY.GDP.PCAP.CD" if kind == "World Bank" else "ICP.M.U2.N.000000.4.ANR")
    ctry = c3.selectbox("Country (World Bank)", list(macro.COUNTRIES), disabled=kind != "World Bank")
    if code:
        ms = fetch({"FRED": lambda: macro.fred(code.strip()), "World Bank": lambda: macro.worldbank(macro.COUNTRIES[ctry], code.strip()),
                    "ECB": lambda: macro.ecb(code.strip())}[kind])
        if ms is not None:
            fig = go.Figure(go.Scatter(x=ms.series.index, y=ms.series.values, name=ms.title))
            fig.update_layout(height=380, title=ms.title, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            d, v = ms.last
            st.caption(f"Latest {v:,.3f} on {d:%Y-%m-%d} · " + source_caption([ms]))
            st.download_button("Download CSV", ms.series.rename(ms.id).to_csv().encode(), f"{ms.id}.csv", "text/csv")
