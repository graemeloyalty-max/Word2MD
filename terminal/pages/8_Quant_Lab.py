import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gterm import ui
from gterm.core import config
from gterm.data.base import DataSourceError
from gterm.quant import backtest, factors, strategies, universe

ui.page_setup("Quant Lab", "🧪")
settings = config.load_settings()

st.warning("**Read before trusting any result.** The universe is today's companies, so firms that failed or were delisted are "
           "missing (survivorship bias): results look better than reality. Free fundamentals cover only ~4 years, so value and "
           "quality tests are short. Past performance does not predict future returns.", icon="⚠️")

# ---- universe & global settings ---------------------------------------------
with st.expander("Universe and settings", expanded=False):
    txt = st.text_area("Universe (comma-separated tickers)", ", ".join(factors.DEFAULT_UNIVERSE), height=90)
    c1, c2, c3, c4 = st.columns(4)
    base = c1.selectbox("Currency", config.BASE_CURRENCIES, index=config.BASE_CURRENCIES.index(settings["base_currency"]))
    period = c2.selectbox("History", ["2y", "5y", "10y"], index=1)
    cost_bps = c3.number_input("Transaction cost (bps per trade)", 0.0, 100.0, 10.0, 1.0,
                               help="10 bps = 0.10% of the value traded. Covers commission, spread and market impact.")
    lag = c4.number_input("Execution lag (trading days)", 0, 5, 1,
                          help="Days between deciding and trading. 1 = trade at the next day's close. 0 is optimistic.")
symbols = tuple(dict.fromkeys(s.strip().upper() for s in txt.split(",") if s.strip()))


@st.cache_data(ttl=6 * 3600, show_spinner="Loading universe prices (first time takes about a minute)…")
def load_universe(symbols, base, period):
    return universe.load_prices(list(symbols), base, period)


@st.cache_data(ttl=24 * 3600, show_spinner="Loading fundamentals for value/quality…")
def load_fund(symbols, base):
    return universe.load_fundamentals(list(symbols), base)


try:
    u = load_universe(symbols, base, period)
except DataSourceError as exc:
    st.error(str(exc))
    st.stop()
prices = u.prices
if u.errors:
    st.caption("Skipped: " + "; ".join(e.split(":")[0] for e in u.errors))
st.caption(f"Source: **{u.source}** · {prices.shape[1]} assets · {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d} · "
           f"all prices converted to {base} at daily FX · costs {cost_bps:g} bps · execution lag {lag} day(s)")
month_ends = strategies.rebalance_dates(prices.index, "M")


def factor_scores(name: str, report_lag: int = 90) -> pd.DataFrame:
    if name == "Momentum (12-1)":
        return factors.momentum_scores(prices, month_ends)
    if name == "Low volatility":
        return factors.low_vol_scores(prices, month_ends)
    ni_b, ni, eq, sh, _ = load_fund(symbols, base)
    if name == "Value (earnings yield)":
        return factors.value_scores(prices, ni_b, sh, month_ends, report_lag)
    return factors.quality_scores(ni, eq, prices.columns, month_ends, report_lag)


def metrics_table(results: list) -> pd.DataFrame:
    df = pd.DataFrame({r.name: r.summary() for r in results}).T
    pct = ["CAGR", "Volatility", "Max drawdown", "Hit rate (months)", "Total return", "Annual turnover", "Total costs"]
    return df.style.format({**{c: "{:.1%}" for c in pct}, "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Calmar": "{:.2f}", "Trades": "{:.0f}"})


def equity_chart(results: list, title: str, height=380):
    fig = go.Figure()
    for r in results:
        fig.add_scatter(x=r.equity.index, y=r.equity * 100, name=r.name)
    fig.update_layout(height=height, title=title, yaxis_title=f"growth of 100 ({base})", margin=dict(l=0, r=0, t=40, b=0),
                      legend=dict(orientation="h", y=-0.15))
    return fig


bench = backtest.run(prices, strategies.buy_and_hold(prices), cost_bps, lag, name="Benchmark: equal-weight buy & hold")

t_sig, t_bt, t_sweep, t_arena = st.tabs(["🔬 Signal research", "📊 Backtest", "🗺️ Parameter sweep", "🏟️ Strategy arena"])

# ---------------------------------------------------------------- signal research
with t_sig:
    c1, c2, c3 = st.columns(3)
    fac = c1.selectbox("Factor", factors.FACTORS)
    rep_lag = c2.number_input("Reporting lag for fundamentals (days)", 0, 365, 90, 15,
                              help="Annual results only count as known this many days after the fiscal year end.",
                              disabled=fac in ("Momentum (12-1)", "Low volatility"))
    n_q = c3.selectbox("Groups", [3, 4, 5], index=2, help="Quintiles need at least 10 stocks with scores each month.")
    scores = factor_scores(fac, rep_lag)
    fwd = factors.forward_returns(prices, month_ends)
    ic = factors.information_coefficient(scores, fwd)
    if ic.empty:
        st.info("Not enough overlapping data for this factor yet.")
    else:
        s = factors.ic_summary(ic)
        m = st.columns(4)
        m[0].metric("Mean IC", f"{s['Mean IC']:+.3f}")
        m[1].metric("t-stat", f"{s['t-stat']:+.2f}", "significant" if abs(s["t-stat"]) >= 2 else "not significant", delta_color="off")
        m[2].metric("Months IC > 0", f"{s['Share of months IC > 0']:.0%}")
        m[3].metric("Months tested", f"{s['Months']}", f"{ic.index[0]:%Y-%m} to {ic.index[-1]:%Y-%m}", delta_color="off")
        g1, g2 = st.columns(2)
        with g1:
            fig = go.Figure()
            fig.add_bar(x=ic.index, y=ic.values, name="Monthly IC", marker_color=["seagreen" if v > 0 else "crimson" for v in ic])
            fig.add_scatter(x=ic.index, y=ic.rolling(12, min_periods=6).mean(), name="12-month average")
            fig.update_layout(height=320, title="Information coefficient by month", margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            qr = factors.quantile_returns(scores, fwd, n_q)
            if not qr.empty:
                ann = (1 + qr.mean()) ** 12 - 1
                fig = go.Figure(go.Bar(x=ann.index, y=ann.values * 100, marker_color="steelblue"))
                fig.update_layout(height=320, title=f"Annualised return by group (Q1 low → Q{n_q} high score)",
                                  yaxis_title="%", margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
        if not qr.empty:
            ls = (1 + (qr[f"Q{n_q}"] - qr["Q1"])).cumprod()
            fig = go.Figure(go.Scatter(x=ls.index, y=ls * 100, name="Top minus bottom group"))
            fig.update_layout(height=260, title=f"Top group minus bottom group, cumulative (growth of 100, {base}, before costs)",
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        st.caption("IC = rank correlation between this month's scores and next month's returns. Above ~0.03–0.05 with a "
                   "t-stat above 2 is usually considered meaningful.")
    with st.expander("How the factors and tests work"):
        st.markdown(factors.__doc__)

# ---------------------------------------------------------------- backtest
with t_bt:
    c1, c2 = st.columns([2, 3])
    choice = c1.selectbox("Strategy", list(strategies.STRATEGIES) + ["Factor: top group"])
    params = {}
    with c2:
        if choice == "Momentum rotation":
            a, b = st.columns(2)
            params = {"lookback": a.slider("Lookback (months)", 1, 24, 12), "top_n": b.slider("Hold top N", 1, min(20, prices.shape[1]), 8)}
        elif choice == "Trend following (SMA crossover)":
            a, b = st.columns(2)
            params = {"fast": a.slider("Fast SMA (days)", 5, 100, 50), "slow": b.slider("Slow SMA (days)", 50, 300, 200)}
        elif choice == "Inverse volatility":
            params = {"window": st.slider("Volatility window (days)", 20, 252, 63)}
        elif choice == "Factor: top group":
            a, b = st.columns(2)
            bt_fac = a.selectbox("Factor", factors.FACTORS, key="btfac")
            frac = b.slider("Hold top fraction", 0.1, 0.5, 0.2, 0.05)
    try:
        if choice == "Factor: top group":
            dec = strategies.from_scores(factor_scores(bt_fac), frac)
        else:
            dec = strategies.STRATEGIES[choice](prices, **params)
        res = backtest.run(prices, dec, cost_bps, lag, name=choice)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.plotly_chart(equity_chart([res, bench], f"{choice} vs benchmark"), use_container_width=True)
    st.dataframe(metrics_table([res, bench]), use_container_width=True)
    g1, g2 = st.columns(2)
    with g1:
        dd = res.equity / res.equity.cummax() - 1
        fig = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", line=dict(color="crimson", width=1)))
        fig.update_layout(height=260, title="Drawdown (%)", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        if not res.weights.empty:
            w = res.weights.loc[:, (res.weights > 0).any()]
            top = w.mean().nlargest(10).index
            w_plot = w[top].assign(Other=w.drop(columns=top).sum(axis=1))
            fig = px.area(w_plot, title="Weights after each rebalance (top 10 + other)")
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=40, b=0), showlegend=False, yaxis_title="weight")
            st.plotly_chart(fig, use_container_width=True)
    with st.expander("How the backtester works"):
        st.markdown(backtest.__doc__)

# ---------------------------------------------------------------- sweep
with t_sweep:
    c1, c2 = st.columns(2)
    which = c1.selectbox("Strategy to sweep", ["Momentum rotation", "Trend following (SMA crossover)"])
    metric = c2.selectbox("Score each combination by", ["Sharpe", "CAGR", "Max drawdown", "Calmar", "Sortino"])
    if which == "Momentum rotation":
        grid = {"lookback": [1, 3, 6, 9, 12, 18], "top_n": [3, 5, 8, 12, 20]}
        fn = strategies.momentum
    else:
        grid = {"fast": [10, 20, 50, 100], "slow": [100, 150, 200, 250]}
        fn = strategies.sma_crossover
    if st.button("Run sweep", type="primary"):
        with st.spinner(f"Running {len(grid[list(grid)[0]]) * len(grid[list(grid)[1]])} backtests…"):
            table = backtest.sweep(prices, fn, grid, metric, cost_bps)
        fmt = ".2f" if metric in ("Sharpe", "Calmar", "Sortino") else ".1%"
        fig = px.imshow(table.astype(float), text_auto=fmt, color_continuous_scale="RdYlGn", aspect="auto",
                        labels=dict(x=list(grid)[1], y=list(grid)[0], color=metric))
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Beware of picking the single best cell: with enough combinations one will look good by chance (overfitting). "
                   "Prefer settings whose neighbours also do well.")

# ---------------------------------------------------------------- arena
with t_arena:
    picks = st.multiselect("Strategies to compare (default settings)", list(strategies.STRATEGIES) + ["Factor: momentum top 20%", "Factor: value top 20%"],
                           default=["Equal weight, monthly rebalance", "Momentum rotation", "Inverse volatility", "Trend following (SMA crossover)"])
    results = [bench]
    for p in picks:
        try:
            if p == "Factor: momentum top 20%":
                dec = strategies.from_scores(factor_scores("Momentum (12-1)"))
            elif p == "Factor: value top 20%":
                dec = strategies.from_scores(factor_scores("Value (earnings yield)"))
            else:
                dec = strategies.STRATEGIES[p](prices)
            results.append(backtest.run(prices, dec, cost_bps, lag, name=p))
        except ValueError as exc:
            st.warning(f"{p}: {exc}")
    st.plotly_chart(equity_chart(results, "Head to head", 420), use_container_width=True)
    st.dataframe(metrics_table(results), use_container_width=True)
    st.caption("Strategies start trading only once they have enough history (e.g. 12 months for momentum), so early "
               "periods can be flat. Compare over the same window before drawing conclusions.")
