import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gterm import ui
from gterm.analysis import options as op
from gterm.analysis import strategies_opt as so
from gterm.data import registry
from gterm.data.base import DataSourceError

ui.page_setup("Derivatives", "🎯")
live_rf = registry.risk_free_usd_short()
NUMERIC = {"strike": float, "premium": float, "quantity": float, "days": float, "iv": float}
rf_default = round((live_rf[0] if live_rf else 0.042) * 100, 2)
t_price, t_strat = st.tabs(["🧮 Option pricer", "🧩 Strategy builder"])

# ---------------------------------------------------------------- pricer
with t_price:
    c = st.columns(7)
    S = c[0].number_input("Share price", 0.01, 1e7, 100.0)
    K = c[1].number_input("Strike", 0.01, 1e7, 100.0)
    days = c[2].number_input("Days to expiry", 1, 3650, 30)
    vol = c[3].number_input("Volatility (%)", 1.0, 300.0, 25.0) / 100
    r = c[4].number_input("Risk-free (%)", 0.0, 30.0, rf_default, key="pr_r") / 100
    q = c[5].number_input("Dividend yield (%)", 0.0, 20.0, 0.0) / 100
    kind = c[6].radio("Type", ["call", "put"])
    st.caption(f"Risk-free default: {live_rf[1] if live_rf else 'placeholder 4.2%'} · European exercise, Black-Scholes-Merton")
    T = days / 365
    price = op.bs_price(S, K, T, r, vol, kind, q)
    g = op.greeks(S, K, T, r, vol, kind, q)
    m = st.columns(6)
    m[0].metric(f"{kind.title()} price", f"{price:,.4f}")
    m[1].metric("Delta", f"{g['delta']:.4f}", help="Option price change for a 1-unit move in the share price.")
    m[2].metric("Gamma", f"{g['gamma']:.5f}", help="Change in delta for a 1-unit move in the share price.")
    m[3].metric("Vega (per vol pt)", f"{g['vega']:.4f}", help="Price change for a 1 percentage-point rise in volatility.")
    m[4].metric("Theta (per day)", f"{g['theta']:.4f}", help="Price change from one calendar day passing.")
    m[5].metric("Rho (per 1%)", f"{g['rho']:.4f}", help="Price change for a 1 percentage-point rise in rates.")
    intrinsic = max(S - K, 0) if kind == "call" else max(K - S, 0)
    st.caption(f"Intrinsic value {intrinsic:,.4f} + time value {price - intrinsic:,.4f}")
    grid = np.linspace(S * 0.5, S * 1.5, 201)
    fig = go.Figure()
    for d in sorted({days, max(days // 2, 1), 1}, reverse=True):
        fig.add_scatter(x=grid, y=[op.bs_price(x, K, d / 365, r, vol, kind, q) for x in grid], name=f"{d} days left")
    fig.add_scatter(x=grid, y=np.maximum(grid - K, 0) if kind == "call" else np.maximum(K - grid, 0), name="at expiry", line=dict(dash="dot"))
    fig.add_vline(x=S, line=dict(dash="dash", color="gray"))
    fig.update_layout(height=360, xaxis_title="share price", yaxis_title="option value", margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- strategy builder
with t_strat:
    c = st.columns([2, 2, 1, 1, 1])
    sym = c[0].text_input("Underlying (optional)", "SPY", help="Fills the share price; US names can also fill premiums from the live option chain.").strip().upper()
    spot = None
    if sym:
        try:
            qd = registry.quote(sym)
            spot = qd.price
            c[0].caption(f"{qd.name}: {qd.price:,.2f} {qd.currency} [{qd.source}, {qd.price_date:%Y-%m-%d}]")
        except DataSourceError as exc:
            c[0].caption(str(exc))
    spot = c[1].number_input("Share price", 0.01, 1e7, float(round(spot, 2)) if spot else 100.0, key="st_spot")
    days = c[2].number_input("Days", 1, 3650, 30, key="st_days")
    iv = c[3].number_input("IV (%)", 1.0, 300.0, 20.0, key="st_iv") / 100
    r = c[4].number_input("Rate (%)", 0.0, 30.0, rf_default, key="st_r") / 100
    p1, p2 = st.columns([3, 2])
    name = p1.selectbox("Start from a preset", so.PRESETS, index=4)
    if p2.button("Load preset", use_container_width=True) or "legs" not in st.session_state or st.session_state.get("legs_key") != (name, spot, days, iv):
        st.session_state["legs"] = pd.DataFrame([vars(l) for l in so.preset(name, spot, days, iv, r)]).astype(NUMERIC)
        st.session_state["legs_key"] = (name, spot, days, iv)
    if sym and st.button("Use live option prices for these legs", help="Nearest listed expiry and strike; premium = bid/ask mid; IV computed from it."):
        try:
            chain, cspot, _ = registry.options_chain(sym, 20)
            target = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(days=days)
            exps = pd.to_datetime(chain["expiry"]).unique()
            exp = min(exps, key=lambda e: abs(e - target))
            sub = chain[pd.to_datetime(chain["expiry"]) == exp]
            legs_df = st.session_state["legs"].copy().astype(NUMERIC)  # decimals (e.g. 29.6 days) need float columns
            dleft = max((exp - pd.Timestamp.utcnow().tz_localize(None)).total_seconds() / 86400, 0.5)
            for i, row in legs_df.iterrows():
                if row["kind"] == "stock":
                    legs_df.at[i, "premium"] = round(cspot, 2)
                    continue
                opts = sub[sub["type"] == row["kind"]]
                k = opts.iloc[(opts["strike"] - row["strike"]).abs().argsort()[:1]]
                if k.empty:
                    continue
                k = k.iloc[0]
                mid = (k["bid"] + k["ask"]) / 2 if k["bid"] > 0 and k["ask"] > 0 else k["last"]
                legs_df.at[i, "strike"], legs_df.at[i, "premium"], legs_df.at[i, "days"] = float(k["strike"]), round(float(mid), 2), round(dleft, 1)
                ivc = op.implied_vol(float(mid), cspot, float(k["strike"]), dleft / 365, r, row["kind"])
                if not np.isnan(ivc):
                    legs_df.at[i, "iv"] = round(ivc, 4)
            st.session_state["legs"] = legs_df
            st.success(f"Filled from the {sym} chain, expiry {exp:%Y-%m-%d} [{registry.active_source_name()}]")
        except (DataSourceError, ValueError, KeyError, TypeError) as exc:
            st.error(f"Could not use live prices: {exc}")
    st.caption("Edit the legs freely: premium per share (paid if long, received if short; entry price for stock), quantity in contracts "
               "of 100 shares, IV for the 'today' curve and Greeks. Add or delete rows.")
    edited = st.data_editor(st.session_state["legs"], num_rows="dynamic", use_container_width=True, key="legs_editor",
                            column_config={"kind": st.column_config.SelectboxColumn("Kind", options=["call", "put", "stock"], required=True),
                                           "side": st.column_config.SelectboxColumn("Side", options=["long", "short"], required=True),
                                           "iv": st.column_config.NumberColumn("IV", format="%.3f")})
    legs = [so.Leg(**{k: row[k] for k in ["kind", "side", "strike", "premium", "quantity", "days", "iv"]})
            for _, row in edited.dropna(subset=["kind", "side"]).iterrows()]
    if not legs:
        st.info("Add at least one leg.")
        st.stop()
    s = so.price_grid(spot, legs)
    pnl = so.payoff_at_expiry(legs, s)
    today = so.value_today(legs, s, r)
    profit, loss = so.max_profit_loss(s, pnl)
    be = so.breakevens(s, pnl)
    g = so.net_greeks(legs, spot, r)
    prem = so.net_premium(legs)
    m = st.columns(5)
    m[0].metric("Net premium", f"{abs(prem):,.2f}", "paid (debit)" if prem > 0 else "received (credit)", delta_color="off")
    m[1].metric("Max profit", "unlimited" if profit is None else f"{profit:,.2f}")
    m[2].metric("Max loss", "unlimited" if loss is None else f"{loss:,.2f}")
    m[3].metric("Breakeven(s)", ", ".join(f"{b:,.2f}" for b in be) or "none")
    m[4].metric("P&L now at spot", f"{so.value_today(legs, [spot], r)[0]:,.2f}")
    st.caption(f"Net Greeks: delta {g['delta']:,.1f} · gamma {g['gamma']:,.3f} · vega {g['vega']:,.2f} per vol pt · "
               f"theta {g['theta']:,.2f}/day · rho {g['rho']:,.2f} per 1%  (whole position, in currency units)")
    fig = go.Figure()
    fig.add_scatter(x=s, y=pnl, name="P&L at expiry", line=dict(width=2.5))
    fig.add_scatter(x=s, y=today, name=f"P&L today (IV per leg, {days} days)", line=dict(dash="dot"))
    fig.add_hline(y=0, line=dict(color="gray", width=1))
    fig.add_vline(x=spot, line=dict(dash="dash", color="gray"), annotation_text="spot")
    for b in be:
        fig.add_vline(x=b, line=dict(dash="dot", color="orange", width=1))
    fig.update_layout(height=420, xaxis_title="share price at expiry", yaxis_title="profit / loss", margin=dict(l=0, r=0, t=20, b=0),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("How this works"):
        st.markdown(so.__doc__)
