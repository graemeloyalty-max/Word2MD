import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gterm import ui
from gterm.analysis import country as ctry
from gterm.analysis import dcf, multiples, quality
from gterm.core import config, fx, indicators
from gterm.core.statements import LABELS, latest, series
from gterm.data import registry
from gterm.data.base import DataSourceError

ui.page_setup("Equity Research", "🔬")
settings = config.load_settings()
base_ccy = settings["base_currency"]

c1, c2 = st.columns([3, 2])
symbol = c1.text_input("Ticker (US or European)", value=st.query_params.get("symbol", "SAP.DE"),
                       help="e.g. AAPL, MSFT, SAP.DE, ASML.AS, MC.PA, NESN.SW, SHEL.L").strip().upper()
use_edgar = c2.toggle("Extend US history with SEC EDGAR", value=False,
                      help="Adds 10-K figures older than yfinance's ~4 years. US filers only. Needs your SEC User-Agent in Settings.")
if not symbol:
    st.stop()

try:
    f = registry.fundamentals(symbol, use_edgar=use_edgar)
except DataSourceError as exc:
    if use_edgar:
        st.warning(f"SEC EDGAR: {exc} Showing yfinance history only.")
        f = registry.fundamentals(symbol)
    else:
        st.error(str(exc))
        st.stop()

inc, bal, cf = f.statements["income"], f.statements["balance"], f.statements["cashflow"]
fin = f.financial_currency
try:
    price_to_fin, price_fx_note = fx.latest_rate(f.price_currency, fin)
    fin_to_price, _ = fx.latest_rate(fin, f.price_currency)
except (DataSourceError, ValueError) as exc:
    st.error(f"FX conversion unavailable: {exc}")
    st.stop()
mcap_fin = f.price * price_to_fin * f.shares

st.subheader(f"{f.name} · {f.symbol}")
st.caption(f"{f.sector} · {f.industry} · {f.country} · statements in **{fin}**, shares trade in **{f.price_currency}**")
st.caption(ui.provenance(f.source + (f" {f.history_source}" if f.history_source else ""), f.as_of)
           + f" · fiscal years {', '.join(str(c.year) for c in inc.columns)}")
summary = st.container(border=True)  # filled in after the tabs compute their results
results = {}

t_stat, t_dcf, t_mult, t_qual, t_ctry = st.tabs(["📑 Statements", "💰 DCF", "⚖️ Multiples", "🏅 Quality", "🌍 Country & currency"])

# ---------------------------------------------------------------- statements
with t_stat:
    unit = st.radio("Units", ["millions", "billions"], horizontal=True, key="unit")
    div = 1e6 if unit == "millions" else 1e9
    per_share_lines = {"diluted_eps", "operating_margin", "tax_rate"}
    count_lines = {"diluted_shares", "shares_outstanding"}
    for title, df in [("Income statement", inc), ("Balance sheet", bal), ("Cash flow", cf)]:
        st.markdown(f"**{title}** ({fin} {unit})")
        show = df.copy()
        for line in show.index:
            if line in {"operating_margin", "tax_rate"}:
                continue
            if line not in per_share_lines:
                show.loc[line] = show.loc[line] / (div if line not in count_lines else 1e6)
        show.index = [LABELS.get(i, i) + (" (m)" if i in count_lines else "") for i in show.index]
        show.columns = [c.strftime("%Y-%m-%d") for c in show.columns]
        fmt = show.copy().astype(object)
        for idx in show.index:
            pct = idx in (LABELS["operating_margin"], LABELS["tax_rate"])
            fmt.loc[idx] = [("–" if pd.isna(v) else (f"{v:.1%}" if pct else f"{v:,.2f}")) for v in show.loc[idx]]
        st.dataframe(fmt, use_container_width=True)
    with st.expander("How statements were normalised across accounting standards"):
        st.markdown("Every model reads the same standard lines, whatever the company's accounting standard (US GAAP or IFRS) "
                    "or source labels. Adjustments applied to **this** company:")
        for n in f.adjustments or ["None needed."]:
            st.markdown(f"- {n}")
        st.caption("Full rules: gterm/core/statements.py and the README.")

# ---------------------------------------------------------------- DCF
with t_dcf:
    fcf_s = series(cf, "free_cash_flow")
    rev = series(inc, "revenue")
    cagr = (rev.iloc[-1] / rev.iloc[0]) ** (1 / (len(rev) - 1)) - 1 if len(rev) > 1 and rev.iloc[0] > 0 else 0.05
    tax_d = latest(inc, "tax_rate")
    tax_d = 0.21 if np.isnan(tax_d) else tax_d
    debt = latest(bal, "total_debt", 0.0)
    cash = latest(bal, "cash", 0.0)
    minority = latest(bal, "minority_interest", 0.0)
    minority = 0.0 if np.isnan(minority) else minority
    int_exp = abs(latest(inc, "interest_expense", np.nan))
    live_rf = registry.risk_free_usd() if fin == "USD" else None
    rf_d, rf_src = (live_rf if live_rf else (ctry.risk_free_default(fin), ctry.RF_SOURCE + f" ({fin})"))
    kd_d = float(np.clip(int_exp / debt, 0.02, 0.12)) if debt > 0 and not np.isnan(int_exp) else rf_d + 0.015
    crp_d = ctry.crp_for(f.country)
    raw_beta = float(f.beta) if f.beta is not None else None
    beta_d = raw_beta if raw_beta is not None and raw_beta >= 0.3 else 1.0
    is_financial = f.sector == "Financial Services"

    if is_financial:
        st.info("**Banks and insurers:** a free-cash-flow DCF does not fit financial companies, because debt and deposits are their raw "
                "material, not financing. The result below is shown for reference only and is left out of the summary. "
                "Use P/B and P/E in the Multiples tab (Phase 2 may later add a dividend-discount or excess-return model).")
    st.markdown("**Assumptions**: every one is editable. Defaults come from the statements, live data, or the labelled placeholders.")
    k = f"dcf_{symbol}"  # widget keys per symbol so they reset when you change company
    a1, a2, a3, a4 = st.columns(4)
    base_choice = a1.radio("Base FCF", ["Latest year", "3-year average"], key=k + "base")
    base_fcf = fcf_s.iloc[-1] if base_choice == "Latest year" or len(fcf_s) < 3 else fcf_s.iloc[-3:].mean()
    a1.caption(f"= {base_fcf / 1e6:,.0f}m {fin}")
    g_high = a2.number_input("Growth, years 1–5 (%)", -20.0, 50.0, round(float(np.clip(cagr, -0.05, 0.20)) * 100, 1), 0.5, key=k + "g") / 100
    a2.caption(f"default = revenue CAGR {cagr:.1%}, capped −5%…20%")
    years_total = a3.slider("Projection years", 6, 15, 10, key=k + "yrs")
    g_term = a4.number_input("Terminal growth (%)", -2.0, 6.0, 2.5, 0.25, key=k + "gt") / 100
    a4.caption("keep at or below long-run nominal GDP growth")

    b1, b2, b3, b4, b5 = st.columns(5)
    rf = b1.number_input(f"Risk-free, {fin} (%)", 0.0, 20.0, round(rf_d * 100, 2), 0.05, key=k + "rf") / 100
    b1.caption(rf_src)
    beta = b2.number_input("Beta", -1.0, 3.0, round(beta_d, 2), 0.05, key=k + "beta")
    if raw_beta is None:
        b2.caption("beta unavailable: default 1.0")
    elif raw_beta < 0.3:
        b2.caption(f"{f.source} reports {raw_beta:.2f}, implausibly low for a listed company, so the default is 1.0. Edit if you disagree.")
    else:
        b2.caption(f"{f.source} 5y monthly beta")
    erp = b3.number_input("Equity risk premium (%)", 0.0, 12.0, ctry.MATURE_ERP * 100, 0.25, key=k + "erp") / 100
    b3.caption(ctry.ERP_SOURCE)
    crp = b4.number_input(f"Country risk premium, {f.country} (%)", 0.0, 15.0, crp_d * 100, 0.1, key=k + "crp") / 100
    b4.caption(ctry.CRP_SOURCE)
    lam = b5.number_input("Country exposure λ", 0.0, 2.0, 1.0, 0.1, key=k + "lam")
    b5.caption("1 = all revenue from that country")
    c1_, c2_, c3_ = st.columns(3)
    kd = c1_.number_input("Pre-tax cost of debt (%)", 0.0, 20.0, round(kd_d * 100, 2), 0.1, key=k + "kd") / 100
    c1_.caption("default = interest expense / total debt" if debt > 0 and not np.isnan(int_exp) else "default = risk-free + 1.5%")
    tax = c2_.number_input("Tax rate (%)", 0.0, 50.0, round(tax_d * 100, 1), 0.5, key=k + "tax") / 100
    c2_.caption("default = latest effective tax rate")
    shares = c3_.number_input("Shares (m)", 0.0, 1e6, round(f.shares / 1e6, 2), key=k + "sh") * 1e6

    inputs = dcf.DCFInputs(base_fcf=float(base_fcf), growth_high=g_high, years_total=years_total, terminal_growth=g_term,
                           risk_free=rf, beta=beta, equity_risk_premium=erp, country_risk_premium=crp, crp_exposure=lam,
                           pre_tax_cost_of_debt=kd, tax_rate=tax, equity_value_market=mcap_fin, debt=debt, cash=cash,
                           minority_interest=minority, shares=shares)
    try:
        r = dcf.run(inputs)
    except ValueError as exc:
        st.error(str(exc))
        r = None
    if r:
        vps_price = r.value_per_share * fin_to_price
        upside = vps_price / f.price - 1
        if not is_financial and base_fcf > 0:
            results["dcf"] = (vps_price, upside)
        m = st.columns(5)
        m[0].metric("Cost of equity", f"{r.cost_of_equity:.2%}")
        m[1].metric("WACC", f"{r.wacc:.2%}", f"equity weight {r.weight_equity:.0%}", delta_color="off")
        m[2].metric(f"Enterprise value ({fin} bn)", f"{r.enterprise_value / 1e9:,.1f}")
        m[3].metric(f"Value / share ({f.price_currency})", f"{vps_price:,.2f}", f"{upside:+.1%} vs price {f.price:,.2f}")
        m[4].metric("Terminal value share", f"{r.terminal_share:.0%}")
        for w in r.warnings:
            st.warning(w)
        st.caption(f"Cost of equity = {rf:.2%} + {beta:.2f} × {erp:.2%} + {lam:.1f} × {crp:.2%}. "
                   f"Equity value = EV {r.enterprise_value / 1e9:,.1f}bn − debt {debt / 1e9:,.1f}bn + cash {cash / 1e9:,.1f}bn"
                   f" − minorities {minority / 1e9:,.1f}bn = {r.equity_value / 1e9:,.1f}bn {fin}. FX: {price_fx_note}.")
        g1, g2 = st.columns([3, 2])
        with g1:
            p = r.projections
            fig = go.Figure()
            fig.add_bar(x=p.year, y=p.fcf / 1e9, name="Projected FCF")
            fig.add_bar(x=p.year, y=p.pv_fcf / 1e9, name="Present value")
            fig.add_scatter(x=p.year, y=p.growth * 100, name="Growth %", yaxis="y2", mode="lines+markers")
            hist_x = [c.year - fcf_s.index[-1].year for c in fcf_s.index]
            fig.add_bar(x=hist_x, y=fcf_s.values / 1e9, name="Historical FCF", marker_color="gray")
            fig.update_layout(height=360, barmode="group", margin=dict(l=0, r=0, t=40, b=40), legend=dict(orientation="h", y=1.12),
                              yaxis_title=f"{fin} bn", yaxis2=dict(overlaying="y", side="right", title="growth %"),
                              xaxis_title="year (0 = latest reported)")
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            st.markdown(f"**Sensitivity: value per share ({f.price_currency})**")
            sens = dcf.sensitivity(inputs) * fin_to_price
            st.dataframe(sens.style.format("{:,.2f}").background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)
            ig = dcf.implied_growth(inputs, f.price * price_to_fin)
            if ig is not None:
                st.info(f"**Reverse DCF:** the current price implies {ig:.1%} a year FCF growth for years 1–{inputs.years_high}, "
                        "with everything else unchanged.")

# ---------------------------------------------------------------- multiples
with t_mult:
    region = multiples.region_of(f.country)
    defaults = multiples.default_peers(symbol, f.sector, f.country)
    peer_txt = st.text_input(f"Peer group ({f.sector}, {region}): edit freely, comma-separated",
                             ", ".join(defaults), key=f"peers_{symbol}")
    peers_syms = [p.strip().upper() for p in peer_txt.split(",") if p.strip()]
    try:
        target_ks = registry.key_stats(symbol)
    except DataSourceError as exc:
        target_ks = None
        st.error(str(exc))
    peers, failed = [], []
    if target_ks and peers_syms:
        with st.spinner(f"Loading {len(peers_syms)} peers…"):
            for p in peers_syms:
                try:
                    peers.append(registry.key_stats(p))
                except DataSourceError:
                    failed.append(p)
    if failed:
        st.caption(f"No data for: {', '.join(failed)}")
    if target_ks and peers:
        tbl = multiples.peer_table(target_ks, peers)
        st.dataframe(tbl.style.format({"Mkt cap": lambda v: f"{v / 1e9:,.1f}bn" if pd.notna(v) else "–",
                                       **{m: "{:.1f}x" for m in multiples.MULTIPLES.values()}}, na_rep="–")
                     .apply(lambda r: ["font-weight: bold" if r.name == 0 else "" for _ in r], axis=1),
                     use_container_width=True, hide_index=True)
        imp = multiples.implied_prices(target_ks, peers, f.price)
        st.markdown(f"**Implied share price at peer median ({f.price_currency})**")
        st.dataframe(imp.style.format({"Company": "{:.1f}x", "Peer median": "{:.1f}x", "Implied price": "{:,.2f}",
                                       "Premium / (discount) to peers": "{:+.0%}"}, na_rep="–"),
                     use_container_width=True, hide_index=True)
        med_implied = imp["Implied price"].median()
        if pd.notna(med_implied):
            results["multiples"] = (med_implied, med_implied / f.price - 1)
            st.caption(f"Median of the implied prices: **{med_implied:,.2f} {f.price_currency}** ({med_implied / f.price - 1:+.1%}).")
        st.caption(ui.provenance(target_ks.source, target_ks.as_of) + " · trailing multiples as reported by the source")

# ---------------------------------------------------------------- quality
with t_qual:
    q = quality.compute(f.statements, f.sector)
    results["quality"] = np.nan if is_financial else q.overall
    cols = st.columns(5)
    cols[0].metric("Quality score", "–" if np.isnan(q.overall) else f"{q.overall:.0f} / 100")
    fmt_val = {"roic": "{:.1%}", "margin_stability": "{:.1%}", "leverage": "{:.2f}x", "cash_conversion": "{:.0%}"}
    for col, (mkey, name) in zip(cols[1:], quality.NAMES.items()):
        v, s = q.values[mkey], q.scores[mkey]
        col.metric(name, "–" if np.isnan(v) else fmt_val[mkey].format(v), "not scored" if np.isnan(s) else f"score {s:.0f}",
                   delta_color="off")
    for n in q.notes:
        st.warning(n)
    g1, g2 = st.columns(2)
    with g1:
        om = series(inc, "operating_margin")
        fig = go.Figure(go.Bar(x=[c.year for c in om.index], y=om.values * 100, name="Operating margin"))
        fig.update_layout(height=290, title="Operating margin (%)", margin=dict(l=0, r=0, t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        fcf_s2, ni_s = series(cf, "free_cash_flow"), series(inc, "net_income")
        fig = go.Figure()
        fig.add_bar(x=[c.year for c in ni_s.index], y=ni_s.values / 1e9, name="Net income")
        fig.add_bar(x=[c.year for c in fcf_s2.index], y=fcf_s2.values / 1e9, name="Free cash flow")
        fig.update_layout(height=280, title=f"Profit vs cash ({fin} bn)", margin=dict(l=0, r=0, t=40, b=30), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("How the quality score works"):
        st.markdown(quality.__doc__)

# ---------------------------------------------------------------- country & currency
with t_ctry:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Country risk: {f.country}**")
        st.write(f"Country risk premium used in the DCF: **{crp:.2%}** × exposure λ {lam:.1f} = **{crp * lam:.2%}** added to cost of equity.")
        st.caption(ctry.CRP_SOURCE)
        table = pd.DataFrame([{"Country": c, "CRP": v[0], "Currency": v[1]} for c, v in ctry.COUNTRIES.items()]).sort_values("CRP")
        st.dataframe(table.style.format({"CRP": "{:.2%}"}), use_container_width=True, hide_index=True, height=300)
    with c2:
        st.markdown(f"**Share price return: {f.price_currency} vs your base currency ({base_ccy})**")
        try:
            h = registry.prices(symbol, "5y", "1d")
            native = h.frame["Close"]
            in_base, note = fx.convert_series(native, h.currency, base_ccy)
            rows = []
            for label, days in [("1 year", 365), ("3 years", 3 * 365), ("5 years", 5 * 365)]:
                start = native.index[-1] - pd.Timedelta(days=days)
                n, b = native[native.index >= start], in_base[in_base.index >= start]
                rn, rb = indicators.total_return(n), indicators.total_return(b)
                rows.append({"Period": label, f"Return in {h.currency}": rn, f"Return in {base_ccy}": rb,
                             "Currency effect": (1 + rb) / (1 + rn) - 1})
            st.dataframe(pd.DataFrame(rows).style.format({c: "{:+.1%}" for c in rows[0] if c != "Period"}),
                         use_container_width=True, hide_index=True)
            fig = go.Figure()
            fig.add_scatter(x=native.index, y=native / native.iloc[0] * 100, name=f"in {h.currency}")
            fig.add_scatter(x=in_base.index, y=in_base / in_base.iloc[0] * 100, name=f"in {base_ccy}")
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"), yaxis_title="rebased to 100")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(ui.provenance(h.source, h.as_of, h.last_date, h.from_cache) + f" · {note} · price return, dividends excluded")
        except (DataSourceError, ValueError) as exc:
            st.error(f"Could not compute currency returns: {exc}")

# ---------------------------------------------------------------- summary (top of page)
with summary:
    s = st.columns(4)
    s[0].metric(f"Price ({f.price_currency})", f"{f.price:,.2f}")
    if "dcf" in results:
        v, u = results["dcf"]
        s[1].metric("DCF value / share", f"{v:,.2f}", f"{u:+.1%}")
    else:
        why = "n/a for banks/insurers" if is_financial else "n/a: free cash flow ≤ 0" if "dcf" not in results else ""
        s[1].metric("DCF value / share", "–", why, delta_color="off")
    if "multiples" in results:
        v, u = results["multiples"]
        s[2].metric("Peer-multiples value", f"{v:,.2f}", f"{u:+.1%}", help="Median of the prices implied by P/E, EV/EBITDA, P/B and EV/Sales at the peer median. See the Multiples tab.")
    else:
        s[2].metric("Peer-multiples value", "–")
    qv = results.get("quality", np.nan)
    s[3].metric("Quality score", "–" if np.isnan(qv) else f"{qv:.0f} / 100", "n/a for banks/insurers" if is_financial else None, delta_color="off")
    if "dcf" in results and f.price_currency != base_ccy:
        rate, rate_note = fx.latest_rate(f.price_currency, base_ccy)
        st.caption(f"DCF value in your base currency: **{results['dcf'][0] * rate:,.2f} {base_ccy}** · FX {rate_note}")
    st.caption("All values follow the assumptions in the tabs below. Change any input and these update.")
