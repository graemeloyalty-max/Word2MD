import numpy as np
import pandas as pd
import streamlit as st

from gterm import ui
from gterm.analysis import ma
from gterm.core import fx
from gterm.core.statements import latest
from gterm.data import registry
from gterm.data.base import DataSourceError

ui.page_setup("M&A Analytics", "🤝")

c1, c2 = st.columns(2)
acq_sym = c1.text_input("Acquirer", "MSFT").strip().upper()
tgt_sym = c2.text_input("Target", "CRM").strip().upper()


def load(sym):
    f = registry.fundamentals(sym)
    inc, bal = f.statements["income"], f.statements["balance"]
    return f, {"ni": latest(inc, "net_income"), "ebitda": latest(inc, "ebitda"), "rev": latest(inc, "revenue"),
               "net_debt": latest(bal, "net_debt", 0.0), "tax": latest(inc, "tax_rate")}


try:
    fa, a = load(acq_sym)
    ft, t = load(tgt_sym)
    ccy = fa.financial_currency
    a_px = fa.price * fx.latest_rate(fa.price_currency, ccy)[0]
    t_rate, t_note = fx.latest_rate(ft.financial_currency, ccy)
    t_px = ft.price * fx.latest_rate(ft.price_currency, ccy)[0]
except (DataSourceError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"{fa.name} ({fa.price_currency} {fa.price:,.2f}) acquiring {ft.name} ({ft.price_currency} {ft.price:,.2f}). "
           f"Everything in **{ccy}** (acquirer's statement currency); target converted at {t_note}. "
           f"Source: {fa.source}, latest fiscal year, retrieved {fa.as_of:%Y-%m-%d}.")
if "Financial Services" in (fa.sector, ft.sector):
    st.warning("Banks/insurers: earnings-based accretion is fine, but EV/EBITDA multiples are not meaningful for them.")

st.subheader("Deal terms")
k = f"ma_{acq_sym}_{tgt_sym}"
r1 = st.columns(4)
premium = r1[0].slider("Premium to target price (%)", 0, 100, 30, key=k + "p") / 100
pct_stock = r1[1].slider("Paid in acquirer shares (%)", 0, 100, 50, key=k + "s") / 100
kd = r1[2].number_input("Pre-tax cost of debt for cash part (%)", 0.0, 20.0, 5.5, 0.25, key=k + "kd") / 100
tax_d = a["tax"] if not np.isnan(a["tax"]) else 0.21
tax = r1[3].number_input("Tax rate (%)", 0.0, 50.0, round(tax_d * 100, 1), key=k + "t") / 100
r2 = st.columns(4)
syn = r2[0].number_input(f"Annual pre-tax synergies ({ccy} m)", 0.0, 1e7, 0.0, 50.0, key=k + "syn") * 1e6
a_ni = r2[1].number_input(f"Acquirer net income ({ccy} m)", -1e7, 1e7, round(a["ni"] / 1e6, 1), key=k + "ani") * 1e6
t_ni = r2[2].number_input(f"Target net income ({ccy} m)", -1e7, 1e7, round(t["ni"] * t_rate / 1e6, 1), key=k + "tni") * 1e6
r2[3].caption("Net incomes default to the latest reported fiscal year. Use forward estimates if you have them.")

d = ma.DealInputs(acq_price=a_px, acq_shares=fa.shares, acq_net_income=a_ni, tgt_price=t_px, tgt_shares=ft.shares, tgt_net_income=t_ni,
                  premium=premium, pct_stock=pct_stock, cost_of_debt=kd, tax_rate=tax, synergies=syn,
                  tgt_net_debt=t["net_debt"] * t_rate, tgt_ebitda=t["ebitda"] * t_rate, tgt_revenue=t["rev"] * t_rate)
try:
    r = ma.analyse(d)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Result")
m = st.columns(5)
m[0].metric("Accretion / dilution", f"{r.accretion:+.1%}", "accretive" if r.accretion > 0 else "dilutive",
            delta_color="normal" if r.accretion > 0 else "inverse")
m[1].metric("Pro forma EPS", f"{r.pro_forma_eps:,.2f}", f"stand-alone {r.acq_eps:,.2f}", delta_color="off")
m[2].metric(f"Purchase price ({ccy} bn)", f"{r.purchase_equity / 1e9:,.1f}", f"offer {r.offer_price:,.2f}/share", delta_color="off")
m[3].metric("Breakeven synergies (pre-tax)", f"{r.breakeven_synergies / 1e6:,.0f} m",
            "accretive without synergies" if r.breakeven_synergies <= 0 else f"{r.breakeven_synergies / a_ni:.1%} of acquirer NI",
            delta_color="off")
m[4].metric("Target holders own", f"{r.ownership_target_holders:.1%}", "of combined company", delta_color="off")
st.caption(f"New shares issued: {r.new_shares / 1e6:,.1f} m · cash paid: {r.cash_part / 1e9:,.1f} bn {ccy} · after-tax financing cost "
           f"{r.after_tax_financing_cost / 1e6:,.0f} m a year · offer P/E {r.offer_pe:,.1f}x · offer EV/EBITDA {r.offer_ev_ebitda:,.1f}x · "
           f"offer EV/Sales {r.offer_ev_sales:,.2f}x")

g1, g2 = st.columns([3, 2])
with g1:
    st.markdown("**Accretion by premium and stock mix**")
    grid = ma.sensitivity(d)
    st.dataframe(grid.style.format("{:+.1%}").background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)
with g2:
    st.markdown("**Precedent deals (enter your own)**")
    st.caption("A free source of precedent transactions doesn't exist. Add deals from press releases or filings to compare multiples.")
    comps = st.data_editor(pd.DataFrame({"Deal": pd.Series(dtype=str), "Year": pd.Series(dtype="Int64"), "EV/EBITDA": pd.Series(dtype=float),
                                         "EV/Sales": pd.Series(dtype=float), "Premium %": pd.Series(dtype=float)}),
                           num_rows="dynamic", use_container_width=True, key=k + "comps")
if len(comps.dropna(how="all")):
    imp = ma.implied_from_comps(comps, d.tgt_ebitda, d.tgt_revenue, d.tgt_net_debt, ft.shares)
    st.dataframe(imp.style.format({"Median": "{:.2f}", "Implied EV": lambda v: f"{v / 1e9:,.1f} bn" if pd.notna(v) else "–",
                                   "Implied price / share": "{:,.2f}"}, na_rep="–"), use_container_width=True, hide_index=True)
    st.caption(f"Compare: offer price {r.offer_price:,.2f} {ccy} per share at EV/EBITDA {r.offer_ev_ebitda:,.1f}x.")
with st.expander("How the model works"):
    st.markdown(ma.__doc__)
