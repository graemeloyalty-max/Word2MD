"""Shared UI helpers: provenance labels, number formatting, sidebar status."""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from .core import config
from .data import registry


def provenance(source: str, as_of: datetime, price_date=None, cached: bool = False) -> str:
    """The 'Source · as of' caption shown under every number (ground rule 2)."""
    as_of_txt = as_of.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [f"Source: **{source}**"]
    if price_date is not None:
        parts.append(f"price date {price_date:%Y-%m-%d}")
    parts.append(f"retrieved {as_of_txt}" + (" (cached)" if cached else ""))
    return " · ".join(parts)


def fmt_price(x: float, ccy: str = "") -> str:
    if x is None:
        return "–"
    digits = 4 if abs(x) < 10 else 2
    return f"{x:,.{digits}f} {ccy}".strip()


def page_setup(title: str, icon: str = "📈"):
    st.set_page_config(page_title=f"{title} · Global Equity Terminal", page_icon=icon, layout="wide")
    sidebar_status()
    st.title(f"{icon} {title}")


def sidebar_status():
    with st.sidebar:
        src = registry.active_source_name()
        settings = config.load_settings()
        st.markdown("### Global Equity Terminal")
        if src == "demo":
            st.warning("Data source: **demo (synthetic)**. Numbers are NOT real market data.", icon="⚠️")
        else:
            st.success(f"Data source: **{src}**", icon="✅")
        st.caption(f"Base currency: **{settings['base_currency']}** · change in Settings")
