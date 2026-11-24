import streamlit as st

from gterm import ui
from gterm.core import cache, config
from gterm.data import registry

ui.page_setup("Settings", "⚙️")
settings = config.load_settings()

st.subheader("Preferences")
c1, c2 = st.columns(2)
base = c1.selectbox(
    "Base currency", config.BASE_CURRENCIES, index=config.BASE_CURRENCIES.index(settings["base_currency"])
)
source_options = ["auto", "yfinance", "demo"]
source = c2.selectbox(
    "Market data source",
    source_options,
    index=source_options.index(settings["data_source"]),
    help="auto = yfinance when reachable, otherwise the offline demo source.",
)
sec_ua = st.text_input(
    "SEC EDGAR User-Agent (for longer US statement history)",
    settings.get("sec_user_agent", ""),
    placeholder="Your Name your.email@example.com",
    help="The SEC requires every request to name you and give a contact email. Stored only in data/settings.json on this machine.",
)
if st.button("Save preferences", type="primary"):
    config.save_settings({"base_currency": base, "data_source": source, "sec_user_agent": sec_ua.strip()})
    registry._yfinance_reachable.cache_clear()
    st.success("Saved.")
    st.rerun()
st.caption(f"Active source right now: **{registry.active_source_name()}**")

st.subheader("API keys")
st.markdown(
    f"Keys are stored in `{config.ENV_FILE.name}` next to the app, never in code. "
    "Phase 1 needs **no keys**: yfinance is free. Modules that need a key will show "
    "\"add your key in Settings\" until you add it here."
)
status = config.key_status()
for name, (purpose, phase) in config.API_KEYS.items():
    c1, c2, c3 = st.columns([2, 3, 1])
    c1.markdown(f"`{name}`  \n<small>{purpose} · {phase}</small>", unsafe_allow_html=True)
    val = c2.text_input(name, type="password", label_visibility="collapsed", placeholder="set" if status[name] else "not set")
    c3.markdown("✅ set" if status[name] else "— not set")
    if val:
        config.set_api_key(name, val)
        st.toast(f"Saved {name}")

st.subheader("Local cache")
s = cache.stats()
st.write(f"{s['entries']} cached responses · {s['size_kb']} KB · prices kept 6 h, quotes 5 min")
if st.button("Clear cache"):
    n = cache.clear()
    st.success(f"Removed {n} entries.")
