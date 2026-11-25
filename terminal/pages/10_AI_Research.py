import json

import streamlit as st

from gterm import ui
from gterm.ai import agent, tools
from gterm.ai.providers import PROVIDERS, ProviderError
from gterm.core import config

ui.page_setup("AI Research", "🤖")
settings = config.load_settings()

c1, c2, c3 = st.columns([2, 3, 1])
names = list(PROVIDERS)
prov_name = c1.selectbox("Provider", names, index=names.index(settings.get("ai_provider", names[0])) if settings.get("ai_provider") in names else 0)
provider = PROVIDERS[prov_name]()


@st.cache_data(ttl=3600, show_spinner=False)
def models_for(name: str) -> list[str]:
    return PROVIDERS[name]().list_models()


if not config.get_key(provider.key_name):
    st.info(f"Add your free **{provider.key_name}** in Settings to use {prov_name}. "
            + ("Get one at aistudio.google.com → Get API key." if "Gemini" in prov_name else "Get one at console.groq.com → API Keys."))
    st.stop()
try:
    models = models_for(prov_name)
except ProviderError as exc:
    st.error(str(exc))
    st.stop()
saved = settings.get("ai_models", {}).get(prov_name)
model = c2.selectbox("Model", models, index=models.index(saved) if saved in models else 0,
                     help="Pick a model that supports tool (function) calling. The list comes live from the provider.")
max_steps = c3.number_input("Max tool steps", 1, 20, 8, help="Each step is one request against your free quota.")
if (prov_name, model) != (settings.get("ai_provider"), saved):
    config.save_settings({**settings, "ai_provider": prov_name, "ai_models": {**settings.get("ai_models", {}), prov_name: model}})

st.caption("The AI can only use the terminal's own tools: " + ", ".join(f"`{n}`" for n in tools.TOOLS)
           + ". Every number should carry a [source, date] citation. Research, not investment advice.")

key = f"chat_{prov_name}"
if key not in st.session_state:
    st.session_state[key] = {"history": [], "display": [], "tokens": [0, 0]}
chat = st.session_state[key]

for item in chat["display"]:
    with st.chat_message(item["role"]):
        st.markdown(item["text"])
        for s in item.get("steps", []):
            with st.expander(f"🔧 {s['tool']}({', '.join(f'{k}={v}' for k, v in s['args'].items())})"):
                st.code(json.dumps(s["result"], indent=1, default=str)[:4000], language="json")
        if item.get("meta"):
            st.caption(item["meta"])

examples = ["Value SAP.DE: run a DCF, compare it with peers and give me the quality score.",
            "How has inflation in South Africa compared with the euro area over the last 5 years?",
            "What is the latest news on Shell and how has the share performed this year?"]
q = st.chat_input("Ask about a company, market or economy…")
if not chat["display"]:
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            q = ex
if q:
    with st.chat_message("user"):
        st.markdown(q)
    chat["display"].append({"role": "user", "text": q})
    with st.chat_message("assistant"):
        live = st.status("Working…", expanded=True)
        try:
            res = agent.run(provider, model, q, chat["history"], int(max_steps),
                            on_step=lambda s: live.write(f"🔧 `{s['tool']}` {s['args']}" + (" ⚠️ error" if "error" in s["result"] else "")))
        except ProviderError as exc:
            live.update(label="Failed", state="error")
            st.error(str(exc))
            st.stop()
        live.update(label=f"Done: {len(res.steps)} tool call(s)", state="complete", expanded=False)
        st.markdown(res.answer or "_(no answer)_")
        chat["tokens"][0] += res.input_tokens
        chat["tokens"][1] += res.output_tokens
        meta = (f"{prov_name} · {model} · {res.input_tokens:,} input / {res.output_tokens:,} output tokens this answer · "
                f"session total {chat['tokens'][0]:,} / {chat['tokens'][1]:,}")
        st.caption(meta)
    chat["history"] = res.messages
    chat["display"].append({"role": "assistant", "text": res.answer, "steps": res.steps, "meta": meta})
    st.rerun()

if chat["display"] and st.button("🧹 New conversation"):
    del st.session_state[key]
    st.rerun()
