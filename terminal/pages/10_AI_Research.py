import json

import streamlit as st

from gterm import reports, ui
from gterm.ai import agent, team, tools
from gterm.ai.providers import PROVIDERS, ProviderError
from gterm.core import config

ui.page_setup("AI Research", "🤖")
settings = config.load_settings()
NOTE = ("Every number should carry a [source, date] citation; explanations around the numbers come from the model's own "
        "general knowledge, which is unsourced and may be out of date. Free models can misread results, so check the tool "
        "results shown. Research, not investment advice.")


@st.cache_data(ttl=3600, show_spinner=False)
def models_for(name: str) -> list[str]:
    return PROVIDERS[name]().list_models()


def safe_models(name: str) -> list[str]:
    try:
        return models_for(name) if config.get_key(PROVIDERS[name].key_name) else []
    except ProviderError:
        return []


def show_steps(steps, label_agent=False):
    for s in steps:
        who = f"[{s['agent']}] " if label_agent and s.get("agent") else ""
        err = " ⚠️" if isinstance(s["result"], dict) and "error" in s["result"] else ""
        with st.expander(f"🔧 {who}{s['tool']}({', '.join(f'{k}={v}' for k, v in s['args'].items())}){err}"):
            st.code(json.dumps(s["result"], indent=1, default=str)[:4000], language="json")


t_chat, t_team, t_agents, t_reports = st.tabs(["💬 Chat", "👥 Research team", "⚙️ Agents", "📄 Reports"])

# ---------------------------------------------------------------- chat
with t_chat:
    def chat_tab():
        c1, c2, c3 = st.columns([2, 3, 1])
        names = list(PROVIDERS)
        cur = settings.get("ai_provider")
        prov_name = c1.selectbox("Provider", names, index=names.index(cur) if cur in names else 0)
        provider = PROVIDERS[prov_name]()
        if not config.get_key(provider.key_name):
            st.info(f"Add your free **{provider.key_name}** in Settings to use {prov_name}. "
                    + ("Get one at aistudio.google.com → Get API key." if "Gemini" in prov_name else "Get one at console.groq.com → API Keys."))
            return
        try:
            models = models_for(prov_name)
        except ProviderError as exc:
            st.error(str(exc))
            return
        saved = settings.get("ai_models", {}).get(prov_name)
        default = saved if saved in models else provider.default_model(models)
        model = c2.selectbox("Model", models, index=models.index(default),
                             help="Defaults to a model tested with the terminal's tools. If a model is busy (HTTP 503), pick another.")
        max_steps = c3.number_input("Max tool steps", 1, 20, 8, help="Each step is one request against your free quota.")
        if (prov_name, model) != (cur, saved):
            config.save_settings({**settings, "ai_provider": prov_name, "ai_models": {**settings.get("ai_models", {}), prov_name: model}})
        st.caption("Tools: " + ", ".join(f"`{n}`" for n in tools.TOOLS) + ". " + NOTE)

        key = f"chat_{prov_name}"
        chat = st.session_state.setdefault(key, {"history": [], "display": [], "tokens": [0, 0]})
        for item in chat["display"]:
            with st.chat_message(item["role"]):
                st.markdown(item["text"])
                show_steps(item.get("steps", []))
                if item.get("meta"):
                    st.caption(item["meta"])
        examples = ["Value SAP.DE: run a DCF, compare it with peers and give me the quality score.",
                    "How has inflation in South Africa compared with the euro area over the last 5 years?",
                    "What is the latest news on Shell and how has the share performed this year?"]
        q = st.chat_input("Ask about a company, market or economy…")
        if not chat["display"]:
            for col, ex in zip(st.columns(len(examples)), examples):
                if col.button(ex, use_container_width=True):
                    q = ex
        if not q:
            if chat["display"] and st.button("🧹 New conversation"):
                del st.session_state[key]
                st.rerun()
            return
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
                return
            live.update(label=f"Done: {len(res.steps)} tool call(s)", state="complete", expanded=False)
        chat["tokens"][0] += res.input_tokens
        chat["tokens"][1] += res.output_tokens
        meta = (f"{prov_name} · {model} · {res.input_tokens:,} in / {res.output_tokens:,} out tokens · "
                f"session {chat['tokens'][0]:,} / {chat['tokens'][1]:,}")
        chat["history"] = res.messages
        chat["display"].append({"role": "assistant", "text": res.answer or "_(no answer)_", "steps": res.steps, "meta": meta})
        st.rerun()

    chat_tab()

# ---------------------------------------------------------------- research team
with t_team:
    cfgs = team.load_configs()
    active = [n for n in team.SPECIALISTS if cfgs[n].enabled]
    st.markdown("A **planner** splits your question, **specialists** research their part with their own tools, and an **editor** "
                "writes a cited research note. Team: " + " → ".join(
                    [f"Planner ({cfgs['Planner'].model})"] + [f"{n} ({cfgs[n].model})" for n in active] + [f"Editor ({cfgs['Editor'].model})"])
                + ". Change who does what in the **Agents** tab.")
    missing = {cfgs[n].provider for n in ["Planner", *active, "Editor"] if not config.get_key(PROVIDERS[cfgs[n].provider].key_name)}
    if missing:
        st.warning("Missing key for: " + ", ".join(missing) + ". Add it in Settings or switch those agents' provider in the Agents tab.")
    q = st.text_area("Research question", "Write a research note on SAP.DE: fundamentals, valuation versus peers, the German macro "
                     "backdrop and recent news.", height=80)
    calls = 2 + sum(cfgs[n].max_steps for n in active)
    st.caption(f"Uses up to about {calls} requests on your free quotas. {NOTE}")
    if st.button("▶ Run research team", type="primary", disabled=bool(missing) or not q.strip()):
        status = st.status("Research team at work…", expanded=True)

        def on_event(kind, val):
            if kind in ("plan", "agent"):
                status.write(f"**{val}**")
            elif kind == "plan_done":
                status.write("Plan: " + "; ".join(f"*{k}*: {v}" for k, v in val.items()))
            elif kind == "step":
                status.write(f"  🔧 [{val['agent']}] `{val['tool']}` {val['args']}" + (" ⚠️" if "error" in val["result"] else ""))

        try:
            res = team.run_research(q.strip(), cfgs, on_event)
        except (ValueError, ProviderError) as exc:
            status.update(label="Failed", state="error")
            st.error(str(exc))
        else:
            status.update(label=f"Done: {len(res.steps)} tool calls across {len(res.specialist_notes)} specialists",
                          state="complete", expanded=False)
            st.session_state["last_research"] = res
            st.session_state["last_saved"] = str(reports.save(res))
    res = st.session_state.get("last_research")
    if res:
        if res.errors:
            st.warning("Some agents failed: " + "; ".join(f"{k}: {v[:150]}" for k, v in res.errors.items()))
        if not res.plan_parsed:
            st.caption("The planner's reply couldn't be read, so every specialist got the full question.")
        with st.container(border=True):
            st.markdown(res.note)
        tot_in = sum(u[0] for u in res.usage.values())
        tot_out = sum(u[1] for u in res.usage.values())
        st.caption(f"{res.created} · {tot_in:,} input / {tot_out:,} output tokens · "
                   + "; ".join(f"{a}: {u[3]}" for a, u in res.usage.items()) + " · saved to the Reports tab")
        with st.expander("Specialist notes"):
            for n, t in res.specialist_notes.items():
                st.markdown(f"#### {n}\n{t}")
        with st.expander(f"All {len(res.steps)} tool calls (raw results)"):
            show_steps(res.steps, label_agent=True)

# ---------------------------------------------------------------- agent settings
with t_agents:
    st.markdown("Set each agent's provider, model, tools and instructions. Saved on this machine in `data/agents.json`. "
                "Tip: a fast model (e.g. Groq) for data-gathering specialists and a stronger one for the editor.")
    cfgs = team.load_configs()
    tool_names = list(tools.TOOLS)
    with st.form("agents"):
        for name, c in cfgs.items():
            with st.expander(f"{'✅' if c.enabled else '⏸️'} {name}: {c.provider.split(' (')[0]} · {c.model}", expanded=False):
                a, b, d = st.columns([1, 2, 2])
                c.enabled = a.checkbox("Enabled", c.enabled, key=f"en_{name}", disabled=name == "Editor")
                c.provider = b.selectbox("Provider", list(PROVIDERS),
                                         index=list(PROVIDERS).index(c.provider) if c.provider in PROVIDERS else 0, key=f"pv_{name}")
                opts = safe_models(c.provider)
                if opts:
                    c.model = d.selectbox("Model", opts, index=opts.index(c.model) if c.model in opts else 0, key=f"md_{name}")
                else:
                    c.model = d.text_input("Model", c.model, key=f"md_{name}", help="No key or model list for this provider yet.")
                if name in team.SPECIALISTS:
                    c.tools = st.multiselect("Tools this agent may use", tool_names, [t for t in c.tools if t in tool_names], key=f"tl_{name}")
                    c.max_steps = st.slider("Max tool steps", 1, 15, c.max_steps, key=f"ms_{name}")
                c.role = st.text_area("Instructions", c.role, height=110, key=f"rl_{name}")
        st.caption("After changing a provider, save once to refresh that agent's model list.")
        if st.form_submit_button("💾 Save agent settings", type="primary"):
            team.save_configs(cfgs)
            st.success("Saved.")
    if st.button("↺ Reset all agents to defaults"):
        team.reset_configs()
        st.rerun()

# ---------------------------------------------------------------- reports
with t_reports:
    lib = reports.list_reports()
    if not lib:
        st.info("No reports yet. Run the research team: every run is saved here automatically.")
    else:
        labels = [f"{r['created']} · {r['title']}" for r in lib]
        default_i = next((i for i, r in enumerate(lib) if str(r["path"]) == st.session_state.get("last_saved")), 0)
        pick = st.selectbox("Report", range(len(lib)), index=default_i, format_func=lambda i: labels[i])
        rep = reports.load(lib[pick]["path"])
        new_title = st.text_input("Title", rep["title"])
        if new_title != rep["title"]:
            rep["title"] = new_title
            lib[pick]["path"].write_text(json.dumps(rep, indent=1, default=str))
        base = reports.slug(rep["title"])
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("⬇ Word (.docx)", reports.to_docx(rep), f"{base}.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
        c2.download_button("⬇ PDF", reports.to_pdf(rep), f"{base}.pdf", "application/pdf", use_container_width=True)
        c3.download_button("⬇ Markdown", reports.to_markdown(rep).encode(), f"{base}.md", "text/markdown", use_container_width=True)
        if c4.button("🗑 Delete", use_container_width=True):
            reports.delete(lib[pick]["path"])
            st.rerun()
        with st.container(border=True):
            st.markdown(reports.to_markdown(rep))
