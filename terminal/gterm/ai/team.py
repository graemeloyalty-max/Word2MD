"""Multi-agent research: a planner, specialist agents and an editor.

Plain English:
1. The **planner** reads your question and writes one focused sub-question for each specialist
   that is relevant (it can leave some out). If its reply can't be read, every enabled
   specialist gets your full question instead.
2. Each **specialist** (fundamentals, valuation, macro, news) answers its sub-question using only
   its own tools, on its own provider/model. A specialist that asks for a tool outside its list
   gets an error back, not the data.
3. The **editor** has no tools. It gets the specialists' notes plus a fact sheet of the raw tool
   results, writes the research note, and must check each conclusion against the numbers (e.g. a
   multiple above the peer median is a premium, not a discount) and add no new figures.

Agent settings (role, provider, model, tools, step limit) are editable on the page and saved
in data/agents.json. Everything runs on your free keys, one request per step.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from ..core import config
from . import agent, tools
from .providers import PROVIDERS, ProviderError

DEFAULT_PROVIDER = "Google AI Studio (Gemini)"
DEFAULT_MODEL = "gemini-3.5-flash"


@dataclass
class AgentConfig:
    name: str
    role: str  # instructions added to the system prompt
    tools: list = field(default_factory=list)
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    max_steps: int = 6
    enabled: bool = True


DEFAULTS = [
    AgentConfig("Planner", "Split the user's research question into one focused sub-question per relevant specialist. "
                "Reply with JSON only, mapping specialist names to sub-questions, e.g. "
                '{"Fundamentals": "...", "Valuation": "..."}. Omit specialists that are not relevant. Keep tickers as given.',
                [], max_steps=1),
    AgentConfig("Fundamentals", "You analyse financial statements: growth, margins, cash flow, balance sheet and business quality. "
                "Report trends over the years available.", ["get_quote", "get_fundamentals", "get_quality_score"]),
    AgentConfig("Valuation", "You value companies: run the DCF and compare multiples with peers. Always state the key DCF assumptions. "
                "A company multiple ABOVE the peer median means it trades at a PREMIUM (more expensive); BELOW means a discount.",
                ["get_quote", "run_dcf", "get_peer_multiples"]),
    AgentConfig("Macro", "You cover the macro backdrop relevant to the question: growth, inflation, rates and the company's home economy.",
                ["get_macro_indicator", "get_policy_rate"]),
    AgentConfig("News", "You summarise recent news and share-price performance, and flag risks and catalysts.",
                ["search_news", "get_price_performance", "get_quote"]),
    AgentConfig("Editor", "Write the final research note in Markdown with these sections: ## Summary, ## Fundamentals, ## Valuation, "
                "## Macro backdrop, ## News, risks and catalysts, ## Conclusion. Skip sections with no material. "
                "Use only figures that appear in the specialist notes or fact sheet, with their [source, date] citations. "
                "Before writing each conclusion, check it agrees with the numbers; if two figures conflict, say so. "
                "End with: 'Research, not investment advice.'", [], max_steps=1),
]
SPECIALISTS = ("Fundamentals", "Valuation", "Macro", "News")


def _path():
    return config.DATA_DIR / "agents.json"


def load_configs() -> dict[str, AgentConfig]:
    cfgs = {c.name: AgentConfig(**asdict(c)) for c in DEFAULTS}
    p = _path()
    if p.exists():
        try:
            for name, d in json.loads(p.read_text()).items():
                if name in cfgs:
                    known = {k: v for k, v in d.items() if k in AgentConfig.__dataclass_fields__}
                    cfgs[name] = AgentConfig(**{**asdict(cfgs[name]), **known, "name": name})
        except (json.JSONDecodeError, TypeError):
            pass
    return cfgs


def save_configs(cfgs: dict[str, AgentConfig]) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _path().write_text(json.dumps({n: asdict(c) for n, c in cfgs.items()}, indent=2))


def reset_configs() -> None:
    if _path().exists():
        _path().unlink()


def parse_plan(text: str, enabled: list[str], question: str) -> tuple[dict, bool]:
    """Extract {specialist: sub-question} from the planner's reply. Returns (plan, parsed_ok)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            raw = json.loads(m.group(0))
            plan = {k: str(v).strip() for k, v in raw.items() if k in enabled and str(v).strip()}
            if plan:
                return plan, True
        except json.JSONDecodeError:
            pass
    return {name: question for name in enabled}, False


def fact_sheet(steps: list, limit: int = 9000) -> str:
    """Compact JSON lines of every tool result, for the editor to check figures against."""
    lines, used = [], 0
    for s in steps:
        line = json.dumps({"tool": s["tool"], "args": s["args"], "result": s["result"]}, default=str, separators=(",", ":"))
        if used + len(line) > limit:
            lines.append('{"note":"fact sheet truncated"}')
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


@dataclass
class ResearchResult:
    question: str
    note: str
    plan: dict
    plan_parsed: bool
    specialist_notes: dict  # name -> text
    steps: list  # every tool call: {"agent", "tool", "args", "result"}
    usage: dict  # agent -> (input_tokens, output_tokens, provider, model)
    errors: dict
    created: str


def _provider(cfg: AgentConfig):
    if cfg.provider not in PROVIDERS:
        raise ProviderError(f"Unknown provider {cfg.provider} for {cfg.name}")
    return PROVIDERS[cfg.provider]()


def run_research(question: str, cfgs: dict[str, AgentConfig] | None = None, on_event=None) -> ResearchResult:
    cfgs = cfgs or load_configs()
    emit = on_event or (lambda *a: None)
    enabled = [n for n in SPECIALISTS if cfgs[n].enabled]
    if not enabled:
        raise ValueError("Enable at least one specialist agent.")
    usage, errors, steps, notes = {}, {}, [], {}

    # 1. Plan
    plan, parsed = {n: question for n in enabled}, False
    if cfgs["Planner"].enabled:
        emit("plan", "Planner is splitting the question…")
        p = cfgs["Planner"]
        try:
            turn = _provider(p).chat(p.model, agent.system_prompt() + "\n\n" + p.role + f"\nAvailable specialists: {', '.join(enabled)}.",
                                     [{"role": "user", "text": question}], [])
            usage["Planner"] = (turn.input_tokens, turn.output_tokens, p.provider, p.model)
            plan, parsed = parse_plan(turn.text, enabled, question)
        except ProviderError as exc:
            errors["Planner"] = str(exc)
    emit("plan_done", plan)

    # 2. Specialists
    for name, sub_q in plan.items():
        c = cfgs[name]
        emit("agent", f"{name} ({c.provider.split(' (')[0]} · {c.model}) is working on: {sub_q}")
        defs = [d for d in tools.definitions() if d["name"] in c.tools]
        collected = []

        def record(s, n=name):
            collected.append({**s, "agent": n})
            emit("step", {**s, "agent": n})

        try:
            r = agent.run(_provider(c), c.model, sub_q, max_steps=c.max_steps, tool_defs=defs, allowed=set(c.tools),
                          system=agent.system_prompt() + f"\n\nYou are the {name} specialist on a research team. {c.role}",
                          on_step=record)
            notes[name] = r.answer
            usage[name] = (r.input_tokens, r.output_tokens, c.provider, c.model)
        except ProviderError as exc:
            errors[name] = str(exc)
            notes[name] = (f"(The {name} agent failed before writing its notes: {exc}. "
                           f"Its {len(collected)} tool result(s) are in the fact sheet.)")
        steps += collected  # kept even if the agent failed part-way

    # 3. Edit
    e = cfgs["Editor"]
    emit("agent", f"Editor ({e.provider.split(' (')[0]} · {e.model}) is writing the note…")
    material = "\n\n".join(f"### {n} specialist notes\n{t}" for n, t in notes.items())
    prompt = (f"Research question: {question}\n\n{material}\n\n### Fact sheet (raw tool results, one JSON per line)\n"
              f"{fact_sheet(steps)}\n\nWrite the research note now.")
    try:
        turn = _provider(e).chat(e.model, agent.system_prompt() + "\n\n" + e.role, [{"role": "user", "text": prompt}], [])
        note = turn.text
        usage["Editor"] = (turn.input_tokens, turn.output_tokens, e.provider, e.model)
    except ProviderError as exc:
        errors["Editor"] = str(exc)
        note = "## Specialist notes (editor unavailable)\n\n" + material
    return ResearchResult(question, note, plan, parsed, notes, steps, usage, errors,
                          datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
