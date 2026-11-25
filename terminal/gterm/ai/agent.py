"""The research agent loop: ask the model, run the tools it asks for, repeat until it answers.

Plain English: the model never sees the internet or the app directly. It can only *ask* for one
of the tools in `tools.py`; the app runs the tool, sends the result back, and the model decides
what to do next. A step limit stops runaway loops (and protects free-tier quotas).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import tools
from .providers import Provider, ProviderError

SYSTEM_PROMPT = """You are the research assistant inside a personal global equity research terminal.
Answer using the tools provided; do not invent numbers. Every figure you state must come from a tool
result: cite its source and date in brackets, e.g. [yfinance, 2026-11-24]. If a tool returns an error
or data is missing, say so plainly. Valuation outputs depend on editable assumptions; mention the key
ones (growth, WACC, terminal growth) when you quote a DCF value. Be concise. This is research, not
investment advice."""


@dataclass
class AgentResult:
    answer: str
    messages: list
    steps: list = field(default_factory=list)  # [{"tool", "args", "result"}]
    input_tokens: int = 0
    output_tokens: int = 0
    stopped_early: bool = False


def run(provider: Provider, model: str, question: str, history: list | None = None, max_steps: int = 8,
        system: str = SYSTEM_PROMPT, tool_defs: list | None = None, on_step=None) -> AgentResult:
    messages = list(history or []) + [{"role": "user", "text": question}]
    defs = tool_defs if tool_defs is not None else tools.definitions()
    res = AgentResult("", messages)
    for _ in range(max_steps):
        turn = provider.chat(model, system, messages, defs)
        res.input_tokens += turn.input_tokens
        res.output_tokens += turn.output_tokens
        messages.append({"role": "assistant", "text": turn.text, "tool_calls": turn.tool_calls, "raw": turn.raw})
        if not turn.tool_calls:
            res.answer = turn.text
            return res
        for call in turn.tool_calls:
            result = tools.execute(call.name, call.args)
            messages.append({"role": "tool", "call": call, "result": result})
            step = {"tool": call.name, "args": call.args, "result": result}
            res.steps.append(step)
            if on_step:
                on_step(step)
    res.stopped_early = True
    res.answer = (turn.text or "") + f"\n\n_(Stopped after {max_steps} steps to protect your quota. Ask a narrower question or raise the limit.)_"
    return res


__all__ = ["run", "AgentResult", "SYSTEM_PROMPT", "ProviderError"]
