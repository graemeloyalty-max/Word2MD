import io

import pytest

from gterm import reports
from gterm.ai import team
from gterm.ai.providers import PROVIDERS, Provider, ProviderError, ToolCall, Turn


def test_parse_plan_reads_json_inside_text_and_filters():
    txt = 'Sure! ```json\n{"Valuation": "Value SAP", "Bogus": "x", "News": ""}\n```'
    plan, ok = team.parse_plan(txt, ["Fundamentals", "Valuation", "News"], "Q")
    assert ok and plan == {"Valuation": "Value SAP"}
    plan, ok = team.parse_plan("no json here", ["Fundamentals", "News"], "Q")
    assert not ok and plan == {"Fundamentals": "Q", "News": "Q"}


def test_configs_save_load_and_reset():
    cfgs = team.load_configs()
    assert set(team.SPECIALISTS) <= set(cfgs) and "run_dcf" in cfgs["Valuation"].tools
    cfgs["News"].enabled = False
    cfgs["Valuation"].model = "openai/gpt-oss-120b"
    team.save_configs(cfgs)
    again = team.load_configs()
    assert again["News"].enabled is False and again["Valuation"].model == "openai/gpt-oss-120b"
    team.reset_configs()
    assert team.load_configs()["News"].enabled is True


class Scripted(Provider):
    """Fake provider: answers by inspecting the system prompt to know which agent is asking."""
    log = []

    def chat(self, model, system, messages, tools_):
        Scripted.log.append((system, [m["role"] for m in messages], [t["name"] for t in tools_]))
        if "Split the user's research question" in system:
            return Turn('{"Valuation": "Value SAP.DE", "Macro": "German growth"}', [], 5, 5)
        if "Valuation specialist" in system:
            if messages[-1]["role"] == "user":  # first step: ask for a tool, plus one it is NOT allowed
                return Turn("", [ToolCall("1", "run_dcf", {"symbol": "SAP.DE"}), ToolCall("2", "search_news", {"query": "x"})], 10, 3)
            return Turn("DCF says X [demo, 2026].", [], 20, 5)
        if "Macro specialist" in system:
            raise ProviderError("rate limit reached")
        if "Write the final research note" in system:
            assert "Fact sheet" in messages[0]["text"] and "run_dcf" in messages[0]["text"]
            return Turn("## Summary\nSAP looks **interesting** [demo, 2026].\n\nResearch, not investment advice.", [], 50, 20)
        raise AssertionError(system)


def test_research_team_end_to_end(monkeypatch):
    monkeypatch.setitem(PROVIDERS, "Fake", Scripted)
    cfgs = team.load_configs()
    for c in cfgs.values():
        c.provider = "Fake"
    events = []
    r = team.run_research("Value SAP.DE", cfgs, on_event=lambda k, v: events.append(k))
    assert r.plan_parsed and set(r.plan) == {"Valuation", "Macro"}  # planner left out Fundamentals and News
    dcf = [s for s in r.steps if s["tool"] == "run_dcf"][0]
    assert dcf["agent"] == "Valuation" and "source" in dcf["result"]
    blocked = [s for s in r.steps if s["tool"] == "search_news"][0]
    assert "not available" in blocked["result"]["error"]  # tool outside the agent's list
    assert "Macro" in r.errors and "failed" in r.specialist_notes["Macro"]  # one agent failing doesn't stop the team
    assert r.note.startswith("## Summary") and r.usage["Editor"][:2] == (50, 20)
    assert {"plan", "agent", "step"} <= set(events)
    val_tools = [t for s, roles, t in Scripted.log if "Valuation specialist" in s][0]
    assert set(val_tools) == {"get_quote", "run_dcf", "get_peer_multiples"}  # only its own tools are offered


def sample_report():
    return {"title": "SAP review", "question": "Value SAP", "created": "2026-11-25 10:00 UTC",
            "note": "## Summary\nSAP trades at **€190** [yfinance, 2026-11-25].\n- Point one\n1. First\n\n| A | B |\n|---|---|\n| 1 | 2 |",
            "steps": [{"agent": "Valuation", "tool": "run_dcf", "args": {"symbol": "SAP.DE"},
                       "result": {"source": "yfinance", "as_of": "2026-11-25 10:00 UTC"}},
                      {"agent": "News", "tool": "search_news", "args": {"query": "SAP"}, "result": {"error": "blocked"}}],
            "usage": {"Editor": [100, 50, "Groq", "openai/gpt-oss-120b"]}}


def test_markdown_has_sources_appendix():
    md = reports.to_markdown(sample_report())
    assert "## Sources" in md and "| 1 | Valuation | run_dcf | symbol=SAP.DE | yfinance |" in md
    assert "error: blocked" in md and reports.DISCLAIMER in md


def test_blocks_and_inline_parsing():
    kinds = [k for k, _ in reports.blocks(sample_report()["note"])]
    assert kinds == ["h", "p", "li", "ol", "table"]
    assert reports.inline_runs("a **b** *c* [d](http://e)") == [("a ", False, False), ("b", True, False), (" ", False, False),
                                                              ("c", False, True), (" d (http://e)", False, False)]


def test_docx_and_pdf_exports_are_valid():
    from docx import Document

    data = reports.to_docx(sample_report())
    doc = Document(io.BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "SAP trades at €190" in text and len(doc.tables) == 2  # note table + sources table
    pdf = reports.to_pdf(sample_report())
    assert pdf[:5] == b"%PDF-" and len(pdf) > 1500


def test_save_list_load_delete():
    p = reports.save(sample_report())
    lst = reports.list_reports()
    assert lst[0]["title"] == "SAP review" and reports.load(p)["note"].startswith("## Summary")
    reports.delete(p)
    assert reports.list_reports() == []


def test_snake_case_is_not_italicised():
    assert reports.plain("get_macro_indicator and get_policy_rate") == "get_macro_indicator and get_policy_rate"
    assert reports.inline_runs("_meta line_") == [("meta line", False, True)]
    assert reports.plain("2 * 3 * 4") == "2 * 3 * 4"
