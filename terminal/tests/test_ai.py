import json

import pytest

from gterm.ai import agent, tools
from gterm.ai.providers import GeminiProvider, GroqProvider, Provider, ProviderError, ToolCall, Turn

TOOLS = [{"name": "get_quote", "description": "d", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}]


# ------------------------------------------------------------------ Gemini

GEMINI_CALL = {"candidates": [{"content": {"role": "model", "parts": [
    {"text": "Let me check."},
    {"functionCall": {"name": "get_quote", "args": {"symbol": "SAP.DE"}}, "thoughtSignature": "abc123"}]}}],
    "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 5}}


def test_gemini_parse_tool_call_and_usage():
    t = GeminiProvider.parse_response(GEMINI_CALL)
    assert t.text == "Let me check." and t.tool_calls[0].name == "get_quote" and t.tool_calls[0].args == {"symbol": "SAP.DE"}
    assert (t.input_tokens, t.output_tokens) == (100, 25)


def test_gemini_request_echoes_model_content_and_groups_results():
    t = GeminiProvider.parse_response(GEMINI_CALL)
    c2 = ToolCall("call_2", "get_quote", {"symbol": "AAPL"})
    msgs = [{"role": "user", "text": "Price of SAP?"},
            {"role": "assistant", "text": t.text, "tool_calls": t.tool_calls, "raw": t.raw},
            {"role": "tool", "call": t.tool_calls[0], "result": {"price": 190}},
            {"role": "tool", "call": c2, "result": {"price": 350}}]
    body = GeminiProvider.to_request("sys", msgs, TOOLS)
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"
    assert body["contents"][1]["parts"][1]["thoughtSignature"] == "abc123"  # echoed unchanged
    results = body["contents"][2]
    assert results["role"] == "user" and len(results["parts"]) == 2  # both results in one message
    assert results["parts"][0]["functionResponse"] == {"name": "get_quote", "response": {"result": {"price": 190}}}
    assert body["tools"][0]["functionDeclarations"][0]["name"] == "get_quote"


def test_gemini_skips_thought_parts_and_reports_blocks():
    t = GeminiProvider.parse_response({"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": "Answer"}]}}]})
    assert t.text == "Answer"
    with pytest.raises(ProviderError, match="SAFETY"):
        GeminiProvider.parse_response({"promptFeedback": {"blockReason": "SAFETY"}})


# ------------------------------------------------------------------ Groq

GROQ_CALL = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
    {"id": "call_x", "type": "function", "function": {"name": "get_quote", "arguments": "{\"symbol\": \"SHEL.L\"}"}}]}}],
    "usage": {"prompt_tokens": 80, "completion_tokens": 12}}


def test_groq_parse_and_request_round_trip():
    t = GroqProvider.parse_response(GROQ_CALL)
    assert t.text == "" and t.tool_calls[0].args == {"symbol": "SHEL.L"} and t.tool_calls[0].id == "call_x"
    msgs = [{"role": "user", "text": "Shell?"}, {"role": "assistant", "text": "", "tool_calls": t.tool_calls},
            {"role": "tool", "call": t.tool_calls[0], "result": {"price": 3552}}]
    body = GroqProvider.to_request("m", "sys", msgs, TOOLS)
    assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "tool"]
    assert json.loads(body["messages"][2]["tool_calls"][0]["function"]["arguments"]) == {"symbol": "SHEL.L"}
    assert body["messages"][3]["tool_call_id"] == "call_x" and body["tool_choice"] == "auto"
    assert body["tools"][0] == {"type": "function", "function": TOOLS[0]}


def test_groq_bad_json_arguments_dont_crash():
    bad = {"choices": [{"message": {"tool_calls": [{"id": "1", "function": {"name": "get_quote", "arguments": "{oops"}}]}}]}
    assert GroqProvider.parse_response(bad).tool_calls[0].args == {"_unparsed": "{oops"}


def test_missing_key_message():
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        GroqProvider().key


# ------------------------------------------------------------------ agent loop with a scripted fake model

class FakeProvider(Provider):
    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def chat(self, model, system, messages, tools_):
        self.seen.append(list(messages))
        return self.turns.pop(0)


def test_agent_runs_tools_then_answers():
    fake = FakeProvider([Turn("", [ToolCall("1", "get_quote", {"symbol": "AAPL"})], 10, 2),
                         Turn("Apple trades at X [demo].", [], 30, 8)])
    r = agent.run(fake, "m", "Price of Apple?")
    assert r.answer.startswith("Apple trades") and len(r.steps) == 1
    assert r.steps[0]["result"]["symbol"] == "AAPL" and "source" in r.steps[0]["result"]
    assert (r.input_tokens, r.output_tokens) == (40, 10)
    assert fake.seen[1][-1]["role"] == "tool"  # the tool result was sent back to the model


def test_agent_step_limit():
    loop = [Turn("", [ToolCall(str(i), "get_quote", {"symbol": "AAPL"})]) for i in range(5)]
    r = agent.run(FakeProvider(loop), "m", "q", max_steps=3)
    assert r.stopped_early and len(r.steps) == 3


# ------------------------------------------------------------------ tools on demo data

def test_tools_return_sources_and_handle_errors():
    for name, args in [("get_quote", {"symbol": "SAP.DE"}), ("get_price_performance", {"symbol": "SAP.DE"}),
                       ("get_fundamentals", {"symbol": "SAP.DE"}), ("run_dcf", {"symbol": "SAP.DE"}),
                       ("get_quality_score", {"symbol": "SAP.DE"})]:
        out = tools.execute(name, args)
        assert "error" not in out, (name, out)
        assert out["source"] and out["as_of"]
    assert "error" in tools.execute("run_dcf", {"symbol": "JPM"})  # bank
    assert "error" in tools.execute("get_quote", {"wrong": 1})
    assert "error" in tools.execute("nope", {})
    assert {d["name"] for d in tools.definitions()} == set(tools.TOOLS)
