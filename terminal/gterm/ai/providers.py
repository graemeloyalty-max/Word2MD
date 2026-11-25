"""AI providers behind one interface, like the data adapters.

The agent speaks a neutral format; each adapter translates to its service's API:

| Provider | Key (free tier) | API |
|---|---|---|
| Google AI Studio (Gemini) | GEMINI_API_KEY from aistudio.google.com | generativelanguage.googleapis.com, `generateContent` |
| Groq | GROQ_API_KEY from console.groq.com | api.groq.com, OpenAI-compatible `chat/completions` |

Neutral messages: {"role": "user"|"assistant"|"tool", ...}
- user:      {"role": "user", "text": str}
- assistant: {"role": "assistant", "text": str, "tool_calls": [ToolCall], "raw": provider-specific copy}
- tool:      {"role": "tool", "call": ToolCall, "result": dict}
Tools: [{"name", "description", "parameters": JSON schema}]

Free tiers have per-minute and per-day request limits; a rate-limit error is reported clearly.
Check each provider's current terms: some free tiers may use prompts to improve their models.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ..core import config


class ProviderError(Exception):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Turn:
    text: str
    tool_calls: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    raw: object = None  # provider-specific message, echoed back unchanged on the next request


# Groq sits behind Cloudflare, which rejects Python's default "Python-urllib" client name (error 1010).
USER_AGENT = "GlobalEquityTerminal/1.0 (personal research tool)"


def _post(url: str, body: dict, headers: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, **headers})
    return _send(req, timeout)


def _get_json(url: str, headers: dict) -> dict:
    return _send(urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers}), 30)


RETRY_WAITS = (2, 5)  # seconds; for "overloaded / try again" answers (HTTP 500, 502, 503)


def _send(req, timeout) -> dict:
    import time

    for wait in (*RETRY_WAITS, None):
        try:
            return _send_once(req, timeout)
        except _Retryable as exc:
            if wait is None:
                what = "rate limit still reached" if exc.code == 429 else "service busy"
                raise ProviderError(f"{_host(req)}: {what} (HTTP {exc.code}) after retries: {exc.detail}") from None
            time.sleep(exc.wait if exc.wait is not None else wait)


class _Retryable(Exception):
    def __init__(self, code, detail, wait=None):
        self.code, self.detail, self.wait = code, detail, wait


MAX_RATE_WAIT = 30  # seconds: wait out short free-tier rate limits (e.g. Groq's tokens-per-minute) instead of failing


def _suggested_wait(detail: str, retry_after) -> float | None:
    """How long the provider says to wait: Retry-After header, or text like 'try again in 10.98s' / 'retry in 7s'."""
    try:
        if retry_after:
            return float(retry_after) + 0.5
    except ValueError:
        pass
    m = re.search(r"(?:try again|retry) in ([0-9.]+)\s*(ms|s)", detail, re.I)
    if m:
        v = float(m.group(1))
        return (v / 1000 if m.group(2).lower() == "ms" else v) + 0.5
    return None


def _host(req) -> str:
    import urllib.parse

    return urllib.parse.urlparse(req.full_url).hostname or "provider"


def _send_once(req, timeout) -> dict:
    host = _host(req)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        if exc.code in (500, 502, 503):
            raise _Retryable(exc.code, detail) from exc
        if exc.code == 429:
            wait = _suggested_wait(detail, exc.headers.get("retry-after") if exc.headers else None)
            if wait is not None and wait <= MAX_RATE_WAIT:
                raise _Retryable(429, detail, wait) from exc
            raise ProviderError(f"{host}: rate limit reached (free-tier quota). Wait a minute or switch provider. {detail}") from exc
        if "1010" in detail:
            raise ProviderError(f"{host}: blocked by Cloudflare (error 1010: client signature), not a key problem.") from exc
        if exc.code in (401, 403):
            raise ProviderError(f"{host}: key missing or invalid (HTTP {exc.code}). Add it in Settings. {detail}") from exc
        raise ProviderError(f"{host}: HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        hint = " The network blocked this host." if "Tunnel" in str(exc) or "403" in str(exc) else ""
        raise ProviderError(f"{host} request failed: {exc}.{hint}") from exc


class Provider:
    name = "base"
    key_name = ""
    #: models verified live with the terminal's tools (25 Nov 2026), tried in order as the default
    preferred: tuple = ()

    def default_model(self, available: list[str]) -> str:
        return next((m for m in self.preferred if m in available), available[0] if available else "")

    @property
    def key(self) -> str:
        k = config.get_key(self.key_name)
        if not k:
            raise ProviderError(f"No {self.key_name} set. Add your free key in Settings.")
        return k

    def list_models(self) -> list[str]:
        raise NotImplementedError

    def chat(self, model: str, system: str, messages: list, tools: list) -> Turn:
        raise NotImplementedError


# ---------------------------------------------------------------- Google AI Studio (Gemini)

class GeminiProvider(Provider):
    name = "Google AI Studio (Gemini)"
    key_name = "GEMINI_API_KEY"
    preferred = ("gemini-3.5-flash", "gemini-flash-latest", "gemini-2.5-flash")
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def _headers(self):
        return {"x-goog-api-key": self.key}

    def list_models(self) -> list[str]:
        data = _get_json(f"{self.BASE}/models?pageSize=200", self._headers())
        return sorted(m["name"].removeprefix("models/") for m in data.get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", []))

    @staticmethod
    def to_request(system: str, messages: list, tools: list) -> dict:
        contents = []
        for m in messages:
            if m["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": m["text"]}]})
            elif m["role"] == "assistant":
                # Echo the model's own content unchanged (it may carry thought signatures Gemini requires back).
                contents.append(m.get("raw") or {"role": "model", "parts": [{"text": m.get("text", "")}]})
            elif m["role"] == "tool":
                part = {"functionResponse": {"name": m["call"].name, "response": {"result": m["result"]}}}
                if contents and contents[-1]["role"] == "user" and all("functionResponse" in p for p in contents[-1]["parts"]):
                    contents[-1]["parts"].append(part)  # all results of one turn go in a single message
                else:
                    contents.append({"role": "user", "parts": [part]})
        body = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [{"functionDeclarations": [
                {"name": t["name"], "description": t["description"], "parameters": t["parameters"]} for t in tools]}]
        return body

    @staticmethod
    def parse_response(data: dict) -> Turn:
        cands = data.get("candidates") or []
        if not cands:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates returned")
            raise ProviderError(f"Gemini returned no answer: {reason}")
        content = cands[0].get("content") or {"role": "model", "parts": []}
        texts, calls = [], []
        for i, p in enumerate(content.get("parts", [])):
            if p.get("thought"):
                continue  # thought summaries are not part of the answer
            if "text" in p:
                texts.append(p["text"])
            if "functionCall" in p:
                fc = p["functionCall"]
                calls.append(ToolCall(fc.get("id") or f"call_{i}", fc["name"], fc.get("args") or {}))
        usage = data.get("usageMetadata", {})
        return Turn("".join(texts), calls, usage.get("promptTokenCount", 0),
                    usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0), content)

    def chat(self, model, system, messages, tools) -> Turn:
        data = _post(f"{self.BASE}/models/{model}:generateContent", self.to_request(system, messages, tools), self._headers())
        return self.parse_response(data)


# ---------------------------------------------------------------- Groq (OpenAI-compatible)

class GroqProvider(Provider):
    name = "Groq"
    key_name = "GROQ_API_KEY"
    preferred = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
    BASE = "https://api.groq.com/openai/v1"

    def _headers(self):
        return {"Authorization": f"Bearer {self.key}"}

    def list_models(self) -> list[str]:
        data = _get_json(f"{self.BASE}/models", self._headers())
        skip = ("whisper", "tts", "guard", "playai", "orpheus")
        return sorted(m["id"] for m in data.get("data", []) if m.get("active", True) and not any(s in m["id"] for s in skip))

    @staticmethod
    def to_request(model: str, system: str, messages: list, tools: list) -> dict:
        out = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("text") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": c.id, "type": "function",
                                          "function": {"name": c.name, "arguments": json.dumps(c.args)}} for c in m["tool_calls"]]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["call"].id, "content": json.dumps(m["result"], default=str)})
        body = {"model": model, "messages": out}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
            body["tool_choice"] = "auto"
        return body

    @staticmethod
    def parse_response(data: dict) -> Turn:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("Groq returned no answer")
        msg = choices[0].get("message", {})
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_unparsed": c["function"].get("arguments")}
            calls.append(ToolCall(c["id"], c["function"]["name"], args))
        usage = data.get("usage", {})
        return Turn(msg.get("content") or "", calls, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), msg)

    def chat(self, model, system, messages, tools) -> Turn:
        url = f"{self.BASE}/chat/completions"
        try:
            return self.parse_response(_post(url, self.to_request(model, system, messages, tools), self._headers()))
        except ProviderError as exc:
            # Some models invent tools from their training (e.g. gpt-oss calling "open_file"); Groq rejects the call
            # server-side with HTTP 400. Retry once with an explicit list of the allowed tools.
            if "tool call validation failed" not in str(exc).lower() or not tools:
                raise
            reminder = (f"{system}\n\nIMPORTANT: you may call ONLY these tools: {', '.join(t['name'] for t in tools)}. "
                        "There are no other tools (no browsing, no file access). If none fits, answer in text.")
            return self.parse_response(_post(url, self.to_request(model, reminder, messages, tools), self._headers()))


PROVIDERS = {"Google AI Studio (Gemini)": GeminiProvider, "Groq": GroqProvider}
