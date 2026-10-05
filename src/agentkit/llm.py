"""LLM backend protocol plus a deterministic scripted backend and an Anthropic adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class ScriptedLLM:
    """Replays a fixed list of responses. Makes agent logic unit-testable without a network.

    Entries may be ``LLMResponse`` objects or callables ``(messages) -> LLMResponse`` for
    scripts that need to react to earlier tool results.
    """

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if not self.script:
            raise RuntimeError("ScriptedLLM ran out of responses")
        nxt = self.script.pop(0)
        return nxt(messages) if callable(nxt) else nxt


class AnthropicLLM:  # pragma: no cover - requires network + key
    """Adapter for the Anthropic Messages API (``pip install 'agentkit[anthropic]'``)."""

    def __init__(self, model: str, max_tokens: int = 1024):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        api_msgs: list[dict] = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                blocks: list[dict] = ([{"type": "text", "text": m["content"]}] if m["content"] else [])
                blocks += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m["tool_calls"]
                ]
                api_msgs.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if api_msgs and api_msgs[-1]["role"] == "user" and isinstance(api_msgs[-1]["content"], list):
                    api_msgs[-1]["content"].append(block)
                else:
                    api_msgs.append({"role": "user", "content": [block]})
            else:
                api_msgs.append({"role": m["role"], "content": m["content"]})
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, messages=api_msgs, tools=tools
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return LLMResponse(text, calls)
