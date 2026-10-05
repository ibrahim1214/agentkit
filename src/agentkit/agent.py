"""The agent loop: call the model, execute requested tools, feed results back, repeat."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import LLM, ToolCall
from .tools import Tool, ToolError, ToolRegistry


class MaxStepsExceeded(RuntimeError):
    pass


@dataclass
class TraceEvent:
    step: int
    kind: str  # "model" | "tool" | "error"
    name: str
    detail: str
    seconds: float


@dataclass
class AgentResult:
    answer: str
    messages: list[dict]
    trace: list[TraceEvent] = field(default_factory=list)
    steps: int = 0

    @property
    def tools_used(self) -> list[str]:
        return [e.name for e in self.trace if e.kind == "tool"]


DEFAULT_SYSTEM = (
    "You are a careful assistant with access to tools. Use a tool when you need facts or "
    "computation you cannot do reliably yourself. When you have the answer, reply in plain text."
)


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        system: str = DEFAULT_SYSTEM,
        max_steps: int = 8,
        max_tool_output_chars: int = 4000,
        approve: Callable[[ToolCall], bool] | None = None,
    ):
        self.llm = llm
        self.registry = ToolRegistry(tools)
        self.system = system
        self.max_steps = max_steps
        self.max_out = max_tool_output_chars
        self.approve = approve  # human-in-the-loop guardrail hook

    def _run_tool(self, call: ToolCall) -> tuple[str, bool]:
        """Execute one tool call. Errors are returned to the model so it can self-correct."""
        if self.approve and not self.approve(call):
            return f"tool call '{call.name}' was denied by the user", True
        try:
            out = self.registry.get(call.name)(**call.arguments)
        except ToolError as e:
            return f"ToolError: {e}", True
        except Exception as e:  # noqa: BLE001 - tool bugs must not crash the loop
            return f"{type(e).__name__}: {e}", True
        text = out if isinstance(out, str) else json.dumps(out, default=str)
        if len(text) > self.max_out:
            text = text[: self.max_out] + f"... [truncated {len(text) - self.max_out} chars]"
        return text, False

    def run(self, task: str, history: list[dict] | None = None) -> AgentResult:
        messages: list[dict] = list(history or []) + [{"role": "user", "content": task}]
        trace: list[TraceEvent] = []
        specs = self.registry.specs()

        for step in range(1, self.max_steps + 1):
            t0 = time.perf_counter()
            resp = self.llm.complete(self.system, messages, specs)
            trace.append(TraceEvent(step, "model", "llm", f"{len(resp.tool_calls)} tool call(s)", time.perf_counter() - t0))
            messages.append({"role": "assistant", "content": resp.text, "tool_calls": resp.tool_calls})

            if not resp.tool_calls:
                return AgentResult(resp.text, messages, trace, step)

            for call in resp.tool_calls:
                t1 = time.perf_counter()
                result, failed = self._run_tool(call)
                trace.append(
                    TraceEvent(step, "error" if failed else "tool", call.name, result[:200], time.perf_counter() - t1)
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result})

        raise MaxStepsExceeded(f"no final answer after {self.max_steps} steps")
