from typing import Literal, Optional

import pytest

from agentkit import Agent, LLMResponse, MaxStepsExceeded, ScriptedLLM, Tool, ToolCall, ToolError, ToolRegistry, tool
from agentkit.builtin import calculator, safe_eval, word_count


@tool
def greet(name: str, times: int = 1, style: Literal["formal", "casual"] = "casual", tags: Optional[list[str]] = None) -> str:
    """Greet someone.

    Args:
        name: Person to greet.
        times: How many times.
    """
    return " ".join([f"{'Good day' if style == 'formal' else 'hey'} {name}"] * times)


def test_schema_generation():
    s = greet.spec()
    props = s["input_schema"]["properties"]
    assert s["name"] == "greet" and s["description"] == "Greet someone."
    assert props["name"] == {"type": "string", "description": "Person to greet."}
    assert props["times"]["type"] == "integer"
    assert props["style"]["enum"] == ["formal", "casual"]
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert s["input_schema"]["required"] == ["name"]


def test_validation_errors():
    with pytest.raises(ToolError, match="missing required"):
        greet()
    with pytest.raises(ToolError, match="unexpected argument"):
        greet(name="a", bogus=1)
    with pytest.raises(ToolError, match="should be integer"):
        greet(name="a", times="2")
    with pytest.raises(ToolError, match="must be one of"):
        greet(name="a", style="rude")
    with pytest.raises(ToolError):
        greet(name="a", times=True)  # bool is not an int


def test_tool_call_ok():
    assert greet(name="Ana", times=2) == "hey Ana hey Ana"


def test_registry_duplicates_and_unknown():
    reg = ToolRegistry([greet])
    with pytest.raises(ValueError):
        reg.register(greet)
    with pytest.raises(ToolError, match="unknown tool"):
        reg.get("nope")
    assert len(reg.specs()) == 1


@pytest.mark.parametrize(
    "expr,val",
    [("1+2*3", 7), ("(17+5)*3", 66), ("2**10", 1024), ("-4+10", 6), ("7//2", 3), ("1/4", 0.25)],
)
def test_safe_eval_values(expr, val):
    assert safe_eval(expr) == val


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('x')", "1/0", "2**9999", "[1,2]", "1 +"])
def test_safe_eval_rejects(expr):
    with pytest.raises(ToolError):
        safe_eval(expr)


def test_agent_runs_tools_and_answers():
    llm = ScriptedLLM(
        [
            LLMResponse("", [ToolCall("a", "calculator", {"expression": "6*7"})]),
            lambda msgs: LLMResponse(f"The answer is {msgs[-1]['content']}."),
        ]
    )
    res = Agent(llm, [calculator]).run("6*7?")
    assert res.answer == "The answer is 42." and res.steps == 2
    assert res.tools_used == ["calculator"]
    assert [m["role"] for m in res.messages] == ["user", "assistant", "tool", "assistant"]


def test_agent_surfaces_tool_errors_to_model_for_self_correction():
    llm = ScriptedLLM(
        [
            LLMResponse("", [ToolCall("a", "calculator", {"expression": "1/0"})]),
            LLMResponse("", [ToolCall("b", "calculator", {"expression": "1/1"})]),
            LLMResponse("done"),
        ]
    )
    res = Agent(llm, [calculator]).run("go")
    tool_msgs = [m["content"] for m in res.messages if m["role"] == "tool"]
    assert tool_msgs[0].startswith("ToolError") and tool_msgs[1] == "1"
    assert [e.kind for e in res.trace if e.kind != "model"] == ["error", "tool"]


def test_unknown_tool_and_tool_crash_do_not_crash_loop():
    @tool
    def boom(x: int) -> str:
        """Always fails."""
        raise RuntimeError("kaput")

    llm = ScriptedLLM(
        [
            LLMResponse("", [ToolCall("1", "missing", {})]),
            LLMResponse("", [ToolCall("2", "boom", {"x": 1})]),
            LLMResponse("recovered"),
        ]
    )
    res = Agent(llm, [boom]).run("x")
    out = [m["content"] for m in res.messages if m["role"] == "tool"]
    assert "unknown tool" in out[0] and "RuntimeError: kaput" in out[1]
    assert res.answer == "recovered"


def test_max_steps_guard():
    loop = LLMResponse("", [ToolCall("x", "word_count", {"text": "a b"})])
    llm = ScriptedLLM([loop] * 10)
    with pytest.raises(MaxStepsExceeded):
        Agent(llm, [word_count], max_steps=3).run("spin")


def test_approval_hook_can_deny():
    llm = ScriptedLLM([LLMResponse("", [ToolCall("a", "calculator", {"expression": "1+1"})]), LLMResponse("ok")])
    res = Agent(llm, [calculator], approve=lambda c: False).run("x")
    assert "denied" in [m for m in res.messages if m["role"] == "tool"][0]["content"]


def test_long_tool_output_is_truncated():
    @tool
    def big() -> str:
        """Return a lot of text."""
        return "x" * 10_000

    llm = ScriptedLLM([LLMResponse("", [ToolCall("a", "big", {})]), LLMResponse("ok")])
    res = Agent(llm, [big], max_tool_output_chars=100).run("x")
    assert "truncated" in [m for m in res.messages if m["role"] == "tool"][0]["content"]


def test_history_is_preserved():
    llm = ScriptedLLM([LLMResponse("hi again")])
    res = Agent(llm).run("second", history=[{"role": "user", "content": "first"}, {"role": "assistant", "content": "hi", "tool_calls": []}])
    assert [m["content"] for m in llm.calls[0]] == ["first", "hi", "second"]
    assert res.answer == "hi again"


def test_tool_is_plain_dataclass():
    assert isinstance(calculator, Tool) and calculator.name == "calculator"
