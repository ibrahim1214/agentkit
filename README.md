# agentkit

A **minimal, typed tool-calling agent framework** in ~300 lines, with zero runtime dependencies. It shows what's actually inside agent frameworks: schema generation, the model↔tool loop, error recovery, and guardrails.

- **`@tool` decorator** — JSON Schema is generated from type hints (`str/int/float/bool`, `list[T]`, `Literal`, `Optional`) and `Args:` docstrings
- **Argument validation** before execution (missing/unexpected/mistyped args, enums, bool≠int)
- **ReAct-style loop** with multi-tool turns, history, and a `max_steps` guard
- **Self-correcting errors** — tool failures and bad calls are returned to the model as tool results instead of crashing the run
- **Guardrails** — output truncation, human-approval hook (`approve=`), AST-based calculator (never `eval`)
- **Tracing** — per-step model/tool timing in `result.trace`
- **Pluggable LLMs** — a one-method `LLM` protocol; `ScriptedLLM` for deterministic tests, `AnthropicLLM` adapter included

## Quickstart

```bash
pip install -e ".[dev]"
python examples/demo.py
pytest -q
```

```python
from agentkit import Agent, tool

@tool
def get_weather(city: str, unit: str = "c") -> str:
    """Look up the weather.

    Args:
        city: City name.
        unit: 'c' or 'f'.
    """
    return f"22°{unit.upper()} and sunny in {city}"

agent = Agent(my_llm, [get_weather])
result = agent.run("Is it nice out in Lisbon?")
print(result.answer, result.tools_used, result.steps)
```

### Real model

```bash
pip install -e ".[anthropic]"   # and set ANTHROPIC_API_KEY
```
```python
from agentkit.llm import AnthropicLLM
agent = Agent(AnthropicLLM("<model-id>"), [get_weather])
```

### Bring your own backend

```python
class MyLLM:
    def complete(self, system, messages, tools) -> LLMResponse:
        ...  # return LLMResponse(text, [ToolCall(id, name, args), ...])
```

## How the loop works

```
user task ─► LLM ─┬─ no tool calls ──► final answer
                  └─ tool calls ─► validate ─► run ─► result (or error text) ─┐
                        ▲                                                       │
                        └───────────────────────────────────────────────────────┘
                                   (repeat up to max_steps)
```

## Design decisions

- **Errors are data.** A malformed call becomes a `ToolError: ...` tool message so the model can retry — this is tested in `test_agent_surfaces_tool_errors_to_model_for_self_correction`.
- **Deterministic testing.** `ScriptedLLM` accepts callables, so tests can assert on what the model *saw* (`llm.calls`) and react to tool output.
- **Safe by default.** No `eval`, bounded exponent, output truncation, and an approval hook for side-effecting tools.

## License

MIT
