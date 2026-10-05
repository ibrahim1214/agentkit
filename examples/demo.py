"""Offline demo: a scripted model drives the agent loop through two tool calls."""

from agentkit import Agent, LLMResponse, ScriptedLLM, ToolCall
from agentkit.builtin import calculator, word_count

llm = ScriptedLLM(
    [
        LLMResponse("I'll compute that.", [ToolCall("c1", "calculator", {"expression": "(17 + 5) * 3"})]),
        LLMResponse("", [ToolCall("c2", "word_count", {"text": "agents call tools in a loop"})]),
        LLMResponse("(17 + 5) * 3 = 66, and the phrase has 6 words."),
    ]
)

result = Agent(llm, [calculator, word_count]).run("What is (17+5)*3, and how many words in 'agents call tools in a loop'?")
print(result.answer)
print("tools used:", result.tools_used)
for e in result.trace:
    print(f"  step {e.step} {e.kind:5} {e.name:11} {e.detail[:60]}")

# To use a real model:
#   from agentkit.llm import AnthropicLLM
#   Agent(AnthropicLLM("<model-id>"), [calculator, word_count]).run(...)
