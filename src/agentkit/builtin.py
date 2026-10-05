"""Safe built-in tools."""

from __future__ import annotations

import ast
import operator as op
from datetime import datetime, timezone

from .tools import ToolError, tool

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv}
_UN = {ast.USub: op.neg, ast.UAdd: op.pos}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ToolError("exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_eval(node.operand))
    raise ToolError("only arithmetic expressions are allowed")


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic via the AST. Never uses ``eval``."""
    try:
        return _eval(ast.parse(expression, mode="eval").body)
    except ZeroDivisionError as e:
        raise ToolError("division by zero") from e
    except SyntaxError as e:
        raise ToolError(f"invalid expression: {e.msg}") from e


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '(3 + 4) * 2 ** 3'.

    Args:
        expression: Arithmetic using + - * / // % ** and parentheses.
    """
    v = safe_eval(expression)
    return str(int(v)) if float(v).is_integer() else f"{v:.10g}"


@tool
def current_time(tz: str = "UTC") -> str:
    """Return the current time as an ISO-8601 string (UTC only).

    Args:
        tz: Only 'UTC' is supported.
    """
    if tz.upper() != "UTC":
        raise ToolError("only UTC is supported")
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text.

    Args:
        text: The text to count.
    """
    return len(text.split())
