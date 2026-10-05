"""Tool definitions: JSON Schema is generated from Python type hints and docstrings."""

from __future__ import annotations

import inspect
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable, get_args, get_origin, get_type_hints


class ToolError(Exception):
    """Raised for invalid tool arguments or unknown tools. Surfaced to the model, not the caller."""


_PRIMITIVES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _schema_for(tp: Any) -> dict:
    origin = get_origin(tp)
    if tp in _PRIMITIVES:
        return {"type": _PRIMITIVES[tp]}
    if origin in (list, typing.List):
        (inner,) = get_args(tp) or (str,)
        return {"type": "array", "items": _schema_for(inner)}
    if origin is typing.Literal:
        return {"enum": list(get_args(tp))}
    if origin in (typing.Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return _schema_for(args[0])
        return {"anyOf": [_schema_for(a) for a in args]}
    if tp is dict or origin is dict:
        return {"type": "object"}
    return {"type": "string"}


def _param_docs(doc: str) -> dict[str, str]:
    """Parse a Google-style ``Args:`` section."""
    out: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        s = line.strip()
        if s.lower() in {"args:", "arguments:", "parameters:"}:
            in_args = True
            continue
        if in_args:
            if not s:
                break
            if ":" in s:
                name, _, desc = s.partition(":")
                out[name.split("(")[0].strip()] = desc.strip()
    return out


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    @classmethod
    def from_function(cls, fn: Callable[..., Any], name: str | None = None) -> "Tool":
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)
        doc = inspect.getdoc(fn) or ""
        summary = doc.split("\n\n")[0].split("Args:")[0].strip() or fn.__name__
        pdocs = _param_docs(doc)
        props: dict[str, dict] = {}
        required: list[str] = []
        for pname, p in sig.parameters.items():
            schema = _schema_for(hints.get(pname, str))
            if pname in pdocs:
                schema["description"] = pdocs[pname]
            props[pname] = schema
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        return cls(
            name or fn.__name__,
            summary,
            {"type": "object", "properties": props, "required": required, "additionalProperties": False},
            fn,
        )

    def validate(self, args: dict) -> dict:
        props = self.parameters["properties"]
        for r in self.parameters["required"]:
            if r not in args:
                raise ToolError(f"{self.name}: missing required argument '{r}'")
        for k, v in args.items():
            if k not in props:
                raise ToolError(f"{self.name}: unexpected argument '{k}'")
            t = props[k].get("type")
            ok = (
                t is None
                or (t == "string" and isinstance(v, str))
                or (t == "integer" and isinstance(v, int) and not isinstance(v, bool))
                or (t == "number" and isinstance(v, (int, float)) and not isinstance(v, bool))
                or (t == "boolean" and isinstance(v, bool))
                or (t == "array" and isinstance(v, list))
                or (t == "object" and isinstance(v, dict))
            )
            if not ok:
                raise ToolError(f"{self.name}: argument '{k}' should be {t}, got {type(v).__name__}")
            if "enum" in props[k] and v not in props[k]["enum"]:
                raise ToolError(f"{self.name}: '{k}' must be one of {props[k]['enum']}")
        return args

    def __call__(self, **kwargs: Any) -> Any:
        return self.fn(**self.validate(kwargs))

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None):
    """Decorator: ``@tool`` turns a typed function into a :class:`Tool`."""

    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool.from_function(f, name)

    return wrap(fn) if fn else wrap


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"duplicate tool name: {t.name}")
        self._tools[t.name] = t

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"unknown tool '{name}'. available: {sorted(self._tools)}")
        return self._tools[name]

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)
