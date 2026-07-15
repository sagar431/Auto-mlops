"""Declarative MCP tool registration, validation, dispatch, and serialization."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mcp.types import TextContent, Tool
from pydantic import BaseModel


@dataclass(frozen=True)
class ToolSpec:
    """One source of truth pairing MCP metadata, schema, and executable handler."""

    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[BaseModel], Any]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("MCP tool name must not be empty")
        if not self.description.strip():
            raise ValueError(f"MCP tool '{self.name}' description must not be empty")
        if not isinstance(self.input_model, type) or not issubclass(self.input_model, BaseModel):
            raise TypeError(f"MCP tool '{self.name}' input_model must be a Pydantic model")
        if not callable(self.handler):
            raise TypeError(f"MCP tool '{self.name}' handler must be callable")

    def as_mcp_tool(self) -> Tool:
        """Generate the public MCP catalog entry from this specification."""
        return Tool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_model.model_json_schema(),
        )


class ToolRegistry:
    """Ordered collection that rejects drift between catalog and dispatch."""

    def __init__(self, specs: list[ToolSpec] | tuple[ToolSpec, ...] = ()) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        """Return specifications in deterministic registration order."""
        return tuple(self._specs.values())

    def register(self, spec: ToolSpec) -> None:
        """Register exactly one unique, validated tool specification."""
        if spec.name in self._specs:
            raise ValueError(f"Duplicate MCP tool name: {spec.name}")
        self._specs[spec.name] = spec

    def catalog(self) -> list[Tool]:
        """Generate the ordered MCP tool catalog."""
        return [spec.as_mcp_tool() for spec in self.specs]

    async def call_mcp(self, name: str, arguments: Any) -> list[TextContent]:
        """Validate, dispatch, and serialize using the established root contract."""
        spec = self._specs.get(name)
        if spec is None:
            result = {"error": f"Unknown tool: {name}"}
            return [
                TextContent(type="text", text=json.dumps(result, indent=2, default=str))
            ]

        try:
            validated = spec.input_model(**arguments)
            result = spec.handler(validated)
            if inspect.isawaitable(result):
                result = await result
            return [
                TextContent(type="text", text=json.dumps(result, indent=2, default=str))
            ]
        except Exception as exc:
            error_result = {"error": str(exc), "tool": name}
            return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
