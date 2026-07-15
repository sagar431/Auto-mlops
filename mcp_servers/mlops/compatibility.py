"""Adapters that keep root-module imports and monkeypatch seams effective."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class RootModuleHandler:
    """Resolve a root handler at call time so monkeypatches remain observable."""

    root_module: ModuleType
    handler_name: str
    argument_aliases: dict[str, str] = field(default_factory=dict)

    def __call__(self, validated: BaseModel) -> Any:
        arguments = validated.model_dump()
        for schema_name, handler_name in self.argument_aliases.items():
            arguments[handler_name] = arguments.pop(schema_name)
        handler = getattr(self.root_module, self.handler_name)
        return handler(**arguments)
