"""Provider-neutral view of the tool definitions.

``GITHUB_TOOLS`` / ``DATABRICKS_TOOLS`` / ``CALC_TOOLS`` stay in the Anthropic
shape (``input_schema``) because tests, evals and the prompt reference them.
This module derives ``ToolSpec`` objects from those lists so each provider can
render its own wire format; ``to_anthropic_tools`` round-trips exactly.
"""

from __future__ import annotations

from typing import Any

from tools import calc_tool, databricks_tool, github_tool
from utils.llm.types import ToolSpec

ANTHROPIC_TOOL_LISTS = (
    github_tool.GITHUB_TOOLS,
    databricks_tool.DATABRICKS_TOOLS,
    calc_tool.CALC_TOOLS,
)


def tool_spec_from_anthropic(tool: dict[str, Any]) -> ToolSpec:
    return ToolSpec(
        name=tool["name"],
        description=tool.get("description", ""),
        parameters=tool["input_schema"],
    )


def all_tool_specs() -> list[ToolSpec]:
    return [
        tool_spec_from_anthropic(tool)
        for tool_list in ANTHROPIC_TOOL_LISTS
        for tool in tool_list
    ]


def to_anthropic_tools(specs: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": spec.name, "description": spec.description, "input_schema": spec.parameters}
        for spec in specs
    ]
