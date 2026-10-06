"""tools.registry derives provider-neutral ToolSpecs that round-trip to the Anthropic shape."""

from tools import calc_tool, databricks_tool, github_tool
from tools.registry import all_tool_specs, to_anthropic_tools, tool_spec_from_anthropic
from utils.llm.anthropic_provider import tools_to_anthropic
from utils.llm.types import ToolSpec


def _anthropic_tools():
    return github_tool.GITHUB_TOOLS + databricks_tool.DATABRICKS_TOOLS + calc_tool.CALC_TOOLS


def test_specs_round_trip_to_the_anthropic_lists_exactly():
    specs = all_tool_specs()

    assert to_anthropic_tools(specs) == _anthropic_tools()
    assert tools_to_anthropic(specs) == _anthropic_tools()


def test_spec_order_and_names_match_worker_tool_order():
    assert [spec.name for spec in all_tool_specs()] == [tool["name"] for tool in _anthropic_tools()]
    assert len(set(spec.name for spec in all_tool_specs())) == len(all_tool_specs())


def test_parameters_alias_input_schema_without_copying():
    spec = tool_spec_from_anthropic(calc_tool.CALC_TOOLS[0])
    assert isinstance(spec, ToolSpec)
    assert spec.parameters is calc_tool.CALC_TOOLS[0]["input_schema"]
    assert spec.description == calc_tool.CALC_TOOLS[0]["description"]


def test_missing_description_defaults_to_empty():
    spec = tool_spec_from_anthropic({"name": "x", "input_schema": {"type": "object"}})
    assert spec.description == ""
    assert to_anthropic_tools([spec]) == [{"name": "x", "description": "", "input_schema": {"type": "object"}}]
