"""Deterministic ``calculate`` tool: accepted arithmetic and every rejection class."""

import json

import pytest

from tools import calc_tool
from tools.calc_tool import CALC_TOOLS, dispatch


def calc(expression, variables=None, precision=None):
    payload = {"expression": expression}
    if variables is not None:
        payload["variables"] = variables
    if precision is not None:
        payload["precision"] = precision
    return dispatch("calculate", payload)


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


def test_tool_definition_contract():
    assert len(CALC_TOOLS) == 1
    tool = CALC_TOOLS[0]
    assert tool["name"] == "calculate" == calc_tool.CALCULATE_TOOL_NAME
    assert tool["description"].startswith("Deterministic arithmetic.")
    assert "Never compute in your head" in tool["description"]
    schema = tool["input_schema"]
    assert schema["required"] == ["expression"]
    assert set(schema["properties"]) == {"expression", "variables", "precision"}
    assert schema["properties"]["precision"]["minimum"] == 0
    assert schema["properties"]["precision"]["maximum"] == 10
    json.dumps(tool)  # must be serialisable for the Anthropic API


def test_dispatch_rejects_unknown_tool_and_non_object_input():
    assert "error" in dispatch("other", {})
    assert dispatch("calculate", "1 + 1")["validation_failure"] == "invalid_expression"


# ---------------------------------------------------------------------------
# Accepted arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "result", "display"),
    [
        ("1 + 2 * 3", 7, "7"),
        ("(1 + 2) * 3", 9, "9"),
        ("10 - 4 - 3", 3, "3"),
        ("7 / 2", 3.5, "3.5"),
        ("7 // 2", 3, "3"),
        ("-7 // 2", -4, "-4"),
        ("7 % 3", 1, "1"),
        ("2 ** 10", 1024, "1,024"),
        ("-3", -3, "-3"),
        ("+3", 3, "3"),
        ("0.1 + 0.2", 0.3, "0.3"),
        ("1234567.891 * 1000", 1234567891, "1,234,567,891"),
        ("abs(-4.5)", 4.5, "4.5"),
        ("round(2.5)", 3, "3"),
        ("round(2.345, 2)", 2.35, "2.35"),
        ("min(3, 1, 2)", 1, "1"),
        ("max(3, 1, 2)", 3, "3"),
        ("sum(1, 2, 3.5)", 6.5, "6.5"),
        ("avg(1, 2, 3, 4)", 2.5, "2.5"),
        ("mean(2, 4)", 3, "3"),
        ("median(1, 5, 3)", 3, "3"),
        ("median(1, 2, 3, 4)", 2.5, "2.5"),
        ("sqrt(16)", 4, "4"),
        ("floor(2.7)", 2, "2"),
        ("ceil(2.1)", 3, "3"),
        ("floor(-2.1)", -3, "-3"),
    ],
)
def test_basic_operations(expression, result, display):
    out = calc(expression)

    assert "error" not in out, out
    assert out["expression"] == expression
    assert out["variables"] == {}
    assert out["result"] == result
    assert isinstance(out["result"], (int, float))
    assert out["result_display"] == display
    assert out["steps"] == []


def test_decimal_arithmetic_avoids_binary_float_artifacts():
    assert calc("0.1 + 0.2")["result_display"] == "0.3"
    assert calc("1.1 * 3")["result"] == 3.3
    assert calc("100 * 1.1")["result_display"] == "110"


@pytest.mark.parametrize(
    ("expression", "variables", "result", "display"),
    [
        ("pct(part, whole)", {"part": 35, "whole": 140}, 25, "25"),
        ("pct(15000, 1284000)", None, pytest.approx(1.1682243), "1.168224"),
        ("pct_change(old, new)", {"old": 200, "new": 250}, 25, "25"),
        ("pct_change(250, 200)", None, -20, "-20"),
        ("pct_change(old, new)", {"old": 1284000, "new": 1269000}, pytest.approx(-1.168224), "-1.168224"),
    ],
)
def test_percent_helpers(expression, variables, result, display):
    out = calc(expression, variables)

    assert "error" not in out, out
    assert out["result"] == result
    assert out["result_display"] == display


@pytest.mark.parametrize(
    ("expression", "result"),
    [
        ("days_between('2026-01-01', '2026-03-01')", 59),
        ("days_between('2026-03-01', '2026-01-01')", -59),
        ("days_between('2024-02-28', '2024-03-01')", 2),  # leap year
        ("hours_between('2026-01-01T00:00:00Z', '2026-01-02T06:30:00Z')", 30.5),
        ("hours_between('2026-01-01T00:00:00+00:00', '2026-01-01T02:00:00-01:00')", 3),
        ("minutes_between('2026-01-01T00:00:00', '2026-01-01T01:15:00')", 75),
        ("minutes_between('2026-01-01 08:00:00', '2026-01-01 08:00:30')", 0.5),
        ("days_between('2026-01-01', '2026-01-31') * 24", 720),
    ],
)
def test_date_and_time_differences(expression, result):
    out = calc(expression)

    assert "error" not in out, out
    assert out["result"] == result


def test_date_functions_reject_mixed_timezone_awareness_and_bad_dates():
    mixed = calc("hours_between('2026-01-01T00:00:00Z', '2026-01-01T02:00:00')")
    assert mixed["validation_failure"] == "invalid_expression"
    assert "timezone" in mixed["error"]

    bad = calc("days_between('2026-13-01', '2026-01-01')")
    assert bad["validation_failure"] == "invalid_expression"
    assert "2026-13-01" in bad["error"]

    non_literal = calc("days_between(start, end)", {"start": 1, "end": 2})
    assert non_literal["validation_failure"] == "invalid_expression"


def test_variables_are_substituted_and_echoed():
    variables = {"revenue": 100.5, "tax": 9.5, "count": 4}

    out = calc("sum(revenue, tax) / count", variables)

    assert out["result"] == 27.5
    assert out["result_display"] == "27.5"
    assert out["variables"] == variables
    assert out["steps"] == ["sum(100.5, 9.5) / 4"]


def test_variables_accept_numeric_strings_and_reject_non_numbers():
    assert calc("x * 2", {"x": "21"})["result"] == 42
    bad = calc("x * 2", {"x": "twenty"})
    assert bad["validation_failure"] == "invalid_expression"
    assert calc("x", {"x": True})["validation_failure"] == "invalid_expression"
    assert calc("x", {"x": None})["validation_failure"] == "invalid_expression"
    assert calc("x", {"not an identifier": 1})["validation_failure"] == "invalid_expression"
    assert calc("x", "x=1")["validation_failure"] == "invalid_expression"


def test_unknown_name_is_rejected_with_hint():
    out = calc("x + 1")

    assert out["validation_failure"] == "invalid_expression"
    assert "x" in out["error"] and "variables" in out["error"]
    assert out["expression"] == "x + 1"
    assert out["variables"] == {}


def test_function_names_cannot_be_shadowed_or_used_as_values():
    assert calc("abs", {})["validation_failure"] == "invalid_expression"
    assert calc("abs(1)", {"abs": 5})["validation_failure"] == "invalid_expression"


# ---------------------------------------------------------------------------
# Precision / display
# ---------------------------------------------------------------------------


def test_precision_controls_display_only():
    out = calc("2 / 3")
    assert out["result"] == pytest.approx(0.6666666667)
    assert out["result_display"] == "0.666667"  # default precision 6

    assert calc("2 / 3", precision=2)["result_display"] == "0.67"
    assert calc("2 / 3", precision=0)["result_display"] == "1"
    assert calc("2 / 3", precision=10)["result_display"] == "0.6666666667"
    assert calc("2 / 3", precision=2)["result"] == out["result"]


def test_display_uses_thousands_separators_and_half_up_rounding():
    assert calc("1234567.5", precision=0)["result_display"] == "1,234,568"
    assert calc("0.125", precision=2)["result_display"] == "0.13"
    assert calc("-1234.5678", precision=2)["result_display"] == "-1,234.57"
    assert calc("1000000 * 1000000")["result_display"] == "1,000,000,000,000"


@pytest.mark.parametrize("precision", [-1, 11, 2.5, "two", True])
def test_invalid_precision_is_rejected(precision):
    out = calc("1 + 1", precision=precision)

    assert out["validation_failure"] == "invalid_precision"


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "__import__('os').system('true')",
        "().__class__",
        "().__class__.__bases__[0].__subclasses__()",
        "(1).__class__",
        "abs.__call__(1)",
        "x.real",
        "x[0]",
        "[1, 2][0]",
        "(lambda: 1)()",
        "[i for i in range(3)]",
        "{i: i for i in range(3)}",
        "(i for i in range(3))",
        "1 < 2",
        "1 == 1",
        "1 and 2",
        "1 or 2",
        "not 1",
        "1 if 2 else 3",
        "'a'",
        "'a' + 'b'",
        "f'{1}'",
        "b'1'",
        "True",
        "None",
        "1 << 2",
        "1 & 2",
        "1 | 2",
        "~1",
        "open('x')",
        "eval('1')",
        "exec('1')",
        "getattr(abs, 'x')",
        "range(3)",
        "sum([1, 2])",
        "sum(*[1, 2])",
        "round(1, digits=2)",
        "x := 1",
        "1 +",
        "",
        "   ",
    ],
)
def test_disallowed_syntax_is_rejected(expression):
    out = calc(expression, {"x": 1})

    assert out["validation_failure"] == "invalid_expression", out
    assert "error" in out
    assert "result" not in out


def test_no_builtins_or_dunder_access_reachable():
    # Even if bound as a variable name, dunder names never resolve to Python objects.
    out = calc("__builtins__", {})
    assert out["validation_failure"] == "invalid_expression"
    out = calc("__import__", {})
    assert out["validation_failure"] == "invalid_expression"


def test_non_string_expression_is_rejected():
    assert dispatch("calculate", {"expression": 5})["validation_failure"] == "invalid_expression"
    assert dispatch("calculate", {})["validation_failure"] == "invalid_expression"


@pytest.mark.parametrize(
    "expression",
    ["1 / 0", "1 // 0", "1 % 0", "x / (y - y)", "0 ** -1"],
)
def test_division_by_zero_is_a_structured_error(expression):
    out = calc(expression, {"x": 1, "y": 3})

    assert out["validation_failure"] == "division_by_zero"
    assert "error" in out
    assert "result" not in out


def test_percent_helpers_report_zero_denominators():
    assert calc("pct(1, 0)")["validation_failure"] == "division_by_zero"
    assert calc("pct_change(0, 5)")["validation_failure"] == "division_by_zero"


def test_huge_exponent_and_wide_base_are_rejected():
    assert calc("2 ** 1001")["validation_failure"] == "invalid_expression"
    assert calc("2 ** -1001")["validation_failure"] == "invalid_expression"
    boundary = calc("2 ** 1000")
    assert "error" not in boundary
    # 28-digit Decimal context: correct to 28 significant digits, not bit-exact.
    assert boundary["result"] == pytest.approx(2**1000, rel=1e-26)
    wide_base = "1" * 31
    assert calc(f"{wide_base} ** 2")["validation_failure"] == "invalid_expression"
    assert calc("(10 ** 30) ** 2")["validation_failure"] == "invalid_expression"


def test_fractional_and_negative_powers_follow_real_arithmetic():
    assert calc("4 ** 0.5")["result"] == 2
    assert calc("2 ** -1")["result"] == 0.5
    assert calc("(-8) ** (1 / 3)")["validation_failure"] == "math_domain_error"
    assert calc("sqrt(-1)")["validation_failure"] == "math_domain_error"


def test_expression_length_and_node_count_limits():
    too_long = "1 + " * 600 + "1"
    assert len(too_long) > 2000
    assert calc(too_long)["validation_failure"] == "invalid_expression"

    too_many_nodes = " + ".join(["1"] * 150)
    assert len(too_many_nodes) <= 2000
    out = calc(too_many_nodes)
    assert out["validation_failure"] == "invalid_expression"
    assert "complex" in out["error"]

    fine = " + ".join(["1"] * 60)
    assert calc(fine)["result"] == 60


def test_error_payloads_echo_the_input():
    out = calc("1 / 0", {"a": 1})

    assert out == {
        "error": "Division by zero",
        "validation_failure": "division_by_zero",
        "expression": "1 / 0",
        "variables": {"a": 1},
    }


def test_results_are_json_serialisable():
    for expression in ("1 / 3", "2 ** 1000", "days_between('2026-01-01', '2026-01-02')"):
        json.dumps(calc(expression))
