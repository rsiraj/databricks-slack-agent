"""Deterministic ``calculate`` tool.

The model must never do arithmetic in its head: every derived number (sums,
ratios, percentages, percent change, averages, durations) goes through this
tool and the returned ``result`` is quoted verbatim.

Safety model
------------
Expressions are parsed with :mod:`ast` in ``eval`` mode and walked by an
explicit whitelist evaluator; nothing is ever passed to ``eval``/``exec``.
Only numeric constants, names bound in ``variables``, unary ``+``/``-``,
binary ``+ - * / // % **``, parentheses, and calls to a fixed function table
are accepted. Attribute access, subscripts, lambdas, comprehensions,
comparisons, boolean operators, string constants outside date helpers, and any
other node type are rejected with ``validation_failure="invalid_expression"``.
Arithmetic runs on :class:`decimal.Decimal` with a 28-digit context.
"""

from __future__ import annotations

import ast
import math
import statistics
from datetime import date, datetime
from decimal import (
    ROUND_HALF_UP,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from typing import Any

CALCULATE_TOOL_NAME = "calculate"

MAX_EXPRESSION_CHARS = 2000
MAX_AST_NODES = 200
MAX_EXPONENT = 1000
MAX_BASE_DIGITS = 30
DECIMAL_PRECISION = 28
DEFAULT_PRECISION = 6
MAX_PRECISION = 10

CALC_TOOLS: list[dict[str, Any]] = [
    {
        "name": CALCULATE_TOOL_NAME,
        "description": (
            "Deterministic arithmetic. Use for ANY derived number: sums, differences, ratios, "
            "percentages, percent change, averages, durations, unit conversions. Never compute "
            "in your head; pass the expression here and quote the returned result. Supports "
            "+ - * / // % ** and parentheses; functions abs, round, min, max, sum, avg, mean, "
            "median, sqrt, floor, ceil, pct(part, whole), pct_change(old, new), "
            "days_between('YYYY-MM-DD', 'YYYY-MM-DD'), hours_between(iso, iso), "
            "minutes_between(iso, iso). Bind numbers from tool results to names via "
            "`variables` and reference them in the expression."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Arithmetic expression, e.g. 'pct(part, whole)' or "
                        "'(new - old) / old * 100'."
                    ),
                },
                "variables": {
                    "type": "object",
                    "description": "Optional name -> number bindings used by the expression.",
                    "additionalProperties": {"type": "number"},
                },
                "precision": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": MAX_PRECISION,
                    "description": (
                        "Decimal places for result_display (default 6). Does not affect result."
                    ),
                },
            },
            "required": ["expression"],
        },
    }
]


class CalcError(Exception):
    """Raised for any rejected or failed evaluation; carries the structured error."""

    def __init__(self, message: str, failure: str = "invalid_expression"):
        super().__init__(message)
        self.message = message
        self.failure = failure


def _invalid(message: str) -> CalcError:
    return CalcError(message, "invalid_expression")


# ---------------------------------------------------------------------------
# Whitelisted functions
# ---------------------------------------------------------------------------

_DATE_FUNCTIONS = {"days_between", "hours_between", "minutes_between"}


def _to_decimal(value: Any, what: str = "value") -> Decimal:
    if isinstance(value, bool):
        raise _invalid(f"{what} must be a number, not a boolean")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise _invalid(f"{what} must be a finite number")
        return Decimal(repr(value))
    if isinstance(value, str):
        try:
            result = Decimal(value.strip())
        except InvalidOperation as exc:
            raise _invalid(f"{what} {value!r} is not a number") from exc
        if not result.is_finite():
            raise _invalid(f"{what} must be a finite number")
        return result
    raise _invalid(f"{what} must be a number")


def _flatten_numbers(args: tuple[Any, ...], name: str) -> list[Decimal]:
    values: list[Decimal] = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            values.extend(_to_decimal(item, f"{name}() argument") for item in arg)
        else:
            values.append(_to_decimal(arg, f"{name}() argument"))
    if not values:
        raise _invalid(f"{name}() requires at least one number")
    return values


def _parse_datetime(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise _invalid(f"{name}() arguments must be ISO date/time strings")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed_date = date.fromisoformat(text)
        except ValueError as exc:
            raise _invalid(
                f"{name}() could not parse {value!r}; use YYYY-MM-DD or an ISO 8601 timestamp"
            ) from exc
        parsed = datetime(parsed_date.year, parsed_date.month, parsed_date.day)
    return parsed


def _seconds_between(start: Any, end: Any, name: str) -> Decimal:
    start_dt = _parse_datetime(start, name)
    end_dt = _parse_datetime(end, name)
    if (start_dt.tzinfo is None) != (end_dt.tzinfo is None):
        raise _invalid(
            f"{name}() arguments must both be timezone-aware or both timezone-naive"
        )
    return Decimal(repr((end_dt - start_dt).total_seconds()))


def _fn_abs(*args: Any) -> Decimal:
    if len(args) != 1:
        raise _invalid("abs() takes exactly one argument")
    return abs(_to_decimal(args[0], "abs() argument"))


def _fn_round(*args: Any) -> Decimal:
    if len(args) not in (1, 2):
        raise _invalid("round() takes one or two arguments")
    value = _to_decimal(args[0], "round() argument")
    digits = 0
    if len(args) == 2:
        digits_decimal = _to_decimal(args[1], "round() digits")
        if digits_decimal != digits_decimal.to_integral_value():
            raise _invalid("round() digits must be an integer")
        digits = int(digits_decimal)
        if not -MAX_PRECISION * 2 <= digits <= MAX_PRECISION * 2:
            raise _invalid(f"round() digits must be between -{MAX_PRECISION * 2} and {MAX_PRECISION * 2}")
    quantum = Decimal(1).scaleb(-digits)
    return value.quantize(quantum, rounding=ROUND_HALF_UP)


def _fn_min(*args: Any) -> Decimal:
    return min(_flatten_numbers(args, "min"))


def _fn_max(*args: Any) -> Decimal:
    return max(_flatten_numbers(args, "max"))


def _fn_sum(*args: Any) -> Decimal:
    return sum(_flatten_numbers(args, "sum"), Decimal(0))


def _fn_avg(*args: Any) -> Decimal:
    values = _flatten_numbers(args, "avg")
    return sum(values, Decimal(0)) / Decimal(len(values))


def _fn_median(*args: Any) -> Decimal:
    values = _flatten_numbers(args, "median")
    return statistics.median(values)


def _fn_sqrt(*args: Any) -> Decimal:
    if len(args) != 1:
        raise _invalid("sqrt() takes exactly one argument")
    value = _to_decimal(args[0], "sqrt() argument")
    if value < 0:
        raise CalcError("sqrt() of a negative number is undefined", "math_domain_error")
    return value.sqrt()


def _fn_floor(*args: Any) -> Decimal:
    if len(args) != 1:
        raise _invalid("floor() takes exactly one argument")
    return Decimal(math.floor(_to_decimal(args[0], "floor() argument")))


def _fn_ceil(*args: Any) -> Decimal:
    if len(args) != 1:
        raise _invalid("ceil() takes exactly one argument")
    return Decimal(math.ceil(_to_decimal(args[0], "ceil() argument")))


def _fn_pct(*args: Any) -> Decimal:
    if len(args) != 2:
        raise _invalid("pct(part, whole) takes exactly two arguments")
    part = _to_decimal(args[0], "pct() part")
    whole = _to_decimal(args[1], "pct() whole")
    if whole == 0:
        raise CalcError("pct(): whole is zero, so the percentage is undefined", "division_by_zero")
    return part / whole * Decimal(100)


def _fn_pct_change(*args: Any) -> Decimal:
    if len(args) != 2:
        raise _invalid("pct_change(old, new) takes exactly two arguments")
    old = _to_decimal(args[0], "pct_change() old")
    new = _to_decimal(args[1], "pct_change() new")
    if old == 0:
        raise CalcError(
            "pct_change(): old value is zero, so percent change is undefined",
            "division_by_zero",
        )
    return (new - old) / old * Decimal(100)


def _fn_days_between(*args: Any) -> Decimal:
    if len(args) != 2:
        raise _invalid("days_between(start, end) takes exactly two arguments")
    return _seconds_between(args[0], args[1], "days_between") / Decimal(86400)


def _fn_hours_between(*args: Any) -> Decimal:
    if len(args) != 2:
        raise _invalid("hours_between(start, end) takes exactly two arguments")
    return _seconds_between(args[0], args[1], "hours_between") / Decimal(3600)


def _fn_minutes_between(*args: Any) -> Decimal:
    if len(args) != 2:
        raise _invalid("minutes_between(start, end) takes exactly two arguments")
    return _seconds_between(args[0], args[1], "minutes_between") / Decimal(60)


_FUNCTIONS = {
    "abs": _fn_abs,
    "round": _fn_round,
    "min": _fn_min,
    "max": _fn_max,
    "sum": _fn_sum,
    "avg": _fn_avg,
    "mean": _fn_avg,
    "median": _fn_median,
    "sqrt": _fn_sqrt,
    "floor": _fn_floor,
    "ceil": _fn_ceil,
    "pct": _fn_pct,
    "pct_change": _fn_pct_change,
    "days_between": _fn_days_between,
    "hours_between": _fn_hours_between,
    "minutes_between": _fn_minutes_between,
}

_BINARY_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_UNARY_OPERATORS = (ast.UAdd, ast.USub)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class _Evaluator:
    def __init__(self, variables: dict[str, Decimal]):
        self.variables = variables

    def evaluate(self, node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return self.evaluate(node.body)
        if isinstance(node, ast.Constant):
            return self._constant(node)
        if isinstance(node, ast.Name):
            return self._name(node)
        if isinstance(node, ast.UnaryOp):
            return self._unary(node)
        if isinstance(node, ast.BinOp):
            return self._binary(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        raise _invalid(f"Unsupported syntax: {type(node).__name__}")

    def _constant(self, node: ast.Constant) -> Decimal:
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _invalid(
                "Only numeric constants are allowed; string constants may appear only as "
                "arguments to days_between/hours_between/minutes_between"
            )
        return _to_decimal(value, "constant")

    def _name(self, node: ast.Name) -> Decimal:
        if not isinstance(node.ctx, ast.Load):
            raise _invalid("Assignment is not allowed")
        if node.id in self.variables:
            return self.variables[node.id]
        if node.id in _FUNCTIONS:
            raise _invalid(f"{node.id} is a function; call it with parentheses")
        raise _invalid(
            f"Unknown name {node.id!r}; bind it in `variables` or use a numeric literal"
        )

    def _unary(self, node: ast.UnaryOp) -> Decimal:
        if not isinstance(node.op, _UNARY_OPERATORS):
            raise _invalid(f"Unsupported unary operator: {type(node.op).__name__}")
        operand = self.evaluate(node.operand)
        return -operand if isinstance(node.op, ast.USub) else operand

    def _binary(self, node: ast.BinOp) -> Decimal:
        if not isinstance(node.op, _BINARY_OPERATORS):
            raise _invalid(f"Unsupported operator: {type(node.op).__name__}")
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod)):
            if right == 0:
                raise CalcError("Division by zero", "division_by_zero")
            if isinstance(op, ast.Div):
                return left / right
            if isinstance(op, ast.FloorDiv):
                return Decimal(math.floor(left / right))
            return left % right
        # Power: bound both operands so the evaluator cannot be used to burn CPU.
        if abs(right) > MAX_EXPONENT:
            raise _invalid(f"Exponent magnitude must not exceed {MAX_EXPONENT}")
        # adjusted() is the exponent of the leading digit, so integer digit count
        # is adjusted() + 1 (Decimal('1E+30') has one coefficient digit but 31
        # integer digits).
        if abs(left) >= 1 and left.adjusted() + 1 > MAX_BASE_DIGITS:
            raise _invalid(f"Power base must have at most {MAX_BASE_DIGITS} digits")
        if left == 0 and right < 0:
            raise CalcError("Zero cannot be raised to a negative power", "division_by_zero")
        if left < 0 and right != right.to_integral_value():
            raise CalcError(
                "A negative base with a fractional exponent has no real result",
                "math_domain_error",
            )
        return left ** right

    def _call(self, node: ast.Call) -> Decimal:
        if not isinstance(node.func, ast.Name):
            raise _invalid("Only direct calls to whitelisted functions are allowed")
        if node.keywords:
            raise _invalid("Keyword arguments are not allowed")
        name = node.func.id
        function = _FUNCTIONS.get(name)
        if function is None:
            raise _invalid(
                f"Unknown function {name!r}; allowed: {', '.join(sorted(_FUNCTIONS))}"
            )
        arguments: list[Any] = []
        for argument in node.args:
            if isinstance(argument, ast.Starred):
                raise _invalid("Starred arguments are not allowed")
            if name in _DATE_FUNCTIONS:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    arguments.append(argument.value)
                    continue
                raise _invalid(f"{name}() arguments must be string date/time literals")
            arguments.append(self.evaluate(argument))
        return function(*arguments)


def _count_nodes(tree: ast.AST) -> int:
    return sum(1 for _ in ast.walk(tree))


def _coerce_variables(raw: Any) -> dict[str, Decimal]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise _invalid("variables must be an object mapping names to numbers")
    variables: dict[str, Decimal] = {}
    for key, value in raw.items():
        name = str(key)
        if not name.isidentifier() or name in _FUNCTIONS:
            raise _invalid(f"Invalid variable name {name!r}")
        variables[name] = _to_decimal(value, f"variable {name!r}")
    return variables


def _coerce_precision(raw: Any) -> int:
    if raw is None:
        return DEFAULT_PRECISION
    if isinstance(raw, bool) or (isinstance(raw, float) and not raw.is_integer()):
        raise CalcError("precision must be an integer between 0 and 10", "invalid_precision")
    try:
        precision = int(raw)
    except (TypeError, ValueError) as exc:
        raise CalcError("precision must be an integer between 0 and 10", "invalid_precision") from exc
    if not 0 <= precision <= MAX_PRECISION:
        raise CalcError(
            f"precision must be between 0 and {MAX_PRECISION}", "invalid_precision"
        )
    return precision


def _substituted_expression(expression: str, variables: dict[str, Decimal]) -> str | None:
    if not variables:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError):
        return None

    class _Substitute(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802 (ast API)
            if node.id in variables:
                return ast.copy_location(ast.Constant(value=_plain_number(variables[node.id])), node)
            return node

    try:
        return ast.unparse(_Substitute().visit(tree))
    except Exception:  # pragma: no cover - unparse failures are cosmetic only
        return None


def _plain_number(value: Decimal) -> int | float:
    if value == value.to_integral_value():
        # Integers beyond float precision stay exact ints; floats are for fractions.
        return int(value)
    return float(value)


def _format_display(value: Decimal, precision: int) -> str:
    quantum = Decimal(1).scaleb(-precision)
    # Quantizing needs enough precision to hold every integer digit plus the
    # requested decimals; the 28-digit arithmetic context is too small for very
    # large (but legal, e.g. 10 ** 961) results.
    with localcontext() as context:
        context.prec = max(DECIMAL_PRECISION, len(value.as_tuple().digits) + value.adjusted() + precision + 4)
        rounded = value.quantize(quantum, rounding=ROUND_HALF_UP)
    text = f"{rounded:,.{precision}f}"
    if precision > 0 and "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"-0", ""}:
        text = "0"
    return text


def calculate(
    expression: Any,
    variables: Any = None,
    precision: Any = None,
) -> dict[str, Any]:
    """Evaluate ``expression`` deterministically and return the structured result."""
    echo_variables = variables if isinstance(variables, dict) else {}
    try:
        if not isinstance(expression, str) or not expression.strip():
            raise _invalid("expression must be a non-empty string")
        if len(expression) > MAX_EXPRESSION_CHARS:
            raise _invalid(f"expression must be at most {MAX_EXPRESSION_CHARS} characters")
        bound = _coerce_variables(variables)
        display_precision = _coerce_precision(precision)
        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError) as exc:
            raise _invalid(f"Could not parse expression: {exc.msg if hasattr(exc, 'msg') else exc}") from exc
        if _count_nodes(tree) > MAX_AST_NODES:
            raise _invalid(f"expression is too complex (more than {MAX_AST_NODES} nodes)")

        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            context.traps[DivisionByZero] = True
            context.traps[InvalidOperation] = True
            context.traps[Overflow] = True
            try:
                value = _Evaluator(bound).evaluate(tree)
            except DivisionByZero as exc:
                raise CalcError("Division by zero", "division_by_zero") from exc
            except Overflow as exc:
                raise CalcError("Result is too large to represent", "overflow") from exc
            except InvalidOperation as exc:
                raise CalcError("Arithmetic produced an undefined result", "math_domain_error") from exc
            value = +value  # apply context precision to the final value
        if not value.is_finite():
            raise CalcError("Result is not a finite number", "math_domain_error")
    except CalcError as exc:
        return {
            "error": exc.message,
            "validation_failure": exc.failure,
            "expression": expression if isinstance(expression, str) else str(expression),
            "variables": echo_variables,
        }

    steps: list[str] = []
    substituted = _substituted_expression(expression, bound)
    if substituted is not None and substituted != expression:
        steps.append(substituted)
    return {
        "expression": expression,
        "variables": echo_variables,
        "result": _plain_number(value),
        "result_display": _format_display(value, display_precision),
        "steps": steps,
    }


def dispatch(name: str, inp: dict[str, Any]) -> dict[str, Any]:
    """Tool dispatcher with the same signature as the other tool modules."""
    if name != CALCULATE_TOOL_NAME:
        return {"error": f"Unknown calc tool: {name}"}
    if not isinstance(inp, dict):
        return {
            "error": "calculate input must be an object",
            "validation_failure": "invalid_expression",
            "expression": "",
            "variables": {},
        }
    return calculate(inp.get("expression"), inp.get("variables"), inp.get("precision"))
