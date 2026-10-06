"""Path filter that decides whether the conversation-eval gate runs for a PR."""

import pytest

from evals import gate_scope
from evals.gate_scope import decide, is_behavior_path, model_sdk_pins

PYPROJECT = """
[project]
name = "x"
dependencies = [
    "anthropic>=1.3,<2",
    "openai>=3.8,<4",
    "boto3>=1.40",
]
"""

LOCK = """
version = 1

[[package]]
name = "anthropic"
version = "1.4.0"

[[package]]
name = "boto3"
version = "1.40.0"

[[package]]
name = "openai"
version = "3.8.0"
"""


@pytest.mark.parametrize(
    "path",
    [
        "prompts/core/system.md",
        "prompts/providers/openai.md",
        "tools/databricks_tool.py",
        "utils/numeric_guard.py",
        "utils/llm/openai_provider.py",
        "worker.py",
        "evals/fixtures/new_scenario.json",
        "evals/scoring.py",
        "evals/baseline.json",
        ".github/workflows/pr-check.yml",
        "./prompts/core/system.md",
    ],
)
def test_behavior_paths(path):
    assert is_behavior_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "docs/SETUP.md",
        "tests/test_scoring.py",
        "infrastructure/main.tf",
        "receiver.py",
        "scripts/export_sessions.py",
        ".github/workflows/deploy.yml",
        ".github/workflows/model-comparison.yml",
        "pyproject.toml",
        "uv.lock",
        "Dockerfile",
        "toolsmith/notes.md",  # prefix match must be on the directory, not the string
        "workers.py",
        "evals/README.md",  # documentation about the harness, not part of it
    ],
)
def test_non_behavior_paths(path):
    assert not is_behavior_path(path)


def test_model_sdk_pins_reads_constraints_and_locked_versions():
    pins = model_sdk_pins(PYPROJECT, LOCK)
    assert pins == {
        "anthropic.constraint": "anthropic>=1.3,<2",
        "anthropic.locked": "1.4.0",
        "openai.constraint": "openai>=3.8,<4",
        "openai.locked": "3.8.0",
    }


def test_model_sdk_pins_tolerates_missing_or_broken_files():
    assert model_sdk_pins(None, None) == {
        "anthropic.constraint": None,
        "anthropic.locked": None,
        "openai.constraint": None,
        "openai.locked": None,
    }
    broken = model_sdk_pins("not = [toml", "[[package]]\nname = 'anthropic'\nversion = ")
    assert all(value is None for value in broken.values())


def test_docs_only_change_skips_the_gate():
    decision = decide(["README.md", "docs/SETUP.md", "tests/test_x.py"])
    assert decision.run is False
    assert decision.reason == "no behavior-affecting paths changed"


def test_prompt_change_runs_the_gate_with_the_path_in_the_reason():
    decision = decide(["prompts/core/system.md", "README.md"])
    assert decision.run is True
    assert "prompts/core/system.md" in decision.reason
    assert "README.md" not in decision.reason


def test_dependency_bump_of_unrelated_package_skips_the_gate():
    head_lock = LOCK.replace('name = "boto3"\nversion = "1.40.0"', 'name = "boto3"\nversion = "1.41.0"')
    base = model_sdk_pins(PYPROJECT, LOCK)
    head = model_sdk_pins(PYPROJECT, head_lock)

    decision = decide(["uv.lock"], base, head)

    assert decision.run is False
    assert "pins are identical" in decision.reason


def test_anthropic_lock_bump_runs_the_gate():
    head_lock = LOCK.replace('name = "anthropic"\nversion = "1.4.0"', 'name = "anthropic"\nversion = "1.5.0"')
    decision = decide(["uv.lock"], model_sdk_pins(PYPROJECT, LOCK), model_sdk_pins(PYPROJECT, head_lock))

    assert decision.run is True
    assert "anthropic.locked '1.4.0' -> '1.5.0'" in decision.reason


def test_openai_constraint_change_in_pyproject_runs_the_gate():
    head_pyproject = PYPROJECT.replace('"openai>=3.8,<4"', '"openai>=3.9,<4"')
    decision = decide(
        ["pyproject.toml", "uv.lock"],
        model_sdk_pins(PYPROJECT, LOCK),
        model_sdk_pins(head_pyproject, LOCK),
    )

    assert decision.run is True
    assert "openai.constraint" in decision.reason


def test_dependency_files_without_pin_snapshots_are_treated_as_unchanged_pins():
    # Callers that cannot read the two revisions pass no pins; the decision is
    # then driven by the other paths only.
    assert decide(["uv.lock"]).run is False
    assert decide(["uv.lock", "tools/x.py"]).run is True


def test_removed_sdk_pin_counts_as_a_change():
    head_lock = LOCK.replace('[[package]]\nname = "openai"\nversion = "3.8.0"\n', "")
    decision = decide(["uv.lock"], model_sdk_pins(PYPROJECT, LOCK), model_sdk_pins(PYPROJECT, head_lock))
    assert decision.run is True
    assert "openai.locked '3.8.0' -> None" in decision.reason


def test_github_output_and_cli_wiring(tmp_path, monkeypatch, capsys):
    output = tmp_path / "out.txt"
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(
        gate_scope, "decide_from_git", lambda base, head: gate_scope.Decision(True, [f"{base}..{head} touched prompts/"])
    )

    assert gate_scope.main(["--base", "abc", "--head", "def"]) == 0

    assert output.read_text() == "run=true\nreason=abc..def touched prompts/\n"
    assert "Gate runs: abc..def touched prompts/" in summary.read_text()
    assert "conversation-evals: RUN" in capsys.readouterr().out


def test_gate_scope_is_stdlib_only():
    """The scope job installs no dependencies; the module must import without them."""
    import ast
    import sys
    from pathlib import Path

    source = Path(gate_scope.__file__).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names)
    assert imported <= stdlib, imported - stdlib
