"""Decide whether a PR can affect model behavior, so the eval gate only runs then.

``python -m evals.gate_scope --base <sha> --head <sha>`` is run by the
``Evaluation scope`` job in ``.github/workflows/pr-check.yml`` (stdlib only: the
job installs no dependencies). It lists the files the PR changes, compares the
``anthropic`` / ``openai`` SDK pins in ``pyproject.toml`` and ``uv.lock`` at the
two revisions, prints its reasoning, and writes ``run=true|false`` plus a
one-line ``reason`` to ``$GITHUB_OUTPUT``. The ``Conversation evaluations`` job
has ``if: needs.eval-scope.outputs.run == 'true'``; a skipped required check
counts as passing for branch protection.

What counts as "can change model behavior":

- ``prompts/**`` (system prompt, business context, provider addenda),
- ``tools/**`` (tool schemas and the results the model sees),
- ``utils/**`` (guards, provider adapters, numeric grounding),
- ``worker.py`` (the agent loop the harness mirrors),
- ``evals/**`` except ``*.md`` (fixtures, scoring, runner, baseline: the
  measuring instrument; its README is not),
- ``.github/workflows/pr-check.yml`` (the gate itself),
- ``pyproject.toml`` / ``uv.lock`` **only** when the resolved version of the
  ``anthropic`` or ``openai`` package, or its constraint in
  ``project.dependencies``, differs between base and head. A routine bump of
  ``boto3`` or a test tool does not run the eval.

The decision itself (``decide``) is a pure function over the changed-file list
and the two pin snapshots, so it is unit-tested without git.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from typing import Iterable

BEHAVIOR_PATH_PREFIXES = ("prompts/", "tools/", "utils/", "evals/")
BEHAVIOR_FILES = ("worker.py", ".github/workflows/pr-check.yml")
DEPENDENCY_FILES = ("pyproject.toml", "uv.lock")
MODEL_SDK_PACKAGES = ("anthropic", "openai")

_REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


@dataclass
class Decision:
    run: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "no behavior-affecting paths changed"


def normalize_path(path: str) -> str:
    path = path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.lstrip("/")


def is_behavior_path(path: str) -> bool:
    """True for a file whose change can alter what the model sees or how it is scored.

    Markdown under ``evals/`` is documentation about the harness, not part of
    it, so ``evals/README.md`` alone does not run the gate. Markdown under
    ``prompts/`` *is* the prompt and does.
    """
    path = normalize_path(path)
    if path in BEHAVIOR_FILES:
        return True
    if path.startswith("evals/") and path.endswith(".md"):
        return False
    return any(path.startswith(prefix) for prefix in BEHAVIOR_PATH_PREFIXES)


def model_sdk_pins(pyproject_text: str | None, lock_text: str | None) -> dict[str, str | None]:
    """The ``anthropic`` / ``openai`` constraint and locked version at one revision.

    Keys are ``<package>.constraint`` (the ``project.dependencies`` entry, or
    ``None`` when absent) and ``<package>.locked`` (the ``[[package]]`` version
    in ``uv.lock``, or ``None``). Unparseable input is treated as "unknown" and
    reported as ``None`` so that a broken file at either revision shows up as a
    difference rather than a silent skip.
    """
    pins: dict[str, str | None] = {}
    constraints: dict[str, str] = {}
    if pyproject_text:
        try:
            project = tomllib.loads(pyproject_text).get("project") or {}
        except tomllib.TOMLDecodeError:
            project = {}
        for requirement in project.get("dependencies") or []:
            match = _REQUIREMENT_NAME_RE.match(str(requirement))
            if match:
                constraints[match.group(1).lower().replace("_", "-")] = str(requirement).strip()
    locked: dict[str, str] = {}
    if lock_text:
        try:
            packages = tomllib.loads(lock_text).get("package") or []
        except tomllib.TOMLDecodeError:
            packages = []
        for package in packages:
            name = str(package.get("name") or "").lower().replace("_", "-")
            if name:
                locked[name] = str(package.get("version") or "")
    for package in MODEL_SDK_PACKAGES:
        pins[f"{package}.constraint"] = constraints.get(package)
        pins[f"{package}.locked"] = locked.get(package)
    return pins


def decide(
    changed_files: Iterable[str],
    base_pins: dict[str, str | None] | None = None,
    head_pins: dict[str, str | None] | None = None,
) -> Decision:
    """Pure decision: should the eval gate run for this set of changes?"""
    files = sorted({normalize_path(path) for path in changed_files if path and path.strip()})
    reasons: list[str] = []
    behavior = [path for path in files if is_behavior_path(path)]
    if behavior:
        shown = ", ".join(behavior[:8]) + (f" (+{len(behavior) - 8} more)" if len(behavior) > 8 else "")
        reasons.append(f"behavior-affecting path(s) changed: {shown}")
    pin_changed = False
    dependency_changed = [path for path in files if path in DEPENDENCY_FILES]
    if dependency_changed:
        base_pins = base_pins or {}
        head_pins = head_pins or {}
        differing = sorted(
            key
            for key in set(base_pins) | set(head_pins)
            if base_pins.get(key) != head_pins.get(key)
        )
        if differing:
            details = ", ".join(
                f"{key} {base_pins.get(key)!r} -> {head_pins.get(key)!r}" for key in differing
            )
            reasons.append(f"model SDK pin changed: {details}")
            pin_changed = True
        elif not behavior:
            reasons.append(
                f"{', '.join(dependency_changed)} changed but the "
                f"{'/'.join(MODEL_SDK_PACKAGES)} pins are identical"
            )
    return Decision(run=bool(behavior or pin_changed), reasons=reasons)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def _show(revision: str, path: str) -> str | None:
    try:
        return _git("show", f"{revision}:{path}")
    except subprocess.CalledProcessError:
        return None


def changed_files_between(base: str, head: str) -> list[str]:
    """Files the PR changes: ``git diff --name-only <merge-base(base, head)> <head>``."""
    merge_base = _git("merge-base", base, head).strip() or base
    output = _git("diff", "--name-only", merge_base, head)
    return [line for line in output.splitlines() if line.strip()]


def decide_from_git(base: str, head: str) -> Decision:
    files = changed_files_between(base, head)
    base_pins = head_pins = None
    if any(normalize_path(path) in DEPENDENCY_FILES for path in files):
        merge_base = _git("merge-base", base, head).strip() or base
        base_pins = model_sdk_pins(_show(merge_base, "pyproject.toml"), _show(merge_base, "uv.lock"))
        head_pins = model_sdk_pins(_show(head, "pyproject.toml"), _show(head, "uv.lock"))
    return decide(files, base_pins, head_pins)


def write_github_output(decision: Decision, path: str | None) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"run={'true' if decision.run else 'false'}\n")
        handle.write(f"reason={decision.reason}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True, help="Base commit (PR base branch tip)")
    parser.add_argument("--head", required=True, help="Head commit (PR head)")
    args = parser.parse_args(argv)
    decision = decide_from_git(args.base, args.head)
    print(f"conversation-evals: {'RUN' if decision.run else 'SKIP'} - {decision.reason}")
    write_github_output(decision, os.environ.get("GITHUB_OUTPUT"))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(
                "## Conversation evaluation scope\n\n"
                f"- Gate {'runs' if decision.run else 'skipped'}: {decision.reason}\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
