"""Replay real-shaped sessions through the pipeline and assert API validity.

This is the "does it survive actual conversation shapes" check that pure unit
tests can't give. It runs the same two transforms the worker applies before
every API call — ``_strip_invalid_leading_messages`` (sanitization) then
``_prepare_messages_for_api`` (compaction) — over each session and asserts the
result is a valid Anthropic message list:

  - the first message is a ``tool_result``-free user message (or the list is empty);
  - every ``tool_use`` has exactly one matching ``tool_result`` and vice versa;
  - the transforms don't mutate their input and don't raise.

Two sources:
  1. ``tests/fixtures/synthetic_sessions.json`` — committed, fake content, so the
     invariants are always exercised in CI with zero real data.
  2. ``scripts/exports/catalog.json`` — your real local catalog, if present. Never
     committed (gitignored); when available locally it lets you validate the
     transforms against the real corpus. Skipped cleanly when absent.
"""

import copy
import json
from pathlib import Path

import pytest

from utils.compaction import _prepare_messages_for_api
from utils.sessions import _is_invalid_first_message, _strip_invalid_leading_messages

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_sessions.json"
LOCAL_CATALOG = Path(__file__).resolve().parent.parent / "scripts" / "exports" / "catalog.json"


def _tool_ids(messages: list[dict]) -> tuple[list[str], list[str]]:
    """Return (tool_use_ids, tool_result_ids) in order of appearance."""
    use_ids, result_ids = [], []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                use_ids.append(block.get("id"))
            elif block.get("type") == "tool_result":
                result_ids.append(block.get("tool_use_id"))
    return use_ids, result_ids


def _assert_api_valid(thread_ts: str, raw_messages: list[dict]) -> None:
    original = copy.deepcopy(raw_messages)

    sanitized = _strip_invalid_leading_messages(list(raw_messages))
    if sanitized:
        assert not _is_invalid_first_message(sanitized[0]), (
            f"{thread_ts}: first message still invalid after sanitization"
        )

    prepared = _prepare_messages_for_api(sanitized)

    use_ids, result_ids = _tool_ids(prepared)
    assert sorted(use_ids) == sorted(result_ids), (
        f"{thread_ts}: tool_use / tool_result IDs are unbalanced after prepare "
        f"(uses={use_ids}, results={result_ids})"
    )
    assert len(use_ids) == len(set(use_ids)), f"{thread_ts}: duplicate tool_use ids"

    # Transforms must not mutate their input.
    assert raw_messages == original, f"{thread_ts}: input was mutated"


def _load(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    # catalog.json is a {thread_ts: record} map; the fixture is a list.
    if isinstance(data, dict):
        return list(data.values())
    return data


def _ids(sessions: list[dict]) -> list[str]:
    return [s.get("thread_ts", f"idx-{i}") for i, s in enumerate(sessions)]


_SYNTHETIC = _load(FIXTURE)


@pytest.mark.parametrize("session", _SYNTHETIC, ids=_ids(_SYNTHETIC))
def test_synthetic_sessions_are_api_valid(session):
    _assert_api_valid(session.get("thread_ts", "?"), session.get("messages", []))


@pytest.mark.skipif(not LOCAL_CATALOG.exists(), reason="no local catalog (scripts/exports/catalog.json)")
def test_local_catalog_sessions_are_api_valid():
    sessions = _load(LOCAL_CATALOG)
    assert sessions, "local catalog is present but empty"
    for session in sessions:
        _assert_api_valid(session.get("thread_ts", "?"), session.get("messages", []))
