"""Message-list compaction for the model API call.

Extracted from ``worker.py`` so it can be imported without pulling in the
worker's heavy dependency chain (boto3 clients, anthropic, slack_sdk). This
module deliberately depends only on the stdlib (``json``, ``copy``) so that
offline tooling (``scripts/export_sessions.py``) and the test suite can import
these functions directly — no AST/exec hacks, no network at import time.

``worker.py`` re-imports both functions, so behaviour is unchanged at runtime.
"""

from __future__ import annotations

import copy
import json

# Opaque per-provider reasoning carried across turns by non-Anthropic
# providers (see utils/llm/types.py). Kept as a literal so this module stays
# stdlib-only.
PROVIDER_REASONING_BLOCK_TYPE = "provider_reasoning"


_JOB_RUN_LISTING_SUMMARY_KEYS = (
    "date",
    "scope",
    "timezone",
    "active_only",
    "fetched_run_count",
    "truncated",
)
_DESCRIBE_COMMENT_MAX_CHARS = 200
_JOB_RUN_LISTING_RUN_KEYS = (
    "run_name",
    "job_id",
    "run_id",
    "life_cycle_state",
    "result_state",
    "start_time",
    "end_time",
    "setup_duration_ms",
    "execution_duration_ms",
    "cleanup_duration_ms",
)


def _compact_job_run_listing(data: dict) -> dict:
    """Keep the summary fields and per-run essentials of a ``databricks_list_job_runs`` result."""
    compact = {key: data.get(key) for key in _JOB_RUN_LISTING_SUMMARY_KEYS if key in data}
    compact["runs"] = [
        {key: run.get(key) for key in _JOB_RUN_LISTING_RUN_KEYS if run.get(key) is not None}
        for run in data.get("runs", [])
        if isinstance(run, dict)
    ]
    return compact


def _trim_comment(value):
    """Cut a long free-text comment; column names/types are the load-bearing part."""
    if isinstance(value, str) and len(value) > _DESCRIBE_COMMENT_MAX_CHARS:
        return value[:_DESCRIBE_COMMENT_MAX_CHARS] + "…[truncated]"
    return value


def _compact_describe_object(data: dict) -> dict:
    """Keep the whole structured ``databricks_describe_object`` result, trimming long comments.

    ``object`` / ``column_count`` / ``columns`` are data the model grounded its
    earlier answer on, so they stay intact and the count still equals
    ``len(columns)``. Only comment prose longer than 200 characters is cut.
    """
    compact = dict(data)
    obj = data.get("object")
    if isinstance(obj, dict):
        compact["object"] = {
            **obj,
            "comment": _trim_comment(obj.get("comment")),
        }
    compact["columns"] = [
        {**column, "comment": _trim_comment(column.get("comment"))}
        if isinstance(column, dict)
        else column
        for column in data.get("columns", [])
    ]
    return compact


def _compact_tool_result_content(raw_content: str) -> str:
    """Produce a compact version of a tool_result's JSON content.

    Keeps the semantics (what was queried, what came back) but strips
    boilerplate: excess sample rows, URLs, SHAs, large patches, and
    file-content middles.  Falls back to simple truncation for
    unrecognised shapes.
    """
    try:
        data = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        if len(raw_content) > 1000:
            return raw_content[:1000] + "…[truncated]"
        return raw_content

    if not isinstance(data, dict):
        return raw_content

    # Error responses are already small
    if "error" in data and len(raw_content) < 500:
        return raw_content

    # Structured object description (databricks_describe_object) — data-bearing,
    # so keep the object header, the authoritative column_count, and every typed
    # column; only long comment prose is trimmed. Must precede the SQL branch:
    # this shape has "columns" but no "sample_rows" and would otherwise hit the
    # generic truncation fallback and be cut mid-JSON.
    if (
        "column_count" in data
        and isinstance(data.get("columns"), list)
        and isinstance(data.get("object"), dict)
    ):
        return json.dumps(_compact_describe_object(data), default=str)

    # SQL results — pass through entirely.  The model's prior response was
    # grounded in these rows; altering them retroactively would make the
    # conversation inconsistent.  Token savings here come from the model
    # using tighter LIMITs up front, not from post-hoc stripping.
    if "sample_rows" in data and "columns" in data:
        return raw_content

    # SQL still running — already tiny
    if data.get("status") in ("PENDING", "RUNNING") and "statement_id" in data:
        return raw_content

    # GitHub file content — pass through entirely. Same reasoning as SQL
    # results above: the model already grounded its earlier answer on this
    # exact content; rewriting it retroactively (a) makes the thread
    # inconsistent and (b) caused the model to re-pull the same file in a
    # later turn (observed in production), which costs more in input
    # tokens than just keeping the original content resident. The
    # tool-layer 60 KB cap in `github_tool._truncate` is the real ceiling
    # here; this compactor isn't where file-content bytes need to be
    # saved.
    if "content" in data and "path" in data and "ref" in data:
        return raw_content

    # GitHub search — keep paths, drop sha / html_url
    if "total_count" in data and "items" in data:
        return json.dumps({
            "total_count": data["total_count"],
            "items": [{"path": it.get("path")} for it in data.get("items", [])],
        }, default=str)

    # GitHub commits — keep date + first-line message
    if "commits" in data and "path" in data:
        return json.dumps({
            "path": data["path"],
            "commits": [
                {"date": c.get("date"), "message": c.get("message")}
                for c in data.get("commits", [])
            ],
        }, default=str)

    # GitHub compare — keep file stats, drop patches entirely
    if "files" in data and "ahead_by" in data:
        return json.dumps({
            "ahead_by": data.get("ahead_by"),
            "total_commits": data.get("total_commits"),
            "files": [
                {"filename": f.get("filename"), "status": f.get("status"),
                 "additions": f.get("additions", 0), "deletions": f.get("deletions", 0)}
                for f in data.get("files", [])
            ],
        }, default=str)

    # Workspace job-run listing (databricks_list_job_runs) — keep the date /
    # scope / counts and a compact row per run (name, ids, states, timings);
    # drop run page URLs, trigger metadata, epoch duplicates of the ISO
    # timestamps, and the interpretation note. Emitting real JSON here matters:
    # the generic fallback below would cut the run list mid-string.
    if "runs" in data and ("scope" in data or "interpretation_note" in data):
        return json.dumps(_compact_job_run_listing(data), default=str)

    # Recent job runs — keep the summary, drop individual run rows
    if "runs" in data and "successful_run_summary" in data:
        return json.dumps({
            "job_id": data.get("job_id"),
            "fetched_run_count": data.get("fetched_run_count"),
            "successful_run_summary": data.get("successful_run_summary"),
        }, default=str)

    # Job run status / cancel — already small
    if "life_cycle_state" in data or "run_id" in data:
        return raw_content

    # Unrecognised shape — truncate if large
    if len(raw_content) > 1000:
        return raw_content[:1000] + "…[truncated]"
    return raw_content


def _prepare_messages_for_api(messages: list[dict], current_run_start: int | None = None) -> list[dict]:
    """Build a token-efficient copy of the message list for the API call.

    Three optimisations, scoped so the current agent run always has full fidelity:

      1. Thinking blocks stripped from assistant turns of *previous* agent
         runs only. Every assistant turn from ``current_run_start`` onward
         keeps its thinking; every assistant turn before it loses it,
         including the previous run's final turn. This is a prompt-cache
         requirement, not a model one: the request prefix must be
         byte-identical between iterations of one run for the conversation to
         be served from cache, and stripping iteration N's thinking on
         iteration N+1 rewrites that turn and invalidates everything after
         it. Prior-run turns are compacted identically on every call of the
         run — from the very first call, before the run has an assistant turn
         of its own — so the prior-run prefix stays cacheable across the run.
      2. Tool-result content compacted in user turns from *previous* agent
         runs only — everything from ``current_run_start`` onward is left
         untouched so the model keeps full access to tool results it's
         actively reasoning over.
      3. ``provider_reasoning`` blocks (opaque reasoning replayed by
         non-Anthropic providers) stripped from assistant turns of *previous*
         agent runs, using the same boundary as (2). Within the current run
         they stay: the OpenAI Responses API pairs each ``function_call`` with
         the ``reasoning`` item that produced it, so the active tool loop must
         keep them intact.

    If ``current_run_start`` is None the function falls back to compacting
    all user turns except the very last one (useful for offline analysis).

    The original list is never mutated: the result is a deep copy (or the
    same untouched list when there is nothing to compact), and no
    ``cache_control`` marker is ever written here — the provider adds its
    breakpoint to its own copy, so DynamoDB always stores the full raw
    history.
    """
    if len(messages) <= 1:
        return messages

    result = copy.deepcopy(messages)

    # --- thinking: strip from prior-run assistant turns only ---------------
    # Every assistant turn at or after the compaction boundary keeps its
    # thinking so the current run's prefix never changes between iterations.
    # The most recent assistant turn is exempted from stripping only when it
    # belongs to the current run (or in the offline fallback, which has no
    # boundary). When it is prior-run history — the previous run's final
    # answer, sitting just before the new user message — it must be stripped
    # on the *first* call of the run too: on call 1 it is the last assistant
    # turn in the list, on call 2 the current run has its own assistant turn,
    # and exempting it on call 1 only would rewrite that turn between the two
    # calls and invalidate the cached prior-run prefix once per follow-up.
    last_asst_idx = None
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "assistant":
            last_asst_idx = i
            break

    # --- tool results: determine compaction boundary ----------------------
    if current_run_start is not None:
        compact_tool_before = current_run_start
    else:
        # Offline / export fallback: compact all but the last user turn
        compact_tool_before = len(result)
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") == "user":
                compact_tool_before = i
                break

    if current_run_start is None or (last_asst_idx is not None and last_asst_idx >= current_run_start):
        keep_thinking_idx = last_asst_idx
    else:
        keep_thinking_idx = None

    for i, msg in enumerate(result):
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant" and isinstance(content, list):
            if i < compact_tool_before and i != keep_thinking_idx:
                content = [b for b in content if b.get("type") != "thinking"]
            if i < compact_tool_before:
                content = [b for b in content if b.get("type") != PROVIDER_REASONING_BLOCK_TYPE]
            msg["content"] = content

        if role == "user" and i < compact_tool_before and isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    block["content"] = _compact_tool_result_content(block.get("content", ""))

    return result
