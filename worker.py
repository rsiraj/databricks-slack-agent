"""Worker Lambda — runs the agent loop and posts to Slack.

Drives the model through ``utils.llm`` (a thin provider abstraction over the
Anthropic Messages API, with an OpenAI Responses translation ready to enable)
rather than an agent-runner SDK, so we can post progress updates to the Slack
thread between tool-use iterations.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Any

from utils import deadline
from utils import job_approvals
from utils.admins import admin_user_ids, channel_has_admin
from utils.audit import audit, set_context, tool_input_summary
from utils.channel_access import is_channel_allowed
from utils.clients import ANTHROPIC_TIMEOUT_SECONDS
from utils.compaction import _prepare_messages_for_api
from utils.feedback import (
    FEEDBACK_PROMPT_TEXT,
    RATING_LABELS,
    feedback_prompt_blocks,
    save_feedback,
)
from utils.job_run_evidence_guard import JobRunEvidenceGuard
from utils.llm import (
    DEFAULT_ANTHROPIC_MODEL,
    ModelRequest,
    ModelResponse,
    get_provider,
    provider_name,
    resolve_model,
)
from utils.llm.anthropic_provider import (
    base_model_id,
    recommended_max_tokens,
    usage_from_response,
)
from utils.numeric_guard import (
    NumericEvidenceGuard,
    numeric_guard_max_revisions,
    numeric_guard_mode,
)
from utils.ratelimit import check_channel, check_user
from utils.sessions import load_session, save_session
from utils.slack_formatting import normalize_slack_mrkdwn
from utils.slack_helpers import (
    extract_sql_blocks,
    extract_image_blocks,
    get_bot_user_id,
    get_channel_members,
    get_thread_replies,
    is_private_channel,
    leave_channel,
    post_ephemeral,
    post_message,
    post_message_chunks,
    render_sql_attachment,
    update_message,
    upload_sql_file,
)
from utils.sql_run_guard import SqlRunGuard, auto_satisfy_metadata_precondition
from utils.system_prompt import load_system_prompt
from tools import calc_tool, databricks_tool, github_tool
from tools.registry import all_tool_specs

logger = logging.getLogger(__name__)

# ``xhigh`` is accepted so a provider/model that offers it can be configured;
# the Anthropic provider maps it to ``max`` on models without it.
VALID_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
DEFAULT_EFFORT = "high"


def _resolve_effort(raw: str | None) -> str:
    """Validate ANTHROPIC_EFFORT; fall back to the default with a warning."""
    value = (raw or "").strip().lower()
    if value in VALID_EFFORTS:
        return value
    logger.warning(
        "Invalid ANTHROPIC_EFFORT %r; using %r (valid: %s)",
        raw,
        DEFAULT_EFFORT,
        ", ".join(sorted(VALID_EFFORTS)),
    )
    return DEFAULT_EFFORT


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default
    if value <= 0:
        logger.warning("Non-positive %s=%r; using %d", name, raw, default)
        return default
    return value


# Hard floor for the output cap: the value an invalid ANTHROPIC_MAX_TOKENS or an
# unknown model falls back to, and the cap the Sonnet 4.6 measurements were
# taken at. Terraform always sets ANTHROPIC_MAX_TOKENS (module default 24000 for
# the default model), so the deployed value is explicit; the resolution below
# only matters where the variable is genuinely unset (local runs, forks).
DEFAULT_MAX_TOKENS = 16_000
# The model whose measurements were taken at exactly DEFAULT_MAX_TOKENS; it is
# pinned so the special case survives changes to the production default model.
SONNET_4_6_BASELINE_MODEL = "claude-sonnet-4-6"


def _resolve_max_tokens(model: str, effort: str, default: int = DEFAULT_MAX_TOKENS) -> int:
    """Output cap for one model call, logged once per container.

    ``ANTHROPIC_MAX_TOKENS`` wins whenever it is set. When it is unset,
    ``claude-sonnet-4-6`` (any snapshot) keeps the ``default`` (16,000) cap
    its measurements were taken at so they stay comparable; any other model,
    including the production default ``claude-sonnet-5``, uses
    ``recommended_max_tokens(model, effort)`` (24,000 at ``high`` for Sonnet 5 /
    Opus 5: they spend ~30% more tokens on the same text and think longer at
    ``xhigh`` / ``max``). Unknown models fall back to ``default``. ``default``
    is the trailing positional argument so the eval contract test can read the
    hard-floor literal the same way it reads the other knobs.
    """
    raw = os.environ.get("ANTHROPIC_MAX_TOKENS")
    if raw not in (None, ""):
        value = _positive_int_env("ANTHROPIC_MAX_TOKENS", default)
        source = "ANTHROPIC_MAX_TOKENS"
    elif base_model_id(model) == SONNET_4_6_BASELINE_MODEL:
        value = default
        source = "baseline"
    else:
        value = recommended_max_tokens(model, effort)
        source = "recommended_max_tokens"
    logger.info(
        "max_tokens=%d (source=%s, model=%s, effort=%s)", value, source, model, effort
    )
    return value


# Provider selection lives in utils/llm: LLM_PROVIDER picks the provider,
# LLM_MODEL overrides the model for any provider, ANTHROPIC_MODEL /
# OPENAI_MODEL are the per-provider defaults (claude-sonnet-5 / gpt-5.6-terra).
PROVIDER_NAME = provider_name()
MODEL = resolve_model(PROVIDER_NAME)
EFFORT = _resolve_effort(os.environ.get("ANTHROPIC_EFFORT", DEFAULT_EFFORT))
MAX_TOKENS = _resolve_max_tokens(MODEL, EFFORT, DEFAULT_MAX_TOKENS)
MAX_ITERATIONS = 25
MAX_FINAL_REVISIONS = 1

# Lambda deadline awareness. The handler reserves a safety margin for posting
# a final Slack message and flushing logs; the loop refuses to start another
# model call when less than AGENT_MIN_SECONDS_PER_ITERATION remains.
LAMBDA_DEADLINE_SAFETY_MARGIN_SECONDS = 45.0
AGENT_MIN_SECONDS_PER_ITERATION = float(
    os.environ.get("AGENT_MIN_SECONDS_PER_ITERATION", "90")
)
ANTHROPIC_TIMEOUT_FLOOR_SECONDS = 20.0

DEADLINE_MESSAGE = (
    "I ran out of time before finishing. Try a narrower question or ask me to continue."
)
MAX_TOKENS_TRUNCATION_NOTE = "_(Response was cut off at the output limit.)_"
MAX_TOKENS_NO_TEXT_MESSAGE = "_The answer was too long to finish. Try a narrower question._"
REFUSAL_MESSAGE = "I can't help with that request."

ALL_TOOLS = (
    github_tool.GITHUB_TOOLS + databricks_tool.DATABRICKS_TOOLS + calc_tool.CALC_TOOLS
)
# Provider-neutral view of the same definitions; each provider renders its own shape.
TOOL_SPECS = all_tool_specs()
MAX_STATUS_STEPS = 5
CALC_STATUS_EXPRESSION_CHARS = 60

USAGE_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


def _usage_fields(resp: Any) -> dict[str, int]:
    """Read token usage off a Messages response, defaulting every field to 0."""
    return usage_from_response(resp).as_dict()


class RunMetrics:
    """Accumulates per-run model usage and latency for the run_summary audit event."""

    def __init__(self) -> None:
        self.api_calls = 0
        self.tool_calls = 0
        self.total_latency_ms = 0
        self.total_input_tokens = 0
        self.total_cache_creation_input_tokens = 0
        self.total_cache_read_input_tokens = 0
        self.total_output_tokens = 0
        # Overwritten when the loop returns normally; an exception leaves it.
        self.outcome = "error"

    def record_model_call(self, latency_ms: int, usage: dict[str, int]) -> None:
        self.api_calls += 1
        self.total_latency_ms += latency_ms
        self.total_input_tokens += usage.get("input_tokens", 0)
        self.total_cache_creation_input_tokens += usage.get("cache_creation_input_tokens", 0)
        self.total_cache_read_input_tokens += usage.get("cache_read_input_tokens", 0)
        self.total_output_tokens += usage.get("output_tokens", 0)

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    def summary(self) -> dict[str, Any]:
        return {
            "api_calls": self.api_calls,
            "tool_calls": self.tool_calls,
            "total_latency_ms": self.total_latency_ms,
            "total_input_tokens": self.total_input_tokens,
            "total_cache_creation_input_tokens": self.total_cache_creation_input_tokens,
            "total_cache_read_input_tokens": self.total_cache_read_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "outcome": self.outcome,
            "model": MODEL,
            "effort": EFFORT,
            "provider": PROVIDER_NAME,
        }


def _new_error_id() -> str:
    """Return a short, non-sensitive reference for correlating user reports."""
    return f"agent-{uuid.uuid4().hex[:12]}"


def _is_anthropic_overload_error(error: Exception | None) -> bool:
    """Recognize the typed SDK error without importing Anthropic eagerly."""
    if error is None:
        return False
    return (
        type(error).__name__ == "OverloadedError"
        or getattr(error, "status_code", None) == 529
    )


def _agent_error_fields(error: Exception) -> dict[str, Any]:
    """Return structured, non-secret fields that make failures searchable."""
    status_code = getattr(error, "status_code", None)
    request_id = getattr(error, "request_id", None)
    fields: dict[str, Any] = {
        "failure_type": (
            "anthropic_overloaded"
            if _is_anthropic_overload_error(error)
            else "unhandled_exception"
        ),
    }
    if status_code is not None:
        fields["status_code"] = status_code
    if request_id:
        fields["provider_request_id"] = request_id
    return fields


def _safe_agent_error_message(error_id: str, error: Exception | None = None) -> str:
    if _is_anthropic_overload_error(error):
        return (
            ":warning: The AI service is temporarily overloaded. I retried "
            "automatically but couldn't complete this request. Please try again "
            f"in a minute. Reference: `{error_id}`."
        )
    return (
        ":warning: Something went wrong while processing this request. "
        f"Reference: `{error_id}`."
    )


def _persist_agent_failure_if_balanced(
    messages: list[dict],
    thread_ts: str,
    visible_error: str,
) -> bool:
    """Archive a failed turn only when appending an assistant message is valid."""
    if not messages or messages[-1].get("role") != "user":
        return False
    messages.append({"role": "assistant", "content": visible_error})
    save_session(thread_ts, messages)
    return True


def _dispatch_tool(name: str, inp: dict[str, Any]) -> dict:
    if name == "databricks_run_job":
        return _request_job_confirmation(inp)
    if name.startswith("github_"):
        return github_tool.dispatch(name, inp)
    if name.startswith("databricks_"):
        return databricks_tool.dispatch(name, inp)
    if name == "calculate":
        return calc_tool.dispatch(name, inp)
    return {"error": f"Unknown tool: {name}"}


def _request_job_confirmation(inp: dict) -> dict:
    context = job_approvals.current_request()
    if not context or not context.get("user_id"):
        return {"error": "Job confirmation requires an authenticated Slack request.",
                "validation_failure": "job_confirmation_required"}
    try:
        raw_job_id = inp["job_id"]
        if isinstance(raw_job_id, bool) or not isinstance(raw_job_id, (int, str)):
            raise ValueError("Invalid job_id")
        job_id = int(raw_job_id)
        if job_id <= 0:
            raise ValueError("Invalid job_id")
        params = job_approvals.normalize_parameters(inp.get("notebook_params"))
        key = json.dumps([job_id, params], sort_keys=True)
        if key in context["requests"]:
            return context["requests"][key]
        payload = databricks_tool.get_job_configuration(job_id)
        if "error" in payload:
            return payload
        job = job_approvals.snapshot(payload)
        if job["job_id"] != job_id:
            raise ValueError("The returned configuration does not match the requested job")
        approval_id = job_approvals.create(job, params, context)
        message_ts = post_message(
            context["channel_id"],
            f"Confirm job run: {job['job_name']} (job_id {job_id}). "
            f"Parameters: {json.dumps(params, ensure_ascii=False, sort_keys=True)}",
            thread_ts=context["thread_ts"],
            blocks=job_approvals.confirmation_blocks(approval_id, job, params),
        )
        if not message_ts:
            return {"error": "Could not display the confirmation button; no job was started."}
        job_approvals.activate(approval_id, message_ts)
    except (KeyError, TypeError, ValueError) as exc:
        return {"error": str(exc), "validation_failure": "job_confirmation_required"}
    result = {"status": "confirmation_pending", "job_id": job_id, "job_name": job["job_name"],
              "confirmation_expires_in_minutes": job_approvals.APPROVAL_TTL_SECONDS // 60,
              "message": "Use the Confirm run button in this thread. No job has been started. "
                         "Any current member can confirm once; typed yes does not execute jobs."}
    context["requests"][key] = result
    audit("job_confirmation_requested", job_id=job_id, approval_id=approval_id)
    return result


def _handle_job_confirmation(event: dict) -> dict:
    fields = ("channel_id", "thread_ts", "message_ts", "user_id", "approval_id")
    if not all(isinstance(event.get(key), str) and event[key] for key in fields):
        return {"ok": False}
    channel, thread_ts, user_id = event["channel_id"], event["thread_ts"], event["user_id"]
    set_context(user_id=user_id, channel_id=channel, thread_ts=thread_ts)
    # Membership is checked now, not inferred from the original request or a
    # button's visibility. Any current member can confirm, once, for everyone.
    if not is_channel_allowed(channel) or is_private_channel(channel) is not True:
        return {"ok": False}
    members = get_channel_members(channel)
    if user_id not in members or not channel_has_admin(members):
        return {"ok": False}
    try:
        job = job_approvals.claim(event)
    except Exception:
        logger.exception("Could not claim job confirmation")
        post_ephemeral(channel, user_id, "I couldn't verify that confirmation. No job was started; try the button again.",
                       thread_ts=thread_ts)
        return {"ok": False}
    if job is None:
        post_ephemeral(channel, user_id,
                       "That confirmation was already used, expired, or is no longer valid. No additional run was started.",
                       thread_ts=thread_ts)
        return {"ok": True}

    confirmed = f"<@{user_id}> confirmed job `{job['job_id']}`. Checking its current configuration before starting."

    def show_status(text: str) -> bool:
        blocks = [
            job_approvals.confirmation_blocks(event["approval_id"], job, job["notebook_params"])[0],
            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        ]
        if update_message(channel, event["message_ts"], text, blocks=blocks):
            return True
        return bool(post_message(channel, text, thread_ts=thread_ts))

    # The shared attribution is part of consent: do not launch invisibly when
    # Slack cannot confirm either the card update or the fallback thread post.
    if not show_status(confirmed):
        try:
            job_approvals.complete(event["approval_id"], {"error": "Confirmation could not be published; no run submitted."})
        except Exception:
            logger.exception("Could not record unpublished confirmation; approval remains consumed")
        audit("job_confirmation_delivery_failed", approval_id=event["approval_id"], job_id=job["job_id"])
        return {"ok": False}
    audit("job_confirmation_accepted", approval_id=event["approval_id"], job_id=job["job_id"])
    submitted = False
    try:
        payload = databricks_tool.get_job_configuration(job["job_id"])
        if "error" in payload:
            raise ValueError("The current job configuration could not be verified")
        current = job_approvals.snapshot(payload)
        if current != {key: job[key] for key in current}:
            raise ValueError("The job configuration changed after the button was displayed")
        submitted = True
        result = databricks_tool.run_job(job["job_id"], job["notebook_params"],
                                         idempotency_token=event["approval_id"])
        if result.get("run_id"):
            final = f"<@{user_id}> confirmed job `{job['job_id']}`. Started run `{result['run_id']}`. Ask me for its status anytime."
        elif result.get("status_code") == 403:
            final = f"<@{user_id}> confirmed job `{job['job_id']}`, but Databricks denied permission to run it."
        else:
            final = (f"<@{user_id}> confirmed job `{job['job_id']}`, but I couldn't verify whether the run started. "
                     "Check its recent runs before requesting another run. This button cannot be used again.")
    except Exception as exc:
        logger.exception("Confirmed job execution failed")
        result = {"error": str(exc)}
        if submitted:
            final = (f"<@{user_id}> confirmed job `{job['job_id']}`, but the launch result is uncertain. "
                     "Check its recent runs before requesting another run. This button cannot be used again.")
        else:
            final = (f"<@{user_id}> confirmed job `{job['job_id']}`, but the job's current configuration "
                     "could not be approved. No run was submitted. Request a fresh confirmation after checking the job.")
    card_delivered = show_status(final)
    # Also post a new thread reply so confirmations cannot be missed as edits
    # and their result can be recovered by Slack transcript replay.
    answer_ts = post_message(channel, final, thread_ts=thread_ts)
    try:
        job_approvals.complete(event["approval_id"], result)
    except Exception:
        logger.exception("Could not save confirmed job outcome; approval remains consumed")
    if card_delivered or answer_ts:
        history = load_session(thread_ts) or []
        tool_id = f"confirmed-{event['approval_id']}"
        history.extend([
            {"role": "user", "content": f"Slack user {user_id} clicked Confirm run for job {job['job_id']} "
             f"with parameters {json.dumps(job['notebook_params'], sort_keys=True)}."},
            {"role": "assistant", "content": [{"type": "tool_use", "id": tool_id,
             "name": "databricks_run_job", "input": {"job_id": job["job_id"],
             "notebook_params": job["notebook_params"]}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id,
             "content": json.dumps(result, default=str), "is_error": "error" in result}]},
            {"role": "assistant", "content": final},
        ])
        save_session(thread_ts, history)
    return {"ok": True}


def _tool_outcome(
    tool_name: str,
    result: dict[str, Any],
    *,
    guard_rejected: bool = False,
) -> str:
    """Classify a completed tool call without conflating guards with failures."""
    if guard_rejected or result.get("validation_failure"):
        return "guard_precondition"
    if "error" not in result:
        return "success"
    if result.get("rate_limit_failure"):
        return "tool_failure"
    if tool_name.startswith("databricks_"):
        return "databricks_failure"
    return "tool_failure"


def _databricks_failure_type(result: dict[str, Any], *, raised: bool) -> str:
    if result.get("status") in {"FAILED", "CANCELED", "CLOSED"}:
        return "statement"
    if result.get("status_code"):
        return "http"
    if raised:
        return "tool_exception"
    return "tool_error"


def _databricks_error_code(result: dict[str, Any]) -> str | None:
    """Extract a stable Databricks error code without exporting the full message."""
    match = re.search(r"\[([A-Z][A-Z0-9_.]+)\]", str(result.get("error") or ""))
    return match.group(1) if match else None


def _auto_satisfy_metadata_precondition(
    sql_guard: SqlRunGuard,
    tool_name: str,
    tool_input: dict[str, Any],
    rejection: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Silently describe exact first-use SQL relations, then re-check the query.

    This is a runtime safety net shared with model evaluations so both paths
    enforce the same metadata-before-query behavior.
    """
    result, attempts = auto_satisfy_metadata_precondition(
        sql_guard,
        tool_name,
        tool_input,
        rejection,
        _dispatch_tool,
    )
    for attempt in attempts:
        metadata_input = attempt["input"]
        object_name = ".".join(
            metadata_input[part] for part in ("catalog", "schema", "name")
        )
        exception = attempt["exception"]
        if exception is not None:
            logger.error(
                "Automatic metadata preflight failed for %s",
                object_name,
                exc_info=(type(exception), exception, exception.__traceback__),
            )
        audit(
            "metadata_preflight_injected",
            object_name=object_name,
            succeeded=attempt["succeeded"],
        )
    return result


def _block_field(block: Any, name: str) -> Any:
    """Read a field from a content block given as a dict or an SDK object."""
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def _extract_text(blocks: list) -> str:
    """Concatenate ``text`` blocks; other block types (tool_use, thinking,
    provider_reasoning) are skipped."""
    return "".join(
        _block_field(b, "text")
        for b in blocks
        if _block_field(b, "type") == "text" and _block_field(b, "text")
    )


def _replace_assistant_text(message: dict, text: str) -> None:
    """Keep persisted assistant text identical to the Slack-rendered response."""
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = text
        return
    if not isinstance(content, list):
        message["content"] = [{"type": "text", "text": text}]
        return

    replaced = False
    updated = []
    for block in content:
        if block.get("type") != "text":
            updated.append(block)
        elif not replaced:
            updated.append({**block, "text": text})
            replaced = True
    if not replaced:
        updated.append({"type": "text", "text": text})
    message["content"] = updated


def _user_message_text(message: dict | None) -> str:
    """Return the plain text of a user message (string or text blocks)."""
    if not message or message.get("role") != "user":
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _remove_private_revisions(messages: list[dict], *start_indices: int | None) -> None:
    """Drop each private draft + correction-request pair, highest index first.

    Both final-answer guards record the index of the rejected draft; the
    correction request sits right after it. Anything the model did in response
    (a ``calculate`` tool_use and its result, the final answer) stays in place.
    """
    for index in sorted((i for i in start_indices if i is not None), reverse=True):
        del messages[index : index + 2]


def _strip_mention(text: str) -> str:
    return re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()


def _is_status_message(text: str) -> bool:
    first_line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return first_line in {
        "*Working on it...*",
        "*Checked*",
        "_Working on it..._",
        "_Checked_",
    }


def _is_feedback_message(text: str) -> bool:
    return (text or "").strip() == FEEDBACK_PROMPT_TEXT


def _build_message_history(
    thread_messages: list[dict],
    bot_user_id: str,
    trigger_ts: str,
    trigger_image_blocks: list[dict] | None = None,
) -> list[dict]:
    """Rebuild a user/assistant transcript from Slack thread replies.

    Prior tool calls are not replayed — only the final text the bot posted is included.
    Image attachments on user messages are included as Claude image content blocks.
    The triggering mention (``ts == trigger_ts``) reuses ``trigger_image_blocks``
    already downloaded by the caller instead of fetching the same files again.
    """
    messages: list[dict] = []
    for msg in thread_messages:
        text = (msg.get("text") or "").strip()
        is_bot = bool(msg.get("bot_id")) or msg.get("user") == bot_user_id

        if is_bot:
            if not text:
                continue
            # Skip italic/progress notes — they start with underscore and are noise for the model
            if text.startswith("_") and text.endswith("_"):
                continue
            if _is_status_message(text):
                continue
            if _is_feedback_message(text):
                continue
            messages.append({"role": "assistant", "content": text})
        else:
            cleaned = _strip_mention(text) if text else ""
            if trigger_ts and msg.get("ts") == trigger_ts:
                image_blocks = list(trigger_image_blocks or [])
            else:
                image_blocks = extract_image_blocks(msg)
            if not cleaned and not image_blocks:
                continue
            content: list[dict] = []
            if cleaned:
                content.append({"type": "text", "text": cleaned})
            content.extend(image_blocks)
            messages.append({"role": "user", "content": content})

    # Collapse adjacent same-role turns (Anthropic API requires alternation)
    collapsed: list[dict] = []
    for m in messages:
        if collapsed and collapsed[-1]["role"] == m["role"]:
            prev = collapsed[-1]
            # Normalize to list-of-blocks on merge
            prev_content = (
                prev["content"]
                if isinstance(prev["content"], list)
                else [{"type": "text", "text": prev["content"]}]
            )
            new_content = (
                m["content"]
                if isinstance(m["content"], list)
                else [{"type": "text", "text": m["content"]}]
            )
            prev["content"] = prev_content + new_content
        else:
            collapsed.append(m)

    if not collapsed or collapsed[-1]["role"] != "user":
        return []
    return collapsed


def _build_system() -> list[dict]:
    # Cached block — stable prompt, well above the 1024-token minimum for
    # Sonnet 4.6 / Sonnet 5. Kept in the Anthropic block shape (the numeric
    # guard reads ``text`` off it); ``_call_model`` hands the provider the plain
    # text and the Anthropic provider re-wraps it with the same ``cache_control``
    # marker, then adds a second, moving marker on the last message block so
    # the growing conversation is cached too (utils/llm/anthropic_provider.py).
    return [
        {
            "type": "text",
            "text": load_system_prompt(PROVIDER_NAME),
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _system_text(system: list[dict]) -> str:
    return "".join(
        str(block.get("text") or "")
        for block in system
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _tool_status_label(tool_name: str, tool_input: dict) -> str:
    if tool_name == "github_search_code":
        return f"Search GitHub for `{tool_input.get('query', '')}`"
    if tool_name == "github_find_files":
        return f"Find GitHub files matching `{tool_input.get('name', '')}`"
    if tool_name == "github_get_file_content":
        return f"Read `{tool_input.get('path', '')}` from GitHub"
    if tool_name == "github_get_commits":
        return f"Fetch commit history for `{tool_input.get('path', '')}`"
    if tool_name == "github_compare_commits":
        return f"Compare `{tool_input.get('base', '')}...{tool_input.get('head', '')}`"
    if tool_name == "databricks_execute_sql":
        return "Run Databricks query"
    if tool_name == "databricks_get_statement":
        return f"Check statement `{tool_input.get('statement_id', '')}`"
    if tool_name == "databricks_find_objects":
        return f"Find Databricks objects matching `{tool_input.get('name', '')}`"
    if tool_name == "databricks_describe_object":
        return (
            "Describe Databricks object "
            f"`{tool_input.get('catalog', '')}.{tool_input.get('schema', '')}."
            f"{tool_input.get('name', '')}`"
        )
    if tool_name == "databricks_run_job":
        return f"Trigger Databricks job `{tool_input.get('job_id', '')}`"
    if tool_name == "databricks_get_run_status":
        return f"Check Databricks run `{tool_input.get('run_id', '')}`"
    if tool_name == "databricks_cancel_run":
        return f"Cancel Databricks run `{tool_input.get('run_id', '')}`"
    if tool_name == "databricks_get_recent_runs":
        return f"Fetch recent runs for job `{tool_input.get('job_id', '')}`"
    if tool_name == "databricks_list_job_runs":
        return f"List Databricks runs for `{tool_input.get('date', 'today')}`"
    if tool_name == "calculate":
        expression = " ".join(str(tool_input.get("expression", "")).split())
        if len(expression) > CALC_STATUS_EXPRESSION_CHARS:
            expression = expression[: CALC_STATUS_EXPRESSION_CHARS - 1] + "…"
        return f"Calculate `{expression}`"
    return f"Call `{tool_name}`"


class RunStatus:
    """Owns one editable Slack status message for a full agent run."""

    def __init__(self, channel: str, thread_ts: str):
        self.channel = channel
        self.thread_ts = thread_ts
        self.message_ts: str | None = None
        self.steps: list[dict[str, str]] = []

    def start_tool(self, label: str) -> None:
        self._finish_current("done")
        self.steps.append({"label": label, "state": "running"})
        self._sync("working")

    def finish_tool(self, *, error: bool = False) -> None:
        self._finish_current("error" if error else "done")
        self._sync("working")

    def complete(self) -> None:
        if not self.steps:
            return
        self._finish_current("done")
        self._sync("done")

    def stop(self, note: str) -> None:
        if not self.steps:
            return
        self._finish_current("error")
        self.steps.append({"label": note, "state": "error"})
        self._sync("done")

    def delivery_failed(self, delivered: int, total: int) -> None:
        note = f"Answer delivery stopped after {delivered} of {total} part(s)"
        if self.steps or self.message_ts:
            self._finish_current("error")
            self.steps.append({"label": note, "state": "error"})
            self._sync("done")
            return
        post_message(
            self.channel,
            f":warning: {note}. Please try again.",
            thread_ts=self.thread_ts,
        )

    def _finish_current(self, state: str) -> None:
        if self.steps and self.steps[-1]["state"] == "running":
            self.steps[-1]["state"] = state

    def _render(self, phase: str) -> str:
        title = "_Checked_" if phase == "done" else "_Working on it..._"
        visible = self.steps[-MAX_STATUS_STEPS:]
        hidden = max(0, len(self.steps) - len(visible))

        lines = [title]
        if hidden:
            lines.append(f"_{hidden} earlier check(s) hidden_")

        for step in visible:
            state = step["state"]
            label = step["label"]
            if state == "running":
                lines.append(f"_- In progress: {label}_")
            elif state == "error":
                lines.append(f"_- Had trouble: {label}_")
            else:
                lines.append(f"_- {label}_")

        return "\n".join(lines)

    def _sync(self, phase: str) -> None:
        text = self._render(phase)
        if self.message_ts:
            update_message(self.channel, self.message_ts, text)
            return

        ts = post_message(self.channel, text, thread_ts=self.thread_ts)
        if ts:
            self.message_ts = ts


def _post_feedback_prompt(channel: str, thread_ts: str, answer_ts: str) -> None:
    post_message(
        channel,
        FEEDBACK_PROMPT_TEXT,
        thread_ts=thread_ts,
        blocks=feedback_prompt_blocks(answer_ts),
    )


def _attach_final_sql(channel: str, thread_ts: str, final: str) -> bool:
    """Attach SQL-language code fences from a completely delivered answer."""
    sql_blocks = extract_sql_blocks(final)
    if not sql_blocks:
        return True

    filename = f"query-{thread_ts.replace('.', '-')}.sql"
    sql = render_sql_attachment(sql_blocks)
    file_id = upload_sql_file(
        channel,
        sql,
        thread_ts=thread_ts,
        filename=filename,
    )
    if file_id:
        audit(
            "sql_attachment_uploaded",
            file_id=file_id,
            sql_block_count=len(sql_blocks),
            sql_character_count=len(sql),
        )
        return True

    audit(
        "sql_attachment_failed",
        sql_block_count=len(sql_blocks),
        sql_character_count=len(sql),
    )
    post_message(
        channel,
        ":warning: I couldn't attach the SQL file. The complete SQL is still "
        "available in the answer above.",
        thread_ts=thread_ts,
    )
    return False


def _call_model(
    system: list[dict],
    messages: list[dict],
    current_run_start: int,
    iteration: int,
    metrics: RunMetrics,
) -> ModelResponse:
    """One model call with latency/usage measurement and a bounded timeout.

    The provider owns the wire format; for Anthropic the request is identical
    to the pre-abstraction ``messages.create`` call (see
    ``utils/llm/anthropic_provider.py``).
    """
    timeout_seconds = deadline.bounded_timeout(
        ANTHROPIC_TIMEOUT_SECONDS, floor=ANTHROPIC_TIMEOUT_FLOOR_SECONDS
    )
    provider = get_provider()
    # Timer starts before compaction, as the baseline measurement did.
    started = time.monotonic()
    request = ModelRequest(
        system=_system_text(system),
        messages=_prepare_messages_for_api(messages, current_run_start),
        tools=TOOL_SPECS,
        effort=EFFORT,
        max_tokens=MAX_TOKENS,
        timeout_s=timeout_seconds,
        model=MODEL,
    )
    # The SDK timeout is per network attempt/read, not a total budget. This
    # outer deadline also interrupts retries and continuously active streams.
    with deadline.enforce_remaining():
        resp = provider.create(request)
    latency_ms = int(round((time.monotonic() - started) * 1000))
    usage = resp.usage.as_dict()
    metrics.record_model_call(latency_ms, usage)
    audit(
        "model_call",
        model=resp.model or MODEL,
        requested_model=MODEL,
        provider=resp.provider or provider.name,
        effort=EFFORT,
        iteration=iteration,
        latency_ms=latency_ms,
        stop_reason=resp.raw_stop_reason or resp.stop_reason,
        max_tokens=MAX_TOKENS,
        transport=resp.transport,
        **usage,
    )
    return resp


def _deliver_final_answer(
    messages: list[dict],
    channel: str,
    thread_ts: str,
    status: RunStatus,
    final: str,
    *,
    complete_answer: bool = True,
) -> None:
    """Post the normalized answer, persist confirmed parts, then attach SQL/feedback.

    ``complete_answer=False`` (a max_tokens truncation) still delivers and
    saves, but skips the SQL attachment and feedback panel since the answer
    is known to be cut off.
    """
    status.complete()
    if not final:
        fallback = "_(no response generated)_"
        answer_ts = post_message(channel, fallback, thread_ts=thread_ts)
        if answer_ts:
            _replace_assistant_text(messages[-1], fallback)
            save_session(thread_ts, messages)
        else:
            status.delivery_failed(0, 1)
        return

    delivery = post_message_chunks(channel, final, thread_ts=thread_ts)
    if delivery.timestamps:
        _replace_assistant_text(messages[-1], delivery.delivered_text)
        save_session(thread_ts, messages)
    if delivery.complete:
        if complete_answer:
            _attach_final_sql(channel, thread_ts, final)
            _post_feedback_prompt(channel, thread_ts, delivery.timestamps[-1])
    else:
        logger.error(
            "Slack answer delivery incomplete: %d/%d part(s)",
            len(delivery.timestamps),
            delivery.total_chunks,
        )
        audit(
            "answer_delivery_failed",
            delivered_parts=len(delivery.timestamps),
            total_parts=delivery.total_chunks,
        )
        status.delivery_failed(len(delivery.timestamps), delivery.total_chunks)


def run_agent(
    messages: list[dict],
    channel: str,
    thread_ts: str,
    metrics: RunMetrics | None = None,
) -> str:
    """Run the agent loop. `messages` is mutated in place. Session is persisted
    whenever the message list is in a balanced state (no dangling tool_use).

    Returns the run outcome (``answered``, ``max_iterations``, ``deadline``,
    ``unexpected_stop``, ``max_tokens``, ``refusal``). ``metrics`` is updated
    in place so the caller can emit ``run_summary`` even when this raises.
    """
    if metrics is None:
        metrics = RunMetrics()
    outcome = _run_agent_loop(messages, channel, thread_ts, metrics)
    metrics.outcome = outcome
    return outcome


def _run_agent_loop(
    messages: list[dict], channel: str, thread_ts: str, metrics: RunMetrics
) -> str:
    system = _build_system()
    status = RunStatus(channel, thread_ts)
    sql_guard = SqlRunGuard()
    evidence_guard = JobRunEvidenceGuard()
    # Everything before this index is from previous agent runs and can be
    # compacted.  Everything from here onward is the current run — full
    # fidelity so the model never loses tool results it's actively using.
    current_run_start = len(messages) - 1
    final_revision_count = 0
    revision_start_index: int | None = None

    # Numeric grounding: every figure in the final answer must trace back to a
    # tool result (or the user's own words / the bot's operational constants).
    # The system prompt is deliberately NOT grounded: its business narrative
    # would vouch for coincidental figures. Prior-turn tool results count;
    # prior assistant text does not.
    numeric_mode = numeric_guard_mode()
    numeric_max_revisions = numeric_guard_max_revisions()
    numeric_guard = NumericEvidenceGuard()
    numeric_revision_count = 0
    # One entry per private revision (the budget may allow more than one).
    numeric_revision_start_indices: list[int] = []
    if numeric_mode != "off":
        numeric_guard.record_history(messages[:current_run_start])
        numeric_guard.record_user_text(_user_message_text(messages[current_run_start]))

    for i in range(MAX_ITERATIONS):
        remaining = deadline.remaining_seconds()
        if remaining is not None and remaining < AGENT_MIN_SECONDS_PER_ITERATION:
            logger.warning(
                "Stopping agent loop before Lambda deadline",
                extra={"iteration": i, "remaining_seconds": round(remaining, 1)},
            )
            status.stop("Ran out of time")
            post_message(channel, DEADLINE_MESSAGE, thread_ts=thread_ts)
            _persist_agent_failure_if_balanced(messages, thread_ts, DEADLINE_MESSAGE)
            audit(
                "agent_deadline_reached",
                iteration=i,
                remaining_seconds=round(remaining, 1),
                api_calls=metrics.api_calls,
            )
            return "deadline"

        logger.info("Agent iteration", extra={"iteration": i})
        try:
            resp = _call_model(system, messages, current_run_start, i, metrics)
        except deadline.DeadlineExceeded:
            status.stop("Ran out of time")
            post_message(channel, DEADLINE_MESSAGE, thread_ts=thread_ts)
            _persist_agent_failure_if_balanced(messages, thread_ts, DEADLINE_MESSAGE)
            audit("agent_deadline_reached", iteration=i, api_calls=metrics.api_calls)
            return "deadline"

        # Append the assistant turn as-is (preserves tool_use blocks — and any
        # provider_reasoning the provider needs replayed — for the next request).
        messages.append({"role": "assistant", "content": list(resp.content)})

        if resp.stop_reason == "end_turn":
            draft = _extract_text(resp.content).strip()
            fallback_used = False
            violations = evidence_guard.validate(draft)
            if violations:
                revision_instruction = evidence_guard.revision_instruction(violations)
                if (
                    revision_instruction is not None
                    and final_revision_count < MAX_FINAL_REVISIONS
                ):
                    if revision_start_index is None:
                        revision_start_index = len(messages) - 1
                    messages.append({"role": "user", "content": revision_instruction})
                    final_revision_count += 1
                    audit("final_response_guard_revision", violations=violations)
                    continue
                logger.warning(
                    "Blocked unsupported job-run interpretation: thread_ts=%s violations=%s",
                    thread_ts,
                    ",".join(violations),
                )
                draft = evidence_guard.safe_fallback()
                messages[-1]["content"] = [{"type": "text", "text": draft}]
                audit("final_response_guard_fallback", violations=violations)
                fallback_used = True

            # The numeric guard only checks model-authored text; the evidence
            # guard's deterministic fallback is computed from tool data already.
            if numeric_mode != "off" and not fallback_used:
                unverified = numeric_guard.validate(draft)
                if unverified:
                    figures = [item["text"] for item in unverified]
                    if (
                        numeric_mode == "revise"
                        and numeric_revision_count < numeric_max_revisions
                    ):
                        numeric_revision_start_indices.append(len(messages) - 1)
                        messages.append(
                            {
                                "role": "user",
                                "content": numeric_guard.revision_instruction(unverified),
                            }
                        )
                        numeric_revision_count += 1
                        audit("numeric_guard_revision", figures=figures, count=len(figures))
                        continue
                    logger.warning(
                        "Unverified figures in final answer: thread_ts=%s figures=%s",
                        thread_ts,
                        ", ".join(figures),
                    )
                    draft = numeric_guard.annotate(draft, unverified)
                    audit("numeric_guard_unverified", figures=figures, count=len(figures))

            _remove_private_revisions(
                messages, revision_start_index, *numeric_revision_start_indices
            )

            final = normalize_slack_mrkdwn(draft)
            _deliver_final_answer(messages, channel, thread_ts, status, final)
            return "answered"

        if resp.stop_reason == "max_tokens":
            draft = _extract_text(resp.content).strip()
            if not draft:
                # Thinking-only or a tool_use that was cut off: nothing safe to
                # show, and the dangling assistant turn must not be persisted.
                logger.warning(
                    "Model hit max_tokens with no deliverable text",
                    extra={"iteration": i, "max_tokens": MAX_TOKENS},
                )
                status.stop("Answer was too long to finish")
                post_message(channel, MAX_TOKENS_NO_TEXT_MESSAGE, thread_ts=thread_ts)
                return "max_tokens"
            logger.warning(
                "Model hit max_tokens; delivering truncated text",
                extra={"iteration": i, "max_tokens": MAX_TOKENS},
            )
            # Drop any partially emitted tool_use so the saved history never
            # carries a tool call without a matching tool_result.
            messages[-1]["content"] = [
                block for block in messages[-1]["content"] if block.get("type") != "tool_use"
            ]
            final = normalize_slack_mrkdwn(draft) + "\n\n" + MAX_TOKENS_TRUNCATION_NOTE
            _deliver_final_answer(
                messages, channel, thread_ts, status, final, complete_answer=False
            )
            return "max_tokens"

        if resp.stop_reason == "refusal":
            logger.warning("Model refused the request", extra={"iteration": i})
            audit("model_refusal", iteration=i)
            status.stop("Request declined")
            post_message(channel, REFUSAL_MESSAGE, thread_ts=thread_ts)
            # Don't save — the refusal turn has no user-visible answer to anchor to.
            return "refusal"

        if resp.stop_reason == "pause_turn":
            # Long-running server-side turn paused; resend the assistant turn
            # as-is and let the model continue (bounded by MAX_ITERATIONS).
            logger.info("Model paused its turn; continuing", extra={"iteration": i})
            continue

        if resp.stop_reason != "tool_use":
            raw_reason = resp.raw_stop_reason or resp.stop_reason
            logger.warning("Unexpected stop_reason: %s", raw_reason)
            status.stop(f"Stopped unexpectedly: {raw_reason}")
            post_message(channel, f"_Stopped unexpectedly: {raw_reason}_", thread_ts=thread_ts)
            # Don't save — assistant turn ended without tool_use AND without end_turn,
            # so the messages list is in an undefined state and continuing it could confuse the model.
            return "unexpected_stop"

        # Interleaved text before a tool call is preserved in DynamoDB as part
        # of the assistant turn above. Slack gets the single editable status
        # message instead, which keeps the thread readable.

        tool_uses = [b for b in resp.content if b.get("type") == "tool_use"]
        tool_results: list[dict] = []
        for tu in tool_uses:
            tool_name = str(tu.get("name") or "")
            tool_input = tu.get("input") if isinstance(tu.get("input"), dict) else {}
            tool_use_id = tu.get("id")
            metrics.record_tool_call()
            status.start_tool(_tool_status_label(tool_name, tool_input))
            initial_rejection = sql_guard.before_call(tool_name, tool_input)
            result = initial_rejection
            result = _auto_satisfy_metadata_precondition(
                sql_guard,
                tool_name,
                tool_input,
                result,
            )
            if initial_rejection is not None:
                initial_auto_satisfied = result is None or result is not initial_rejection
                audit(
                    "guard_precondition",
                    tool_name=tool_name,
                    source="sql_run_guard",
                    validation_failure=initial_rejection.get("validation_failure"),
                    objects=(
                        initial_rejection.get("objects")
                        or initial_rejection.get("blocked_objects")
                        or ([initial_rejection["object"]] if initial_rejection.get("object") else [])
                    ),
                    auto_satisfied=initial_auto_satisfied,
                    iteration=i,
                )
            tool_raised = False
            tool_dispatched = False
            if result is None:
                tool_dispatched = True
                try:
                    result = _dispatch_tool(tool_name, tool_input)
                    result = sql_guard.after_call(tool_name, tool_input, result)
                    is_error = isinstance(result, dict) and "error" in result
                except Exception as e:
                    logger.exception("Tool failed: %s", tool_name)
                    result = {"error": str(e)}
                    response = getattr(e, "response", None)
                    if getattr(response, "status_code", None):
                        result["status_code"] = response.status_code
                    result = sql_guard.after_call(tool_name, tool_input, result)
                    is_error = True
                    tool_raised = True
            else:
                is_error = True
            outcome = _tool_outcome(
                tool_name,
                result,
                guard_rejected=not tool_dispatched,
            )
            final_guard_needs_event = outcome == "guard_precondition" and (
                initial_rejection is None or result is not initial_rejection
            )
            if final_guard_needs_event:
                audit(
                    "guard_precondition",
                    tool_name=tool_name,
                    source="tool_validation" if tool_dispatched else "sql_run_guard",
                    validation_failure=result.get("validation_failure"),
                    objects=(
                        result.get("objects")
                        or result.get("blocked_objects")
                        or ([result["object"]] if result.get("object") else [])
                    ),
                    auto_satisfied=False,
                    iteration=i,
                )
            elif outcome == "databricks_failure":
                audit(
                    "databricks_failure",
                    tool_name=tool_name,
                    failure_type=_databricks_failure_type(result, raised=tool_raised),
                    status=result.get("status"),
                    status_code=result.get("status_code"),
                    statement_id=result.get("statement_id"),
                    error_code=_databricks_error_code(result),
                    iteration=i,
                )
            evidence_guard.record_tool_result(tool_name, tool_input, result)
            numeric_guard.record_tool_result(tool_name, tool_input, result)
            status.finish_tool(error=is_error)
            audit(
                "tool_call",
                tool_name=tool_name,
                tool_input_summary=tool_input_summary(tool_input),
                is_error=is_error,
                outcome=outcome,
                iteration=i,
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(result, default=str),
                    "is_error": is_error,
                }
            )

        messages.append({"role": "user", "content": tool_results})
        # Balanced state at the end of the iteration; the next iteration will overwrite
        # with a fresh assistant turn anyway, so saving here is redundant.

    status.stop("Hit max iterations without completing")
    post_message(channel, "_Hit max iterations without completing. Try narrowing the question._", thread_ts=thread_ts)
    if messages and messages[-1].get("role") == "user":
        save_session(thread_ts, messages)  # balanced — last append was tool_results
    else:
        # Only reachable if the final iteration ended on pause_turn: the last
        # message is an assistant turn with no tool_use, so saving it would
        # leave the thread in a state the next mention cannot continue from.
        logger.warning("Max iterations reached on an unbalanced history; not saving")
    return "max_iterations"


def _build_new_user_message(text: str, image_blocks: list[dict]) -> dict:
    """Construct a single user message, with image content blocks if present."""
    if image_blocks:
        return {"role": "user", "content": [{"type": "text", "text": text}, *image_blocks]}
    return {"role": "user", "content": text}


NO_ADMIN_JOIN_MESSAGE = (
    "👋 Hi! I can only operate in channels where at least one designated admin "
    "is a member. None are here, so I'm leaving. Ask an admin to join first, "
    "then re-invite me."
)

NO_ADMIN_LEFT_MESSAGE = (
    "👋 The last admin just left this channel, so I'm heading out too. "
    "Ask an admin to rejoin and re-invite me if you still need me."
)

NOT_ALLOWED_CHANNEL_MESSAGE = (
    "👋 This channel isn't on the bot's approved list. Ask a bot admin for help, "
    "then re-invite me. Heading out."
)

NOT_PRIVATE_CHANNEL_MESSAGE = (
    "👋 I only operate in private channels. This one looks public (or I "
    "couldn't verify its privacy). Heading out."
)


def _post_and_leave(channel: str, text: str) -> None:
    """Post a farewell note, then leave. Leaving must happen even if the post fails."""
    try:
        if post_message(channel, text) is None:
            logger.warning(
                "Farewell message was not confirmed; leaving anyway",
                extra={"channel": channel},
            )
    except Exception:
        logger.exception("Farewell message failed; leaving anyway")
    finally:
        leave_channel(channel)


def _handle_member_joined(ev: dict) -> dict:
    """Bot was added to a channel — leave unless all three checks pass:
    channel is on the allowlist, channel is private, and at least one
    allowlisted admin is a member.
    """
    bot_user_id = get_bot_user_id()
    if ev.get("user") != bot_user_id:
        # Some other user joined a channel we're in. Not our concern.
        return {"ok": True}

    channel = ev.get("channel")
    if not channel:
        return {"ok": True}

    if not is_channel_allowed(channel):
        logger.info("Bot joined non-allowlisted channel; leaving", extra={"channel": channel})
        audit("channel_access_denied", channel_id=channel, reason="not_on_allowlist", trigger="member_joined_channel")
        _post_and_leave(channel, NOT_ALLOWED_CHANNEL_MESSAGE)
        return {"ok": True}

    private = is_private_channel(channel)
    if private is not True:
        # Either explicitly public, or privacy couldn't be verified (None). Fail closed.
        logger.info(
            "Bot joined channel that isn't verifiably private; leaving",
            extra={"channel": channel, "is_private": private},
        )
        audit(
            "channel_access_denied",
            channel_id=channel,
            reason="not_private",
            is_private=private,
            trigger="member_joined_channel",
        )
        _post_and_leave(channel, NOT_PRIVATE_CHANNEL_MESSAGE)
        return {"ok": True}

    members = get_channel_members(channel)
    if channel_has_admin(members):
        logger.info("Bot joined approved channel with admin present", extra={"channel": channel})
        return {"ok": True}

    # No admin present — fail closed. (Empty members from API error also lands here, by design.)
    logger.info("Bot joined channel with no admin; leaving", extra={"channel": channel})
    audit("channel_access_denied", channel_id=channel, reason="no_admin", trigger="member_joined_channel")
    _post_and_leave(channel, NO_ADMIN_JOIN_MESSAGE)
    return {"ok": True}


def _handle_member_left(ev: dict) -> dict:
    """An admin left a channel — leave if they were the last one."""
    user = ev.get("user")
    channel = ev.get("channel")
    if not user or not channel:
        return {"ok": True}

    bot_user_id = get_bot_user_id()
    if user == bot_user_id:
        # We left ourselves; nothing more to do.
        return {"ok": True}

    if user not in admin_user_ids():
        # A non-admin left. We don't care.
        return {"ok": True}

    # An admin left. Re-check membership; if no admin remains, we leave too.
    members = get_channel_members(channel)
    if not members:
        # API error or we're no longer a member. Leaving on a transient error
        # would be disruptive, so stay — but make the gap visible: the next
        # app_mention re-checks admin presence and fails closed.
        logger.warning(
            "Could not verify admin presence after admin departure; staying",
            extra={"channel": channel, "departed_admin": user},
        )
        audit(
            "admin_check_unverifiable",
            channel_id=channel,
            departed_admin=user,
            trigger="member_left_channel",
        )
        return {"ok": True}

    if channel_has_admin(members):
        logger.info(
            "Admin left but other admins remain",
            extra={"channel": channel, "departed_admin": user},
        )
        return {"ok": True}

    logger.info(
        "Last admin left channel; bot leaving",
        extra={"channel": channel, "departed_admin": user},
    )
    _post_and_leave(channel, NO_ADMIN_LEFT_MESSAGE)
    return {"ok": True}


def _handle_app_mention(ev: dict) -> dict:
    channel = ev.get("channel")
    trigger_ts = ev.get("ts")
    thread_ts = ev.get("thread_ts") or trigger_ts
    text = _strip_mention(ev.get("text") or "")
    user_id = ev.get("user")

    if not channel or (not text and not ev.get("files")):
        logger.info("Nothing to do — missing channel or text")
        return {"ok": True}

    # Set ambient audit context for the rest of this invocation. Subsequent
    # audit() calls in worker.py AND in tool layers (databricks_tool,
    # github_tool) auto-pick these up via contextvars.
    set_context(user_id=user_id, channel_id=channel, thread_ts=thread_ts)

    # Defense-in-depth: even if the bot is still a member of a channel that
    # was just removed from the allowlist (or whose privacy state changed),
    # refuse silently on the mention itself. The member_joined_channel guard
    # handles the initial-join case; this handles "config drifted while bot
    # was inside".
    if not is_channel_allowed(channel):
        logger.info(
            "Ignoring mention from non-allowlisted channel",
            extra={"channel": channel, "user": user_id},
        )
        audit("channel_access_denied", reason="not_on_allowlist", trigger="app_mention")
        leave_channel(channel)
        return {"ok": True}
    if is_private_channel(channel) is not True:
        logger.info(
            "Ignoring mention from non-private channel",
            extra={"channel": channel, "user": user_id},
        )
        audit("channel_access_denied", reason="not_private", trigger="app_mention")
        leave_channel(channel)
        return {"ok": True}

    # Third check, re-validated per mention: an allowlisted admin must still be
    # a member. An empty member list means the API failed (or we are no longer
    # in the channel) — fail closed by not answering, but do not leave on what
    # may be a transient error.
    members = get_channel_members(channel)
    if not members:
        logger.warning(
            "Could not verify admin presence on mention; not responding",
            extra={"channel": channel, "user": user_id},
        )
        audit("channel_access_denied", reason="admin_unverifiable", trigger="app_mention")
        return {"ok": True}
    if not channel_has_admin(members):
        logger.info(
            "Ignoring mention from channel with no admin present; leaving",
            extra={"channel": channel, "user": user_id},
        )
        audit("channel_access_denied", reason="no_admin", trigger="app_mention")
        _post_and_leave(channel, NO_ADMIN_LEFT_MESSAGE)
        return {"ok": True}

    audit(
        "mention_received",
        prompt=text,
        has_images=bool(ev.get("files")),
        is_thread_followup=bool(ev.get("thread_ts")),
    )

    # Rate-limit checks run AFTER channel access (so non-allowlisted attempts
    # don't burn counter slots) but BEFORE the agent loop (so we don't pay
    # for Anthropic + Databricks calls we're about to refuse). The user
    # check increments the per-user counters atomically; the channel check
    # is independent. Either one blocking returns a single Slack ack and
    # stops the invocation.
    user_allowed, user_reason = check_user(user_id)
    if not user_allowed:
        audit("rate_limit_blocked", scope="user", reason=user_reason)
        post_message(
            channel,
            f"_Rate limit reached for you ({user_reason}). Try again in a bit._",
            thread_ts=thread_ts,
        )
        return {"ok": True}

    channel_allowed, channel_reason = check_channel(channel)
    if not channel_allowed:
        audit("rate_limit_blocked", scope="channel", reason=channel_reason)
        post_message(
            channel,
            f"_Rate limit reached for this channel ({channel_reason}). Try again in a bit._",
            thread_ts=thread_ts,
        )
        return {"ok": True}

    image_blocks = extract_image_blocks(ev)
    if not text and not image_blocks:
        return {"ok": True}
    new_user_msg = _build_new_user_message(text, image_blocks)

    # Three paths to construct the message list, in order of preference:
    # 1. DDB session exists → continue from saved state (full tool_use/tool_result fidelity)
    # 2. No session, but mention is inside an existing thread → bridge from Slack thread history
    #    (lossy — only includes prior text, no tool calls). Used for threads that predate this
    #    feature, or threads whose 365-day TTL has expired.
    # 3. Fresh thread (mention has no thread_ts) → start with just the new user message.

    saved = load_session(thread_ts)
    if saved is not None:
        logger.info("Continuing session", extra={"thread_ts": thread_ts, "prior_msgs": len(saved)})
        messages = saved + [new_user_msg]
    elif ev.get("thread_ts"):
        logger.info("No session found; bridging from Slack thread history", extra={"thread_ts": thread_ts})
        bot_user_id = get_bot_user_id()
        replies = get_thread_replies(channel, thread_ts)
        bridged = _build_message_history(
            replies, bot_user_id, trigger_ts, trigger_image_blocks=image_blocks
        )
        messages = bridged if bridged else [new_user_msg]
    else:
        messages = [new_user_msg]

    metrics = RunMetrics()
    try:
        with job_approvals.request_context(user_id, channel, thread_ts):
            run_agent(messages, channel, thread_ts, metrics=metrics)
    except Exception as e:
        metrics.outcome = "error"
        error_id = _new_error_id()
        visible_error = _safe_agent_error_message(error_id, e)
        logger.exception("Agent run failed: error_id=%s", error_id)
        audit(
            "agent_error",
            error_id=error_id,
            error=str(e),
            error_type=type(e).__name__,
            **_agent_error_fields(e),
        )
        _persist_agent_failure_if_balanced(messages, thread_ts, visible_error)
        post_message(
            channel,
            visible_error,
            thread_ts=thread_ts,
        )
    finally:
        # Emitted on every exit path (answered, max_iterations, deadline,
        # unexpected_stop, max_tokens, refusal, error) so per-run cost and
        # latency are always queryable next to the agent_error record.
        audit("run_summary", **metrics.summary())

    return {"ok": True}


def _handle_feedback_event(event: dict) -> dict:
    channel_id = str(event.get("channel_id") or "")
    thread_ts = str(event.get("thread_ts") or "")
    answer_ts = str(event.get("answer_ts") or "")
    user_id = str(event.get("user_id") or "")
    rating = str(event.get("rating") or "")
    comment = str(event.get("comment") or "")

    set_context(user_id=user_id, channel_id=channel_id, thread_ts=thread_ts)
    try:
        save_feedback(
            channel_id=channel_id,
            thread_ts=thread_ts,
            answer_ts=answer_ts,
            user_id=user_id,
            rating=rating,
            comment=comment,
        )
    except Exception as exc:
        logger.exception("Failed to save answer feedback")
        audit(
            "answer_feedback_error",
            answer_ts=answer_ts,
            rating=rating,
            error_type=type(exc).__name__,
        )
        if channel_id and user_id:
            post_ephemeral(
                channel_id,
                user_id,
                "I couldn't save that feedback. Please try again.",
                thread_ts=thread_ts or None,
            )
        return {"ok": False}

    audit(
        "answer_feedback",
        answer_ts=answer_ts,
        rating=rating,
        has_comment=bool(comment.strip()),
    )
    label = RATING_LABELS[rating]
    post_ephemeral(
        channel_id,
        user_id,
        f"Thanks — your “{label}” feedback was saved. "
        "Choose another rating anytime to update it.",
        thread_ts=thread_ts,
    )
    return {"ok": True}


def _set_invocation_deadline(context) -> None:
    """Derive the invocation deadline from the Lambda context, if it offers one."""
    get_remaining = getattr(context, "get_remaining_time_in_millis", None)
    if not callable(get_remaining):
        return
    try:
        remaining_ms = float(get_remaining())
    except Exception:
        logger.warning("Could not read remaining Lambda time; running without a deadline")
        return
    deadline.set_deadline_from_remaining(
        remaining_ms / 1000.0,
        safety_margin=LAMBDA_DEADLINE_SAFETY_MARGIN_SECONDS,
    )


def _dispatch_event(event: dict) -> dict:
    if event.get("job_confirmation_event"):
        return _handle_job_confirmation(event["job_confirmation_event"])
    feedback_event = event.get("feedback_event")
    if feedback_event:
        return _handle_feedback_event(feedback_event)

    payload = event.get("slack_event") or {}
    ev = payload.get("event") or {}
    ev_type = ev.get("type")

    if ev_type == "member_joined_channel":
        return _handle_member_joined(ev)
    if ev_type == "member_left_channel":
        return _handle_member_left(ev)
    return _handle_app_mention(ev)


def lambda_handler(event, context):
    logger.info("Worker invoked")
    _set_invocation_deadline(context)
    try:
        return _dispatch_event(event)
    finally:
        deadline.clear()
