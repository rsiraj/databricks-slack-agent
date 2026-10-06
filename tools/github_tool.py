"""GitHub tool — scoped to a single data-warehouse repo, configured via env vars.

Every GitHub call goes through ``_github_request``:

  - URL path segments (file paths, refs, SHAs) are validated and percent-encoded
    before they are placed in a URL; ``ref`` is always sent as a query parameter.
  - Connection errors, timeouts, and 5xx responses are retried up to three
    attempts with exponential backoff; sleeps never outlive the invocation
    deadline in ``utils.deadline``.
  - A 403/429 carrying ``Retry-After`` or ``X-RateLimit-Remaining: 0`` waits
    once (bounded by the deadline and a 30s cap) and retries; a second hit is
    surfaced as a structured ``rate_limited`` error instead of an exception.
  - A 401 refreshes the cached app secret once and retries.
  - Any other non-2xx response becomes ``{"error": ..., "status_code": ...}``
    with the GitHub ``message`` so the model sees something actionable.
"""

import functools
import logging
import os
import re
import secrets
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import requests

from utils import deadline as run_deadline
from utils.audit import audit

logger = logging.getLogger(__name__)

# Set GITHUB_REPO on the worker Lambda to the `<owner>/<name>` of the repo where
# your team's data-warehouse SQL / notebooks live. The default is a placeholder.
REPO = os.environ.get("GITHUB_REPO", "example-org/example-data-repo")
DEFAULT_REF = os.environ.get("GITHUB_DEFAULT_REF", "main")
GITHUB_API = "https://api.github.com"
TREE_CACHE_TTL_SECONDS = 300
_tree_cache: dict[str, Any] = {"paths": None, "truncated": False, "expires_at": 0.0}

MAX_HTTP_ATTEMPTS = 3
MAX_RETRY_DELAY_SECONDS = 10.0
MAX_RATE_LIMIT_WAIT_SECONDS = 30.0
RETRYABLE_STATUS_CODES = frozenset(range(500, 600))
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
# Seconds of the invocation deadline always left for the model to respond.
RESPONSE_RESERVE_SECONDS = 30.0
# GitHub's compare endpoint returns at most 300 changed files.
COMPARE_FILE_CAP = 300
SEARCH_PER_PAGE = 20

REF_PATTERN = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")

GITHUB_TOOLS = [
    {
        "name": "github_search_code",
        "description": (
            f"Search code in {REPO}. Use this to locate notebooks or SQL files "
            "that define a table, column, metric, or transformation. Results are scoped to this "
            "repository automatically — do not include `repo:` in your query. Only the first "
            f"{SEARCH_PER_PAGE} matches are returned; `truncated` is true when more exist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "GitHub code-search query (without the `repo:` qualifier). "
                        "Use plain search terms plus supported qualifiers; do not put a standalone "
                        "hyphen between terms. Examples: 'loan_application_fact path:notebooks', "
                        "'Orders Merge path:notebooks', 'funnel_conversion language:sql'."
                    ),
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "github_get_file_content",
        "description": (
            f"Retrieve the raw contents of a file from {REPO} at `{DEFAULT_REF}`. "
            "Use after github_search_code to inspect the notebook / SQL that defines the logic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file within the repo.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "github_find_files",
        "description": (
            f"Find file paths in {REPO} by filename or path substring without using GitHub "
            "code-search syntax. Use this when locating a known or partially known filename; "
            "then call github_get_file_content to read the selected file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Case-insensitive filename or path substring.",
                },
                "path_prefix": {
                    "type": "string",
                    "description": "Optional path prefix such as 'notebooks/' or 'resources/jobs/'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum matching paths to return (default 20, max 100).",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "github_get_commits",
        "description": (
            f"List commits affecting a given file path on `{DEFAULT_REF}`. Use for Change Tracking "
            "(when was this logic last updated, what changed recently)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path within the repo."},
                "per_page": {
                    "type": "integer",
                    "description": "Max commits to return (default 10, max 100).",
                    "default": 10,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "github_compare_commits",
        "description": (
            f"Compare two commits (or refs) on {REPO} to see what changed. "
            "Use for Diff Mode and Root Cause Analysis. `truncated` is true when GitHub capped "
            f"the changed-file list at {COMPARE_FILE_CAP} files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "base": {"type": "string", "description": "Base commit SHA or ref."},
                "head": {"type": "string", "description": "Head commit SHA or ref."},
            },
            "required": ["base", "head"],
        },
    },
]


class GitHubRequestError(Exception):
    """Carries a structured tool error dict to the tool boundary."""

    def __init__(self, error: dict[str, Any]):
        super().__init__(str(error.get("error", "GitHub request failed")))
        self.error = error


def _structured_errors(fn: Callable[..., dict]) -> Callable[..., dict]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            return fn(*args, **kwargs)
        except GitHubRequestError as exc:
            return dict(exc.error)

    return wrapper


def _headers(accept: str = "application/vnd.github+json") -> dict:
    from utils.secrets import get_secret
    return {
        "Accept": accept,
        "Authorization": f"Bearer {get_secret('github_token')}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _refresh_github_token() -> None:
    """Drop the cached app secret so the next ``_headers()`` call re-reads it."""
    try:
        from utils import secrets as app_secrets

        app_secrets._load_app_secret(force_refresh=True)
    except Exception:  # pragma: no cover - refresh is best effort
        logger.exception("Failed to refresh the GitHub token after a 401")


# ---------------------------------------------------------------------------
# Argument safety — nothing model-controlled reaches a URL path unvalidated.
# ---------------------------------------------------------------------------


def _invalid_argument(argument: str, message: str) -> dict:
    return {
        "error": message,
        "validation_failure": "invalid_argument",
        "argument": argument,
    }


def _safe_path_segment(value: Any, *, pattern: re.Pattern[str]) -> str | None:
    """Return ``value`` when it is a string fully matching ``pattern``, else None."""
    if not isinstance(value, str):
        return None
    return value if pattern.match(value) else None


def _safe_ref(value: Any) -> str | None:
    """Branch, tag, or SHA usable as a URL path segment; None when unsafe."""
    ref = _safe_path_segment(value, pattern=REF_PATTERN)
    if ref is None or ".." in ref or "//" in ref or ref.startswith("/") or ref.endswith("/"):
        return None
    return ref


def _safe_repo_path(value: Any) -> str | None:
    """Return a percent-encoded repository path, or None when the path is unsafe.

    Rejects `..` segments, empty segments, query/fragment markers, backslashes,
    control characters, and leading slashes; each remaining segment is encoded
    with ``quote(seg, safe="")`` so it can only ever address one blob.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    if value.startswith("/") or any(ch in value for ch in "?#\\") or any(ord(ch) < 32 for ch in value):
        return None
    segments = value.split("/")
    if any(seg in {"", ".", ".."} for seg in segments):
        return None
    return "/".join(quote(seg, safe="") for seg in segments)


def _default_ref() -> str:
    ref = _safe_ref(DEFAULT_REF)
    if ref is None:
        raise GitHubRequestError(
            {
                "error": (
                    "GITHUB_DEFAULT_REF is not a valid git ref. Ask in #data-assistant-help to fix "
                    "the worker configuration."
                ),
                "config_failure": "invalid_default_ref",
            }
        )
    return ref


def _clamp_per_page(value: Any, default: int) -> int | None:
    try:
        per_page = int(default if value is None else value)
    except (TypeError, ValueError):
        return None
    return max(1, min(per_page, 100))


# ---------------------------------------------------------------------------
# Shared request helper — retries, rate limits, auth refresh, deadline.
# ---------------------------------------------------------------------------


def _retry_budget_seconds() -> float | None:
    remaining = run_deadline.remaining_seconds()
    if remaining is None:
        return None
    return max(0.0, remaining - RESPONSE_RESERVE_SECONDS)


def _can_wait(delay_seconds: float) -> bool:
    budget = _retry_budget_seconds()
    return budget is None or delay_seconds <= budget


def _request_timeout(requested: float) -> float:
    budget = _retry_budget_seconds()
    if budget is None:
        return requested
    return run_deadline.bounded_timeout(min(requested, max(1.0, budget)), floor=1.0)


def _retry_delay_seconds(retry_number: int) -> float:
    exponential = min(2 ** max(retry_number - 1, 0), MAX_RETRY_DELAY_SECONDS)
    jitter_window_ms = int(min(0.25, MAX_RETRY_DELAY_SECONDS - exponential) * 1000)
    jitter = secrets.randbelow(jitter_window_ms + 1) / 1000 if jitter_window_ms else 0.0
    return exponential + jitter


def _header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None) or {}
    value = headers.get(name)
    if value is None:
        # requests' CaseInsensitiveDict handles this; plain dicts in tests may not.
        lowered = {str(k).lower(): v for k, v in headers.items()}
        value = lowered.get(name.lower())
    return None if value is None else str(value)


def _rate_limit_wait_seconds(response: Any) -> float | None:
    """Seconds to wait when the response is a GitHub rate-limit hit, else None."""
    if getattr(response, "status_code", None) not in {403, 429}:
        return None
    retry_after = _header(response, "Retry-After")
    remaining = _header(response, "X-RateLimit-Remaining")
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            return MAX_RATE_LIMIT_WAIT_SECONDS
    if remaining is not None and remaining.strip() == "0":
        reset = _header(response, "X-RateLimit-Reset")
        try:
            return max(0.0, float(reset) - time.time()) if reset else MAX_RATE_LIMIT_WAIT_SECONDS
        except ValueError:
            return MAX_RATE_LIMIT_WAIT_SECONDS
    return None


def _response_message(response: Any) -> str | None:
    try:
        payload = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        message = payload.get("message")
        errors = payload.get("errors")
        if isinstance(errors, list) and errors:
            details = "; ".join(
                str(err.get("message") or err.get("code") or err) if isinstance(err, dict) else str(err)
                for err in errors[:3]
            )
            message = f"{message} ({details})" if message else details
        return str(message).strip()[:500] if message else None
    text = getattr(response, "text", None)
    return text.strip()[:300] if isinstance(text, str) and text.strip() else None


def _http_error(response: Any, operation: str) -> dict:
    status = response.status_code
    message = _response_message(response)
    if status in RETRYABLE_STATUS_CODES:
        text = f"GitHub {operation} is temporarily unavailable (HTTP {status} after {MAX_HTTP_ATTEMPTS} attempts)"
    elif status in {401, 403}:
        text = f"GitHub {operation} was refused (HTTP {status})"
    elif status == 404:
        text = f"GitHub {operation} target was not found (HTTP 404)"
    else:
        text = f"GitHub {operation} request failed (HTTP {status})"
    if message:
        text = f"{text}: {message}"
    return {"error": text, "status_code": status}


def _github_request(
    method: str,
    url: str,
    *,
    operation: str,
    accept: str = "application/vnd.github+json",
    allow_statuses: tuple[int, ...] = (),
    **kwargs: Any,
) -> requests.Response:
    """Send one GitHub API request with bounded retries.

    Returns the response for 2xx and for any status in ``allow_statuses`` (so
    callers can give a tailored answer for 404 / 422). Every other outcome is
    raised as ``GitHubRequestError`` carrying a structured error dict.
    """
    requested_timeout = kwargs.pop("timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS)
    auth_refreshed = False
    rate_limit_waited = False
    transient_failure_count = 0
    attempt = 0
    while True:
        attempt += 1
        kwargs["timeout"] = _request_timeout(requested_timeout)
        try:
            response = requests.request(method, url, headers=_headers(accept), **kwargs)
        except requests.exceptions.SSLError:
            raise
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            transient_failure_count += 1
            delay = _retry_delay_seconds(attempt)
            will_retry = attempt < MAX_HTTP_ATTEMPTS and _can_wait(delay)
            audit(
                "github_transient_failure",
                operation=operation,
                attempt=attempt,
                max_attempts=MAX_HTTP_ATTEMPTS,
                failure_type=type(exc).__name__,
                status_code=None,
                will_retry=will_retry,
                retry_delay_seconds=delay if will_retry else None,
            )
            if not will_retry:
                raise GitHubRequestError(
                    {
                        "error": (
                            f"GitHub {operation} did not respond after {attempt} attempt(s) "
                            f"({type(exc).__name__}). Try again shortly."
                        ),
                        "status_code": None,
                        "transient_failure": type(exc).__name__,
                    }
                ) from exc
            logger.warning(
                "Transient GitHub network failure; retrying attempt %d/%d in %.2fs",
                attempt + 1,
                MAX_HTTP_ATTEMPTS,
                delay,
            )
            time.sleep(delay)
            continue

        status = response.status_code

        if status == 401 and not auth_refreshed:
            logger.warning("GitHub returned 401 — refreshing cached app secret and retrying")
            _refresh_github_token()
            auth_refreshed = True
            continue

        rate_limit_wait = _rate_limit_wait_seconds(response)
        if rate_limit_wait is not None:
            # min(Retry-After, 30s, deadline budget) — the budget keeps the
            # response reserve untouched so the model can still answer.
            wait = run_deadline.bounded_timeout(
                min(rate_limit_wait, MAX_RATE_LIMIT_WAIT_SECONDS), floor=0.0
            )
            budget = _retry_budget_seconds()
            if budget is not None:
                wait = min(wait, budget)
            will_retry = not rate_limit_waited and (budget is None or budget > 0)
            audit(
                "github_rate_limited",
                operation=operation,
                status_code=status,
                retry_after_seconds=rate_limit_wait,
                wait_seconds=wait if will_retry else None,
                will_retry=will_retry,
            )
            if will_retry:
                rate_limit_waited = True
                if wait > 0:
                    time.sleep(wait)
                continue
            raise GitHubRequestError(
                {
                    "error": (
                        f"GitHub API rate limit reached during {operation}; retry in about "
                        f"{int(rate_limit_wait)}s. Reuse results already retrieved in this thread."
                    ),
                    "status_code": status,
                    "rate_limited": True,
                    "retry_after_seconds": rate_limit_wait,
                }
            )

        if status in RETRYABLE_STATUS_CODES:
            transient_failure_count += 1
            delay = _retry_delay_seconds(attempt)
            will_retry = attempt < MAX_HTTP_ATTEMPTS and _can_wait(delay)
            audit(
                "github_transient_failure",
                operation=operation,
                attempt=attempt,
                max_attempts=MAX_HTTP_ATTEMPTS,
                failure_type="http",
                status_code=status,
                will_retry=will_retry,
                retry_delay_seconds=delay if will_retry else None,
            )
            if will_retry:
                logger.warning(
                    "GitHub returned transient HTTP %d; retrying attempt %d/%d in %.2fs",
                    status,
                    attempt + 1,
                    MAX_HTTP_ATTEMPTS,
                    delay,
                )
                time.sleep(delay)
                continue
            raise GitHubRequestError(_http_error(response, operation))

        if transient_failure_count and status < 400:
            audit(
                "github_recovery_succeeded",
                operation=operation,
                attempts=attempt,
                transient_failure_count=transient_failure_count,
            )

        if status >= 400 and status not in allow_statuses:
            raise GitHubRequestError(_http_error(response, operation))
        return response


def _truncate(text: str, limit: int = 60_000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... truncated, {len(text) - limit} more chars ...]"


def _normalize_search_query(query: str) -> str:
    """Remove syntax that commonly makes GitHub code search return HTTP 422."""
    without_repo = re.sub(r"(?:^|\s)repo:\S+", " ", query or "", flags=re.IGNORECASE)
    without_bare_operators = re.sub(r"(?<!\S)[+-](?!\S)", " ", without_repo)
    return re.sub(r"\s+", " ", without_bare_operators).strip()


def _fallback_search_query(query: str) -> str:
    """Build a conservative terms-and-qualifiers query after a 422 response."""
    qualifier_pattern = re.compile(
        r"\b(?:path|language|extension|filename):(?:\"[^\"]+\"|\S+)",
        flags=re.IGNORECASE,
    )
    qualifiers = qualifier_pattern.findall(query)
    text = qualifier_pattern.sub(" ", query)
    terms = re.findall(r"[A-Za-z0-9_./]+", text)
    return " ".join([*terms, *qualifiers]).strip()


def _code_search_request(query: str) -> requests.Response:
    return _github_request(
        "GET",
        f"{GITHUB_API}/search/code",
        operation="search_code",
        params={"q": f"{query} repo:{REPO}", "per_page": SEARCH_PER_PAGE},
        allow_statuses=(422,),
        timeout=30,
    )


@_structured_errors
def search_code(query: str) -> dict:
    normalized = _normalize_search_query(query)
    if not normalized:
        return {
            "total_count": 0,
            "items": [],
            "truncated": False,
            "warning": "GitHub search needs at least one plain search term or supported qualifier.",
        }

    attempted_query = normalized
    r = _code_search_request(attempted_query)
    if r.status_code == 422:
        fallback = _fallback_search_query(normalized)
        if fallback and fallback != attempted_query:
            attempted_query = fallback
            r = _code_search_request(attempted_query)

    if r.status_code == 422:
        return {
            "total_count": 0,
            "items": [],
            "truncated": False,
            "warning": (
                "GitHub rejected the search syntax after safe normalization. "
                "Retry with fewer plain terms or one path/language qualifier."
            ),
            "normalized_query": attempted_query,
        }

    data = r.json()
    items = [
        {"path": it["path"], "sha": it["sha"], "url": it["html_url"]}
        for it in data.get("items", [])
    ]
    total_count = data.get("total_count", 0) or 0
    result = {
        "total_count": total_count,
        "items": items,
        "truncated": total_count > len(items),
    }
    if attempted_query != query:
        result["normalized_query"] = attempted_query
    return result


def _repository_paths() -> tuple[list[str], bool]:
    now = time.time()
    cached_paths = _tree_cache.get("paths")
    if isinstance(cached_paths, list) and _tree_cache["expires_at"] > now:
        return cached_paths, bool(_tree_cache.get("truncated"))

    ref = _default_ref()
    r = _github_request(
        "GET",
        f"{GITHUB_API}/repos/{REPO}/git/trees/{quote(ref, safe='')}",
        operation="find_files",
        params={"recursive": "1"},
        timeout=60,
    )
    payload = r.json()
    paths = [
        item["path"]
        for item in payload.get("tree", [])
        if item.get("type") == "blob" and isinstance(item.get("path"), str)
    ]
    _tree_cache.update(
        {
            "paths": paths,
            "truncated": bool(payload.get("truncated")),
            "expires_at": now + TREE_CACHE_TTL_SECONDS,
        }
    )
    return paths, bool(payload.get("truncated"))


@_structured_errors
def find_files(name: str, path_prefix: str | None = None, limit: int = 20) -> dict:
    needle = (name or "").strip().lower()
    if not needle:
        return {
            "matches": [],
            "error": "name must contain a filename or path substring",
            "validation_failure": "empty_file_search",
        }

    prefix = (path_prefix or "").strip().lstrip("/")
    capped_limit = _clamp_per_page(limit, 20)
    if capped_limit is None:
        return _invalid_argument("limit", "limit must be an integer between 1 and 100.")
    paths, source_truncated = _repository_paths()
    matches = [
        path
        for path in paths
        if needle in path.lower() and (not prefix or path.startswith(prefix))
    ]
    matches.sort(key=lambda path: (needle not in path.rsplit("/", 1)[-1].lower(), len(path), path))
    return {
        "query": name,
        "path_prefix": prefix or None,
        "match_count": len(matches),
        "matches": matches[:capped_limit],
        "truncated": len(matches) > capped_limit,
        "source_tree_truncated": source_truncated,
    }


@_structured_errors
def get_file_content(path: str) -> dict:
    encoded_path = _safe_repo_path(path)
    if encoded_path is None:
        return _invalid_argument(
            "path",
            "path must be a relative repository file path without '..', '?', '#', or "
            "backslashes (for example 'notebooks/Orders - Merge.sql').",
        )
    ref = _default_ref()
    r = _github_request(
        "GET",
        f"{GITHUB_API}/repos/{REPO}/contents/{encoded_path}",
        operation="get_file_content",
        accept="application/vnd.github.raw+json",
        params={"ref": ref},
        allow_statuses=(404,),
        timeout=30,
    )
    if r.status_code == 404:
        return {"error": f"File not found on {ref}: {path}", "status_code": 404}
    return {"path": path, "ref": ref, "content": _truncate(r.text)}


@_structured_errors
def get_commits(path: str, per_page: int = 10) -> dict:
    if _safe_repo_path(path) is None:
        return _invalid_argument(
            "path",
            "path must be a relative repository file path without '..', '?', '#', or backslashes.",
        )
    capped = _clamp_per_page(per_page, 10)
    if capped is None:
        return _invalid_argument("per_page", "per_page must be an integer between 1 and 100.")
    ref = _default_ref()
    r = _github_request(
        "GET",
        f"{GITHUB_API}/repos/{REPO}/commits",
        operation="get_commits",
        params={"path": path, "sha": ref, "per_page": capped},
        timeout=30,
    )
    commits = [
        {
            "sha": c["sha"],
            "author": (c.get("commit", {}).get("author") or {}).get("name"),
            "date": (c.get("commit", {}).get("author") or {}).get("date"),
            "message": (c.get("commit", {}).get("message") or "").split("\n")[0],
            "url": c.get("html_url"),
        }
        for c in r.json()
    ]
    return {"path": path, "commits": commits, "truncated": len(commits) >= capped}


@_structured_errors
def compare_commits(base: str, head: str) -> dict:
    safe_base = _safe_ref(base)
    safe_head = _safe_ref(head)
    if safe_base is None or safe_head is None:
        bad = "base" if safe_base is None else "head"
        return _invalid_argument(
            bad,
            f"{bad} must be a commit SHA, branch, or tag made of letters, digits, '.', '_', "
            "'-' or '/', without '..' or '//'.",
        )
    r = _github_request(
        "GET",
        f"{GITHUB_API}/repos/{REPO}/compare/{quote(safe_base, safe='')}...{quote(safe_head, safe='')}",
        operation="compare_commits",
        timeout=60,
    )
    data = r.json()
    raw_files = data.get("files", []) or []
    files = [
        {
            "filename": f["filename"],
            "status": f["status"],
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "patch": _truncate(f.get("patch", ""), limit=20_000),
        }
        for f in raw_files
    ]
    return {
        "base": base,
        "head": head,
        "ahead_by": data.get("ahead_by"),
        "behind_by": data.get("behind_by"),
        "total_commits": data.get("total_commits"),
        "files": files,
        "truncated": len(raw_files) >= COMPARE_FILE_CAP or bool(data.get("truncated")),
    }


def dispatch(name: str, inp: dict[str, Any]) -> dict:
    try:
        return _dispatch(name, inp)
    except GitHubRequestError as exc:
        return dict(exc.error)


def _dispatch(name: str, inp: dict[str, Any]) -> dict:
    if name == "github_search_code":
        return search_code(inp["query"])
    if name == "github_get_file_content":
        return get_file_content(inp["path"])
    if name == "github_find_files":
        return find_files(inp["name"], inp.get("path_prefix"), inp.get("limit", 20))
    if name == "github_get_commits":
        return get_commits(inp["path"], inp.get("per_page", 10))
    if name == "github_compare_commits":
        return compare_commits(inp["base"], inp["head"])
    return {"error": f"Unknown github tool: {name}"}
