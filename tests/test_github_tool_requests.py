"""tools/github_tool.py — URL path safety, retries, rate limits, auth refresh."""

import json

import pytest
import requests

from tools import github_tool as gt
from utils import deadline as run_deadline


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text if text is not None else (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


@pytest.fixture(autouse=True)
def clear_deadline():
    run_deadline.clear()
    yield
    run_deadline.clear()


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(gt, "_headers", lambda *args, **kwargs: {"Authorization": "Bearer secret"})
    monkeypatch.setattr(gt.secrets, "randbelow", lambda upper: 0)
    state = {"sleeps": [], "audits": [], "calls": []}
    monkeypatch.setattr(gt.time, "sleep", state["sleeps"].append)
    monkeypatch.setattr(gt, "audit", lambda event, **fields: state["audits"].append((event, fields)))
    monkeypatch.setattr(gt, "DEFAULT_REF", "main")
    gt._tree_cache.update({"paths": None, "truncated": False, "expires_at": 0.0})

    def install(responses):
        queue = list(responses)

        def fake_request(method, url, **kwargs):
            state["calls"].append((method, url, kwargs))
            outcome = queue.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(gt.requests, "request", fake_request)

    state["install"] = install
    install([])
    return state


# ---------------------------------------------------------------------------
# URL path safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "../../repos/x",
        "notebooks/../../../x",
        "README.md?ref=feature",
        "README.md#frag",
        "/notebooks/x.sql",
        "notebooks\\x.sql",
        "notebooks//x.sql",
        "notebooks/./x.sql",
        "",
        "   ",
        "a\nb",
    ],
)
def test_get_file_content_rejects_unsafe_paths_without_a_request(env, path):
    result = gt.get_file_content(path)

    assert result["validation_failure"] == "invalid_argument"
    assert result["argument"] == "path"
    assert env["calls"] == []


def test_get_file_content_encodes_each_segment_and_passes_ref_as_param(env):
    env["install"]([FakeResponse(200, text="SELECT 1")])

    result = gt.get_file_content("notebooks/Orders - Merge/v2 (1)&x.sql")

    method, url, kwargs = env["calls"][0]
    assert method == "GET"
    assert url == (
        f"{gt.GITHUB_API}/repos/{gt.REPO}/contents/notebooks/Orders%20-%20Merge/v2%20%281%29%26x.sql"
    )
    assert "ref=" not in url
    assert kwargs["params"] == {"ref": "main"}
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    assert result == {
        "path": "notebooks/Orders - Merge/v2 (1)&x.sql",
        "ref": "main",
        "content": "SELECT 1",
    }


def test_get_file_content_uses_configured_default_ref(env, monkeypatch):
    monkeypatch.setattr(gt, "DEFAULT_REF", "release/2026-09")
    env["install"]([FakeResponse(200, text="x")])

    result = gt.get_file_content("README.md")

    assert env["calls"][0][2]["params"] == {"ref": "release/2026-09"}
    assert result["ref"] == "release/2026-09"


def test_invalid_default_ref_is_a_structured_config_error(env, monkeypatch):
    monkeypatch.setattr(gt, "DEFAULT_REF", "main;rm -rf")

    result = gt.get_file_content("README.md")

    assert result["config_failure"] == "invalid_default_ref"
    assert env["calls"] == []


def test_get_file_content_404_is_structured(env):
    env["install"]([FakeResponse(404, {"message": "Not Found"})])

    result = gt.get_file_content("missing.sql")

    assert result == {"error": "File not found on main: missing.sql", "status_code": 404}


@pytest.mark.parametrize(
    "base,head,bad",
    [
        ("main..feature", "abc1234", "base"),
        ("main", "feature//x", "head"),
        ("main", "abc?x=1", "head"),
        ("../x", "main", "base"),
        ("main", "", "head"),
        ("main", "a" * 201, "head"),
    ],
)
def test_compare_commits_rejects_unsafe_refs(env, base, head, bad):
    result = gt.compare_commits(base, head)

    assert result["validation_failure"] == "invalid_argument"
    assert result["argument"] == bad
    assert env["calls"] == []


def test_compare_commits_encodes_refs(env):
    env["install"]([FakeResponse(200, {"files": [], "ahead_by": 1, "total_commits": 1})])

    result = gt.compare_commits("feature/x", "abc1234")

    assert env["calls"][0][1] == f"{gt.GITHUB_API}/repos/{gt.REPO}/compare/feature%2Fx...abc1234"
    assert result["truncated"] is False


def test_tree_fetch_encodes_default_ref(env, monkeypatch):
    monkeypatch.setattr(gt, "DEFAULT_REF", "release/1")
    env["install"]([FakeResponse(200, {"tree": [{"type": "blob", "path": "a.sql"}]})])

    gt.find_files("a.sql")

    assert env["calls"][0][1].endswith("/git/trees/release%2F1")


# ---------------------------------------------------------------------------
# Retries, rate limits, auth refresh
# ---------------------------------------------------------------------------


def test_5xx_is_retried_with_backoff_then_recovers(env):
    env["install"]([FakeResponse(502), FakeResponse(503), FakeResponse(200, [])])

    result = gt.get_commits("notebooks/a.sql")

    assert result["commits"] == []
    assert len(env["calls"]) == 3
    assert env["sleeps"] == [1.0, 2.0]
    assert [event for event, _ in env["audits"]] == [
        "github_transient_failure",
        "github_transient_failure",
        "github_recovery_succeeded",
    ]
    assert env["audits"][0][1]["status_code"] == 502
    assert env["audits"][-1][1]["transient_failure_count"] == 2


def test_exhausted_5xx_returns_structured_error(env):
    env["install"]([FakeResponse(502, {"message": "Server Error"})] * 3)

    result = gt.get_commits("notebooks/a.sql")

    assert len(env["calls"]) == 3
    assert result["status_code"] == 502
    assert "temporarily unavailable" in result["error"]
    assert "Server Error" in result["error"]
    assert env["audits"][-1][1]["will_retry"] is False


def test_connection_errors_are_retried_then_surfaced(env):
    env["install"](
        [requests.ConnectionError("reset"), requests.ReadTimeout("slow"), requests.ConnectionError("reset")]
    )

    result = gt.search_code("signups path:notebooks")

    assert len(env["calls"]) == 3
    assert env["sleeps"] == [1.0, 2.0]
    assert result["transient_failure"] == "ConnectionError"
    assert result["status_code"] is None
    assert "did not respond" in result["error"]


def test_tls_errors_are_not_retried(env):
    env["install"]([requests.exceptions.SSLError("bad cert")])

    with pytest.raises(requests.exceptions.SSLError):
        gt.get_commits("a.sql")

    assert len(env["calls"]) == 1


def test_rate_limit_waits_once_then_retries(env):
    env["install"](
        [
            FakeResponse(403, {"message": "API rate limit exceeded"}, headers={"Retry-After": "2"}),
            FakeResponse(200, {"total_count": 0, "items": []}),
        ]
    )

    result = gt.search_code("signups")

    assert result["items"] == []
    assert env["sleeps"] == [2.0]
    rate_events = [fields for event, fields in env["audits"] if event == "github_rate_limited"]
    assert rate_events[0]["will_retry"] is True
    assert rate_events[0]["wait_seconds"] == 2.0


def test_rate_limit_uses_reset_header_and_caps_wait(env, monkeypatch):
    monkeypatch.setattr(gt.time, "time", lambda: 1_000.0)
    env["install"](
        [
            FakeResponse(
                429,
                {"message": "slow down"},
                headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1120"},
            ),
            FakeResponse(200, {"total_count": 0, "items": []}),
        ]
    )

    gt.search_code("signups")

    assert env["sleeps"] == [gt.MAX_RATE_LIMIT_WAIT_SECONDS]


def test_rate_limit_wait_is_bounded_by_deadline_budget(env):
    run_deadline.set_deadline_from_remaining(gt.RESPONSE_RESERVE_SECONDS + 7)
    env["install"](
        [
            FakeResponse(403, headers={"Retry-After": "120"}),
            FakeResponse(200, {"total_count": 0, "items": []}),
        ]
    )

    gt.search_code("signups")

    assert len(env["sleeps"]) == 1
    assert 6.5 <= env["sleeps"][0] <= 7.0


def test_rate_limit_with_no_budget_fails_fast(env):
    run_deadline.set_deadline_from_remaining(gt.RESPONSE_RESERVE_SECONDS)
    env["install"]([FakeResponse(403, headers={"Retry-After": "5"})])

    result = gt.search_code("signups")

    assert env["sleeps"] == []
    assert result["rate_limited"] is True


def test_second_rate_limit_hit_is_a_structured_error(env):
    env["install"](
        [
            FakeResponse(403, {"message": "API rate limit exceeded"}, headers={"Retry-After": "1"}),
            FakeResponse(403, {"message": "API rate limit exceeded"}, headers={"Retry-After": "40"}),
        ]
    )

    result = gt.get_commits("a.sql")

    assert env["sleeps"] == [1.0]
    assert result["rate_limited"] is True
    assert result["status_code"] == 403
    assert result["retry_after_seconds"] == 40.0
    assert "rate limit" in result["error"]


def test_plain_403_is_not_treated_as_rate_limit(env):
    env["install"]([FakeResponse(403, {"message": "Resource not accessible by personal access token"})])

    result = gt.get_commits("a.sql")

    assert env["sleeps"] == []
    assert "rate_limited" not in result
    assert result["status_code"] == 403
    assert "Resource not accessible" in result["error"]


def test_401_refreshes_token_once_and_retries(env, monkeypatch):
    refreshes = []
    monkeypatch.setattr(gt, "_refresh_github_token", lambda: refreshes.append(True))
    env["install"]([FakeResponse(401, {"message": "Bad credentials"}), FakeResponse(200, [])])

    result = gt.get_commits("a.sql")

    assert refreshes == [True]
    assert len(env["calls"]) == 2
    assert result["commits"] == []


def test_repeated_401_is_a_structured_error(env, monkeypatch):
    refreshes = []
    monkeypatch.setattr(gt, "_refresh_github_token", lambda: refreshes.append(True))
    env["install"]([FakeResponse(401, {"message": "Bad credentials"})] * 2)

    result = gt.get_commits("a.sql")

    assert refreshes == [True]
    assert result["status_code"] == 401
    assert "Bad credentials" in result["error"]


def test_token_refresh_forces_app_secret_reload(monkeypatch):
    from utils import secrets as app_secrets

    calls = []
    monkeypatch.setattr(
        app_secrets, "_load_app_secret", lambda force_refresh=False: calls.append(force_refresh) or {}
    )

    gt._refresh_github_token()

    assert calls == [True]


def test_generic_4xx_includes_github_message(env):
    env["install"](
        [
            FakeResponse(
                422,
                {
                    "message": "Validation Failed",
                    "errors": [{"resource": "Search", "code": "invalid", "message": "bad qualifier"}],
                },
            )
        ]
    )

    result = gt.get_commits("a.sql")

    assert result["status_code"] == 422
    assert "Validation Failed" in result["error"]
    assert "bad qualifier" in result["error"]


# ---------------------------------------------------------------------------
# Truncation flags and per_page handling
# ---------------------------------------------------------------------------


def test_search_code_reports_truncation(env):
    items = [{"path": f"n/{i}.sql", "sha": str(i), "html_url": f"u{i}"} for i in range(20)]
    env["install"]([FakeResponse(200, {"total_count": 57, "items": items})])

    result = gt.search_code("signups")

    assert result["total_count"] == 57
    assert len(result["items"]) == 20
    assert result["truncated"] is True
    assert env["calls"][0][2]["params"]["per_page"] == gt.SEARCH_PER_PAGE


def test_search_code_not_truncated_when_all_returned(env):
    env["install"]([FakeResponse(200, {"total_count": 1, "items": [{"path": "a", "sha": "b", "html_url": "c"}]})])

    assert gt.search_code("signups")["truncated"] is False


def test_compare_commits_flags_file_cap(env):
    files = [{"filename": f"f{i}", "status": "modified", "patch": ""} for i in range(300)]
    env["install"]([FakeResponse(200, {"files": files, "total_commits": 3})])

    result = gt.compare_commits("abc1234", "def5678")

    assert result["truncated"] is True
    assert len(result["files"]) == 300


@pytest.mark.parametrize("per_page,expected", [("250", 100), (0, 1), (-5, 1), (37, 37), (None, 10), (100.7, 100)])
def test_get_commits_coerces_and_clamps_per_page(env, per_page, expected):
    env["install"]([FakeResponse(200, [])])

    gt.get_commits("a.sql", per_page=per_page)

    assert env["calls"][0][2]["params"] == {"path": "a.sql", "sha": "main", "per_page": expected}


def test_get_commits_rejects_non_numeric_per_page(env):
    result = gt.get_commits("a.sql", per_page="lots")

    assert result["validation_failure"] == "invalid_argument"
    assert result["argument"] == "per_page"
    assert env["calls"] == []


def test_get_commits_rejects_traversal_path(env):
    result = gt.get_commits("../secrets.yml")

    assert result["validation_failure"] == "invalid_argument"
    assert env["calls"] == []


def test_dispatch_converts_request_errors_to_dicts(env):
    env["install"]([FakeResponse(500, {"message": "boom"})] * 3)

    result = gt.dispatch("github_get_commits", {"path": "a.sql", "per_page": 5})

    assert result["status_code"] == 500
    assert "boom" in result["error"]
