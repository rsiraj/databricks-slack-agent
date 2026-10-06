"""Model/effort/max_tokens configuration and usage/latency measurement — worker.py."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

import worker
from utils import clients, deadline
from utils.slack_helpers import MessageDelivery


class _FakeBlock:
    def __init__(self, block_type, **values):
        self.type = block_type
        for key, value in values.items():
            setattr(self, key, value)

    def model_dump(self):
        result = {"type": self.type}
        for key in ("id", "name", "input", "text"):
            if hasattr(self, key):
                result[key] = getattr(self, key)
        return result


class _FakeResponse:
    def __init__(self, stop_reason, content, usage=None, model="claude-sonnet-4-6-20260101"):
        self.stop_reason = stop_reason
        self.content = content
        self.model = model
        if usage is not None:
            self.usage = SimpleNamespace(**usage)


class _FakeAnthropicClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = self
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


@pytest.fixture
def quiet_slack(monkeypatch):
    monkeypatch.setattr(worker, "_build_system", lambda: [])
    monkeypatch.setattr(worker, "post_message", lambda *args, **kwargs: "ts")
    monkeypatch.setattr(worker, "update_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker,
        "post_message_chunks",
        lambda channel, text, thread_ts=None: MessageDelivery((text,), ("answer-ts",), 1),
    )
    monkeypatch.setattr(worker, "_post_feedback_prompt", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_attach_final_sql", lambda *args, **kwargs: True)
    monkeypatch.setattr(worker, "save_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_dispatch_tool", lambda name, inp: {"ok": True})


class TestEffortAndEnv:
    @pytest.mark.parametrize("raw", ["low", "medium", "high", "max", " HIGH ", "Max"])
    def test_valid_efforts_are_normalized(self, raw):
        assert worker._resolve_effort(raw) == raw.strip().lower()

    def test_invalid_effort_falls_back_with_warning(self, caplog):
        caplog.set_level("WARNING")

        assert worker._resolve_effort("turbo") == "high"
        assert "Invalid ANTHROPIC_EFFORT" in caplog.text

    def test_missing_effort_falls_back(self):
        assert worker._resolve_effort(None) == "high"

    def test_defaults_match_production(self):
        # Production (Terraform) runs claude-sonnet-5 at high with
        # ANTHROPIC_MAX_TOKENS=24000; the code defaults must agree so CI, forks
        # and local runs evaluate the same configuration.
        assert worker.MODEL == "claude-sonnet-5"
        assert worker.EFFORT == "high"
        assert worker.MAX_TOKENS == 24_000

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("20000", 20000), ("abc", 16000), ("0", 16000), ("-5", 16000), ("", 16000)],
    )
    def test_positive_int_env(self, monkeypatch, raw, expected):
        monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", raw)

        assert worker._positive_int_env("ANTHROPIC_MAX_TOKENS", 16000) == expected

    def test_positive_int_env_unset(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_MAX_TOKENS", raising=False)

        assert worker._positive_int_env("ANTHROPIC_MAX_TOKENS", 16000) == 16000


class TestMaxTokensResolution:
    """``_resolve_max_tokens``: env wins; Sonnet 4.6 keeps the 16,000 baseline;
    other models get ``recommended_max_tokens`` only when the env var is unset."""

    def test_env_var_wins_for_any_model(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "20000")

        assert worker._resolve_max_tokens("claude-sonnet-5", "max") == 20000
        assert worker._resolve_max_tokens("claude-sonnet-4-6", "high") == 20000

    def test_invalid_env_value_falls_back_to_baseline_not_recommendation(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "abc")

        assert worker._resolve_max_tokens("claude-sonnet-5", "max") == 16000

    @pytest.mark.parametrize("model", ["claude-sonnet-4-6", "claude-sonnet-4-6-20260115", "Claude-Sonnet-4-6"])
    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
    def test_baseline_model_stays_at_16000_when_unset(self, monkeypatch, model, effort):
        monkeypatch.delenv("ANTHROPIC_MAX_TOKENS", raising=False)

        assert worker._resolve_max_tokens(model, effort) == 16000

    @pytest.mark.parametrize(
        ("model", "effort", "expected"),
        [
            ("claude-sonnet-5", "low", 24000),
            ("claude-sonnet-5", "medium", 24000),
            ("claude-sonnet-5", "high", 24000),
            ("claude-sonnet-5", "xhigh", 48000),
            ("claude-sonnet-5", "max", 64000),
            ("claude-opus-5", "high", 24000),
            ("claude-opus-5-20260301", "xhigh", 48000),
            ("gpt-5.6-terra", "high", 16000),
        ],
    )
    def test_other_models_use_recommendation_when_unset(self, monkeypatch, model, effort, expected):
        monkeypatch.delenv("ANTHROPIC_MAX_TOKENS", raising=False)

        assert worker._resolve_max_tokens(model, effort) == expected

    def test_choice_is_logged_with_its_source(self, monkeypatch, caplog):
        monkeypatch.delenv("ANTHROPIC_MAX_TOKENS", raising=False)
        caplog.set_level("INFO", logger="worker")

        worker._resolve_max_tokens("claude-sonnet-5", "high")

        assert "max_tokens=24000" in caplog.text
        assert "source=recommended_max_tokens" in caplog.text

        caplog.clear()
        worker._resolve_max_tokens("claude-sonnet-4-6", "high")
        assert "max_tokens=16000" in caplog.text
        assert "source=baseline" in caplog.text

    def test_module_constant_is_the_hard_floor_not_the_resolved_cap(self):
        assert worker.DEFAULT_MAX_TOKENS == 16000
        assert worker.SONNET_4_6_BASELINE_MODEL == "claude-sonnet-4-6"
        # The default model (claude-sonnet-5 / high) resolves through
        # recommended_max_tokens, matching production's ANTHROPIC_MAX_TOKENS=24000.
        assert worker.MAX_TOKENS == 24000
        assert worker.MAX_TOKENS == worker._resolve_max_tokens(worker.MODEL, worker.EFFORT)


class TestUsageFields:
    def test_reads_all_fields_and_defaults_missing_or_none(self):
        resp = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=1200,
                cache_read_input_tokens=None,
                output_tokens=40,
            )
        )

        assert worker._usage_fields(resp) == {
            "input_tokens": 1200,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 40,
        }

    def test_missing_usage_object(self):
        assert worker._usage_fields(SimpleNamespace()) == {
            "input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 0,
        }

    def test_non_numeric_values_default_to_zero(self):
        resp = SimpleNamespace(usage=SimpleNamespace(input_tokens="n/a"))

        assert worker._usage_fields(resp)["input_tokens"] == 0


class TestRequestTimeout:
    def test_uses_sdk_timeout_type_with_short_connect(self):
        timeout = clients.anthropic_request_timeout(120)

        assert timeout.connect == 10
        assert timeout.read == 120
        assert timeout.write == 120
        assert timeout.pool == 120

    def test_connect_never_exceeds_total(self):
        assert clients.anthropic_request_timeout(5).connect == 5

    def test_default_timeout_env(self):
        assert clients.ANTHROPIC_TIMEOUT_SECONDS == 180.0


def test_model_call_uses_first_class_output_config_and_timeout(monkeypatch, quiet_slack):
    client = _FakeAnthropicClient(
        [
            _FakeResponse(
                "end_turn",
                [_FakeBlock("text", text="Done")],
                usage={"input_tokens": 10, "output_tokens": 2},
            )
        ]
    )
    monkeypatch.setitem(clients._clients, "anthropic_client", client)
    monkeypatch.setattr(worker, "audit", lambda *args, **kwargs: None)

    outcome = worker.run_agent([{"role": "user", "content": "hi"}], "C1", "t1")

    assert outcome == "answered"
    kwargs = client.calls[0]
    assert kwargs["model"] == worker.MODEL
    assert kwargs["max_tokens"] == worker.MAX_TOKENS
    assert kwargs["output_config"] == {"effort": worker.EFFORT}
    assert "extra_body" not in kwargs
    assert kwargs["timeout"].read == clients.ANTHROPIC_TIMEOUT_SECONDS


def test_request_timeout_is_bounded_by_lambda_deadline(monkeypatch, quiet_slack):
    client = _FakeAnthropicClient(
        [_FakeResponse("end_turn", [_FakeBlock("text", text="Done")])]
    )
    monkeypatch.setitem(clients._clients, "anthropic_client", client)
    monkeypatch.setattr(worker, "audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "AGENT_MIN_SECONDS_PER_ITERATION", 0)
    deadline.set_deadline_from_remaining(60)
    try:
        worker.run_agent([{"role": "user", "content": "hi"}], "C1", "t1")
    finally:
        deadline.clear()

    read_timeout = client.calls[0]["timeout"].read
    assert worker.ANTHROPIC_TIMEOUT_FLOOR_SECONDS <= read_timeout <= 60


def test_model_call_audit_and_run_metrics_accumulate(monkeypatch, quiet_slack):
    client = _FakeAnthropicClient(
        [
            _FakeResponse(
                "tool_use",
                [_FakeBlock("tool_use", id="t1", name="github_find_files", input={"name": "x"})],
                usage={
                    "input_tokens": 100,
                    "cache_creation_input_tokens": 5000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 30,
                },
            ),
            _FakeResponse(
                "end_turn",
                [_FakeBlock("text", text="Answer")],
                usage={
                    "input_tokens": 150,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 5000,
                    "output_tokens": 70,
                },
            ),
        ]
    )
    audits = []
    monkeypatch.setitem(clients._clients, "anthropic_client", client)
    monkeypatch.setattr(worker, "audit", lambda event, **fields: audits.append((event, fields)))

    metrics = worker.RunMetrics()
    outcome = worker.run_agent(
        [{"role": "user", "content": "Question"}], "C1", "t1", metrics=metrics
    )

    model_calls = [fields for event, fields in audits if event == "model_call"]
    assert len(model_calls) == 2
    first = model_calls[0]
    assert first["requested_model"] == worker.MODEL
    assert first["model"] == "claude-sonnet-4-6-20260101"
    assert first["effort"] == worker.EFFORT
    assert first["iteration"] == 0
    assert first["stop_reason"] == "tool_use"
    assert first["max_tokens"] == worker.MAX_TOKENS
    assert isinstance(first["latency_ms"], int) and first["latency_ms"] >= 0
    assert first["input_tokens"] == 100
    assert first["cache_creation_input_tokens"] == 5000
    assert model_calls[1]["iteration"] == 1
    assert model_calls[1]["cache_read_input_tokens"] == 5000

    assert outcome == "answered"
    summary = metrics.summary()
    assert summary["outcome"] == "answered"
    assert summary["api_calls"] == 2
    assert summary["tool_calls"] == 1
    assert summary["total_input_tokens"] == 250
    assert summary["total_cache_creation_input_tokens"] == 5000
    assert summary["total_cache_read_input_tokens"] == 5000
    assert summary["total_output_tokens"] == 100
    assert summary["total_latency_ms"] >= 0
    assert summary["model"] == worker.MODEL
    assert summary["effort"] == worker.EFFORT
    # run_summary itself is emitted by the mention handler, not run_agent,
    # so existing delivery tests that inspect the last audit event still hold.
    assert not any(event == "run_summary" for event, _ in audits)


def test_run_metrics_default_outcome_is_error_until_loop_returns():
    metrics = worker.RunMetrics()

    assert metrics.outcome == "error"
    assert set(metrics.summary()) == {
        "api_calls",
        "tool_calls",
        "total_latency_ms",
        "total_input_tokens",
        "total_cache_creation_input_tokens",
        "total_cache_read_input_tokens",
        "total_output_tokens",
        "outcome",
        "model",
        "effort",
        "provider",
    }
