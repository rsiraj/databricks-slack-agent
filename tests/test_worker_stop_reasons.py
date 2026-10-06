"""Agent-loop exit paths: max_tokens, refusal, pause_turn, deadline, max_iterations,
and the run_summary / deadline plumbing in the mention handler and lambda_handler.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest

import worker
from utils import clients
from utils import deadline
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
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content
        self.model = "claude-sonnet-4-6"


class _FakeAnthropicClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = self
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs["messages"]))
        return self.responses.pop(0)


class _Harness:
    def __init__(self, monkeypatch, responses):
        self.client = _FakeAnthropicClient(responses)
        self.posts: list[str] = []
        self.chunks: list[str] = []
        self.saved: list[list[dict]] = []
        self.audits: list[tuple[str, dict]] = []
        self.feedback: list = []
        self.uploads: list = []

        monkeypatch.setitem(clients._clients, "anthropic_client", self.client)
        monkeypatch.setattr(worker, "_build_system", lambda: [])
        monkeypatch.setattr(worker, "_dispatch_tool", lambda name, inp: {"ok": True})
        monkeypatch.setattr(
            worker,
            "post_message",
            lambda channel, text, thread_ts=None, blocks=None: self.posts.append(text) or "ts",
        )
        monkeypatch.setattr(worker, "update_message", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            worker,
            "post_message_chunks",
            lambda channel, text, thread_ts=None: (
                self.chunks.append(text) or MessageDelivery((text,), ("answer-ts",), 1)
            ),
        )
        monkeypatch.setattr(
            worker,
            "_post_feedback_prompt",
            lambda *args, **kwargs: self.feedback.append(args),
        )
        monkeypatch.setattr(
            worker,
            "upload_sql_file",
            lambda *args, **kwargs: self.uploads.append(args) or "file-1",
        )
        monkeypatch.setattr(
            worker,
            "save_session",
            lambda thread_ts, messages: self.saved.append(deepcopy(messages)),
        )
        monkeypatch.setattr(
            worker,
            "audit",
            lambda event, **fields: self.audits.append((event, fields)),
        )

    def run(self, messages=None):
        messages = messages or [{"role": "user", "content": "Question"}]
        metrics = worker.RunMetrics()
        outcome = worker.run_agent(messages, "C1", "thread-1", metrics=metrics)
        return outcome, metrics, messages

    def events(self, name):
        return [fields for event, fields in self.audits if event == name]


def _tool_use(tool_id="t1"):
    return _FakeBlock("tool_use", id=tool_id, name="github_find_files", input={"name": "x"})


class TestMaxTokens:
    def test_text_is_delivered_with_truncation_note_and_saved(self, monkeypatch):
        harness = _Harness(
            monkeypatch,
            [
                _FakeResponse(
                    "max_tokens",
                    [
                        _FakeBlock("text", text="Partial ```sql\nSELECT 1\n``` answer"),
                        _tool_use(),  # cut-off tool call must not be persisted
                    ],
                )
            ],
        )

        outcome, metrics, messages = harness.run()

        assert outcome == "max_tokens"
        assert metrics.outcome == "max_tokens"
        assert len(harness.chunks) == 1
        assert harness.chunks[0].endswith("\n\n" + worker.MAX_TOKENS_TRUNCATION_NOTE)
        assert harness.chunks[0].startswith("Partial")
        assert len(harness.saved) == 1
        saved_assistant = harness.saved[0][-1]
        assert saved_assistant["role"] == "assistant"
        assert all(block["type"] != "tool_use" for block in saved_assistant["content"])
        assert saved_assistant["content"][0]["text"] == harness.chunks[0]
        # A truncated answer gets neither a feedback panel nor an SQL attachment.
        assert harness.feedback == []
        assert harness.uploads == []

    def test_no_text_posts_notice_and_does_not_save(self, monkeypatch):
        harness = _Harness(monkeypatch, [_FakeResponse("max_tokens", [_tool_use()])])

        outcome, _, _ = harness.run()

        assert outcome == "max_tokens"
        assert harness.posts == [worker.MAX_TOKENS_NO_TEXT_MESSAGE]
        assert harness.chunks == []
        assert harness.saved == []


class TestRefusal:
    def test_neutral_message_audit_and_no_save(self, monkeypatch):
        harness = _Harness(
            monkeypatch, [_FakeResponse("refusal", [_FakeBlock("text", text="")])]
        )

        outcome, _, _ = harness.run()

        assert outcome == "refusal"
        assert harness.posts == [worker.REFUSAL_MESSAGE]
        assert harness.saved == []
        assert harness.events("model_refusal") == [{"iteration": 0}]


def test_model_deadline_posts_notice_and_persists_balanced_turn(monkeypatch):
    harness = _Harness(monkeypatch, [])

    def exceeded(*args, **kwargs):
        raise deadline.DeadlineExceeded()

    monkeypatch.setattr(worker, "_call_model", exceeded)
    outcome, metrics, messages = harness.run()
    assert outcome == metrics.outcome == "deadline"
    assert worker.DEADLINE_MESSAGE in harness.posts
    assert harness.saved[-1][-1]["content"] == worker.DEADLINE_MESSAGE


class TestPauseTurn:
    def test_pause_turn_resends_assistant_turn_and_continues(self, monkeypatch):
        harness = _Harness(
            monkeypatch,
            [
                _FakeResponse("pause_turn", [_FakeBlock("text", text="still working")]),
                _FakeResponse("end_turn", [_FakeBlock("text", text="Final")]),
            ],
        )

        outcome, metrics, _ = harness.run()

        assert outcome == "answered"
        assert metrics.api_calls == 2
        second_request = harness.client.calls[1]
        assert second_request[-1]["role"] == "assistant"
        assert second_request[-1]["content"][0]["text"] == "still working"
        assert harness.chunks == ["Final"]

    def test_pause_turn_is_bounded_by_iteration_cap_without_saving(self, monkeypatch):
        monkeypatch.setattr(worker, "MAX_ITERATIONS", 2)
        harness = _Harness(
            monkeypatch,
            [
                _FakeResponse("pause_turn", [_FakeBlock("text", text="a")]),
                _FakeResponse("pause_turn", [_FakeBlock("text", text="b")]),
            ],
        )

        outcome, _, _ = harness.run()

        assert outcome == "max_iterations"
        assert harness.saved == []


class TestUnexpectedStop:
    def test_returns_unexpected_stop_outcome(self, monkeypatch):
        harness = _Harness(
            monkeypatch, [_FakeResponse("stop_sequence", [_FakeBlock("text", text="x")])]
        )

        outcome, _, _ = harness.run()

        assert outcome == "unexpected_stop"
        assert harness.posts == ["_Stopped unexpectedly: stop_sequence_"]
        assert harness.saved == []


class TestMaxIterations:
    def test_tool_loop_hits_cap_and_saves_balanced_history(self, monkeypatch):
        monkeypatch.setattr(worker, "MAX_ITERATIONS", 2)
        harness = _Harness(
            monkeypatch,
            [
                _FakeResponse("tool_use", [_tool_use("t1")]),
                _FakeResponse("tool_use", [_tool_use("t2")]),
            ],
        )

        outcome, metrics, messages = harness.run()

        assert outcome == "max_iterations"
        assert metrics.tool_calls == 2
        assert len(harness.saved) == 1
        assert messages[-1]["role"] == "user"


class TestDeadline:
    def test_loop_stops_before_next_call_when_time_is_short(self, monkeypatch):
        # Consulted by the loop check, then by bounded_timeout inside the model
        # call, then by the next loop check. Plenty of time first, then not.
        values = [1000.0, 1000.0, 30.0]
        calls = {"n": 0}

        def remaining_seconds():
            value = values[min(calls["n"], len(values) - 1)]
            calls["n"] += 1
            return value

        monkeypatch.setattr(worker.deadline, "remaining_seconds", remaining_seconds)
        harness = _Harness(
            monkeypatch,
            [
                _FakeResponse("tool_use", [_tool_use()]),
                _FakeResponse("end_turn", [_FakeBlock("text", text="never reached")]),
            ],
        )

        outcome, metrics, messages = harness.run()

        assert outcome == "deadline"
        assert metrics.api_calls == 1
        assert len(harness.client.calls) == 1
        assert worker.DEADLINE_MESSAGE in harness.posts
        assert harness.events("agent_deadline_reached") == [
            {"iteration": 1, "remaining_seconds": 30.0, "api_calls": 1}
        ]
        # Balanced history (tool_results present) is persisted with the notice.
        assert len(harness.saved) == 1
        assert harness.saved[0][-1] == {"role": "assistant", "content": worker.DEADLINE_MESSAGE}
        assert harness.saved[0][-2]["role"] == "user"

    def test_no_deadline_means_no_early_stop(self, monkeypatch):
        deadline.clear()
        harness = _Harness(
            monkeypatch, [_FakeResponse("end_turn", [_FakeBlock("text", text="ok")])]
        )

        outcome, _, _ = harness.run()

        assert outcome == "answered"
        assert harness.events("agent_deadline_reached") == []

    def test_threshold_is_configurable(self, monkeypatch):
        monkeypatch.setattr(worker, "AGENT_MIN_SECONDS_PER_ITERATION", 5.0)
        monkeypatch.setattr(worker.deadline, "remaining_seconds", lambda: 30.0)
        harness = _Harness(
            monkeypatch, [_FakeResponse("end_turn", [_FakeBlock("text", text="ok")])]
        )

        outcome, _, _ = harness.run()

        assert outcome == "answered"


@pytest.fixture
def mention_env(monkeypatch):
    """Pass every channel/rate-limit guard so _handle_app_mention reaches run_agent."""
    state = {"audits": [], "posts": [], "saved": []}
    monkeypatch.setattr(worker, "is_channel_allowed", lambda channel: True)
    monkeypatch.setattr(worker, "is_private_channel", lambda channel: True)
    monkeypatch.setattr(worker, "get_channel_members", lambda channel: frozenset({"UADMIN"}))
    monkeypatch.setattr(worker, "channel_has_admin", lambda members: True)
    monkeypatch.setattr(worker, "check_user", lambda user_id: (True, None))
    monkeypatch.setattr(worker, "check_channel", lambda channel: (True, None))
    monkeypatch.setattr(worker, "load_session", lambda thread_ts: None)
    monkeypatch.setattr(worker, "extract_image_blocks", lambda message: [])
    monkeypatch.setattr(
        worker, "audit", lambda event, **fields: state["audits"].append((event, fields))
    )
    monkeypatch.setattr(
        worker,
        "post_message",
        lambda channel, text, thread_ts=None, blocks=None: state["posts"].append(text) or "ts",
    )
    monkeypatch.setattr(
        worker, "save_session", lambda thread_ts, messages: state["saved"].append(deepcopy(messages))
    )
    return state


def _mention(text="<@BOT> hello"):
    return {"type": "app_mention", "channel": "C1", "user": "U1", "ts": "1.0", "text": text}


class TestRunSummary:
    def test_emitted_after_successful_run(self, monkeypatch, mention_env):
        def fake_run_agent(messages, channel, thread_ts, metrics=None):
            metrics.record_model_call(120, {"input_tokens": 10, "output_tokens": 3})
            metrics.outcome = "answered"
            return "answered"

        monkeypatch.setattr(worker, "run_agent", fake_run_agent)

        worker._handle_app_mention(_mention())

        summaries = [fields for event, fields in mention_env["audits"] if event == "run_summary"]
        assert len(summaries) == 1
        assert summaries[0]["outcome"] == "answered"
        assert summaries[0]["api_calls"] == 1
        assert summaries[0]["total_latency_ms"] == 120
        assert summaries[0]["total_input_tokens"] == 10
        assert summaries[0]["model"] == worker.MODEL

    def test_emitted_with_error_outcome_when_run_raises(self, monkeypatch, mention_env):
        def fake_run_agent(messages, channel, thread_ts, metrics=None):
            metrics.record_model_call(50, {"input_tokens": 7})
            raise RuntimeError("boom")

        monkeypatch.setattr(worker, "run_agent", fake_run_agent)

        result = worker._handle_app_mention(_mention())

        assert result == {"ok": True}
        events = [event for event, _ in mention_env["audits"]]
        assert events.index("agent_error") < events.index("run_summary")
        summary = dict(mention_env["audits"])["run_summary"]
        assert summary["outcome"] == "error"
        assert summary["api_calls"] == 1
        assert summary["total_input_tokens"] == 7


class TestLambdaDeadline:
    def test_deadline_is_set_from_context_and_cleared(self, monkeypatch):
        observed = {}

        def fake_dispatch(event):
            observed["remaining"] = deadline.remaining_seconds()
            return {"ok": True}

        monkeypatch.setattr(worker, "_dispatch_event", fake_dispatch)
        context = SimpleNamespace(get_remaining_time_in_millis=lambda: 600_000)

        assert worker.lambda_handler({}, context) == {"ok": True}

        expected = 600 - worker.LAMBDA_DEADLINE_SAFETY_MARGIN_SECONDS
        assert expected - 2 <= observed["remaining"] <= expected
        assert deadline.remaining_seconds() is None

    def test_context_without_remaining_time_runs_without_deadline(self, monkeypatch):
        observed = {}
        monkeypatch.setattr(
            worker,
            "_dispatch_event",
            lambda event: observed.setdefault("remaining", deadline.remaining_seconds()) or {"ok": True},
        )

        worker.lambda_handler({}, None)

        assert observed["remaining"] is None

    def test_deadline_cleared_even_when_dispatch_raises(self, monkeypatch):
        def fake_dispatch(event):
            raise RuntimeError("boom")

        monkeypatch.setattr(worker, "_dispatch_event", fake_dispatch)
        context = SimpleNamespace(get_remaining_time_in_millis=lambda: 600_000)

        with pytest.raises(RuntimeError):
            worker.lambda_handler({}, context)

        assert deadline.remaining_seconds() is None
