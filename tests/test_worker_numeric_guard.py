"""Worker wiring for the numeric grounding guard and the ``calculate`` tool."""

from copy import deepcopy

import pytest

import worker
from utils import clients
from tools import calc_tool
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
        self.calls: list[list[dict]] = []

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs["messages"]))
        return self.responses.pop(0)


def _text(text):
    return _FakeResponse("end_turn", [_FakeBlock("text", text=text)])


def _tool_use(name, tool_input, tool_id="t1"):
    return _FakeResponse(
        "tool_use", [_FakeBlock("tool_use", id=tool_id, name=name, input=tool_input)]
    )


LIST_RUNS_RESULT = {
    "date": "2026-08-12",
    "timezone": "America/New_York",
    "runs": [
        {
            "job_id": 101,
            "run_name": "Ops Email",
            "result_state": "SUCCESS",
            "life_cycle_state": "TERMINATED",
            "start_time_ms": 1755000000000,
            "duration_ms": 3960000,
        }
    ],
}

DESCRIBE_RESULT = {
    "object": {"catalog": "acme_prod", "schema": "reporting", "name": "signups"},
    "column_count": 45,
    "columns": [{"name": "signup_id", "type": "string"}],
}


def _dispatch(name, inp):
    """Real calculator, canned everything else (no network)."""
    if name == "calculate":
        return calc_tool.dispatch(name, inp)
    if name == "databricks_list_job_runs":
        return deepcopy(LIST_RUNS_RESULT)
    if name == "databricks_describe_object":
        return deepcopy(DESCRIBE_RESULT)
    return {"ok": True}


class _Harness:
    def __init__(self, monkeypatch, responses):
        self.client = _FakeAnthropicClient(responses)
        self.posts: list[str] = []
        self.chunks: list[str] = []
        self.saved: list[list[dict]] = []
        self.audits: list[tuple[str, dict]] = []
        self.status_labels: list[str] = []

        monkeypatch.setitem(clients._clients, "anthropic_client", self.client)
        monkeypatch.setattr(worker, "_build_system", lambda: [])
        monkeypatch.setattr(worker, "_dispatch_tool", _dispatch)
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
        monkeypatch.setattr(worker, "_post_feedback_prompt", lambda *args, **kwargs: None)
        monkeypatch.setattr(worker, "upload_sql_file", lambda *args, **kwargs: "file-1")
        monkeypatch.setattr(
            worker,
            "save_session",
            lambda thread_ts, messages: self.saved.append(deepcopy(messages)),
        )
        monkeypatch.setattr(
            worker, "audit", lambda event, **fields: self.audits.append((event, fields))
        )
        original_label = worker._tool_status_label

        def recording_label(tool_name, tool_input):
            label = original_label(tool_name, tool_input)
            self.status_labels.append(label)
            return label

        monkeypatch.setattr(worker, "_tool_status_label", recording_label)

    def run(self, question):
        messages = [{"role": "user", "content": question}]
        metrics = worker.RunMetrics()
        outcome = worker.run_agent(messages, "C1", "thread-1", metrics=metrics)
        return outcome, metrics, messages

    def events(self, name):
        return [fields for event, fields in self.audits if event == name]


def _shape(messages):
    shape = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            shape.append((message["role"], "text"))
        else:
            shape.append((message["role"], ",".join(block["type"] for block in content)))
    return shape


def _mentions_correction(messages):
    return any("internal correction" in str(message) for message in messages)


@pytest.fixture(autouse=True)
def _default_mode(monkeypatch):
    monkeypatch.delenv("NUMERIC_GUARD_MODE", raising=False)
    monkeypatch.delenv("NUMERIC_GUARD_MAX_REVISIONS", raising=False)


class TestToolWiring:
    def test_calculate_is_registered_and_dispatched(self):
        assert any(tool["name"] == "calculate" for tool in worker.ALL_TOOLS)
        result = worker._dispatch_tool(
            "calculate", {"expression": "(new - old) / old * 100", "variables": {"old": 1000, "new": 1234}}
        )
        assert result["result"] == 23.4
        assert result["result_display"] == "23.4"

    def test_calculate_status_label_truncates_long_expressions(self):
        short = worker._tool_status_label("calculate", {"expression": "a  +\n b"})
        assert short == "Calculate `a + b`"
        long_expression = "x" * 100
        label = worker._tool_status_label("calculate", {"expression": long_expression})
        assert label.startswith("Calculate `")
        inner = label[len("Calculate `") : -1]
        assert len(inner) == worker.CALC_STATUS_EXPRESSION_CHARS
        assert inner.endswith("…")

    def test_calculate_validation_failure_is_a_guard_precondition(self):
        result = worker._dispatch_tool("calculate", {"expression": "1/0"})
        assert result["validation_failure"] == "division_by_zero"
        assert worker._tool_outcome("calculate", result) == "guard_precondition"

    def test_sql_guard_ignores_the_calculator(self):
        guard = worker.SqlRunGuard()
        assert guard.before_call("calculate", {"expression": "1+1"}) is None
        result = {"result": 2}
        assert guard.after_call("calculate", {"expression": "1+1"}, result) is result


class TestRevisionFlow:
    def test_revision_then_calculate_then_grounded_answer(self, monkeypatch):
        harness = _Harness(
            monkeypatch,
            [
                _text("Revenue grew from 1000 to 1234, a 23.4% increase."),
                _tool_use(
                    "calculate",
                    {"expression": "pct_change(old, new)", "variables": {"old": 1000, "new": 1234}},
                    tool_id="calc-1",
                ),
                _text("Revenue grew from 1000 to 1234, a 23.4% increase (via `calculate`)."),
            ],
        )

        outcome, metrics, messages = harness.run(
            "Revenue was 1000 last month and 1234 this month. Percent change?"
        )

        assert outcome == "answered"
        assert metrics.api_calls == 3
        assert metrics.tool_calls == 1
        # The private correction request reached the model exactly once ...
        assert len(harness.client.calls) == 3
        assert "internal correction" in str(harness.client.calls[1][-1]).lower()
        # The provider promotes the string instruction to a text block so it can
        # carry the moving cache breakpoint; the text itself is unchanged.
        (instruction_block,) = harness.client.calls[1][-1]["content"]
        assert '"23.4%"' in instruction_block["text"]
        # ... the model answered it with a tool call whose result grounded the figure ...
        assert harness.status_labels == ["Calculate `pct_change(old, new)`"]
        assert harness.chunks == [
            "Revenue grew from 1000 to 1234, a 23.4% increase (via `calculate`)."
        ]
        # ... and the saved history keeps the tool exchange but not the draft/request.
        assert _shape(messages) == [
            ("user", "text"),
            ("assistant", "tool_use"),
            ("user", "tool_result"),
            ("assistant", "text"),
        ]
        assert not _mentions_correction(messages)
        assert harness.saved[-1] == messages
        assert harness.events("numeric_guard_revision") == [
            {"figures": ["23.4%"], "count": 1}
        ]
        assert harness.events("numeric_guard_unverified") == []

    def test_grounded_draft_needs_no_revision(self, monkeypatch):
        harness = _Harness(
            monkeypatch,
            [
                _tool_use("databricks_list_job_runs", {"date": "2026-08-12"}),
                _text("Ops Email ran once on 2026-08-12 with a duration of 3,960,000 ms."),
            ],
        )

        outcome, metrics, messages = harness.run("What ran on 2026-08-12?")

        assert outcome == "answered"
        assert metrics.api_calls == 2
        assert harness.events("numeric_guard_revision") == []
        assert harness.events("numeric_guard_unverified") == []
        assert _shape(messages) == [
            ("user", "text"),
            ("assistant", "tool_use"),
            ("user", "tool_result"),
            ("assistant", "text"),
        ]

    def test_budget_exhausted_footnotes_remaining_figures(self, monkeypatch):
        harness = _Harness(
            monkeypatch,
            [
                _text("There were 4,567 signups."),
                _text("There were 4,567 signups, I believe."),
            ],
        )

        outcome, metrics, messages = harness.run("How many signups?")

        assert outcome == "answered"
        assert metrics.api_calls == 2
        assert harness.chunks == [
            "There were 4,567 signups, I believe.\n\n"
            "_Figures not verified against a tool result: 4,567._"
        ]
        assert _shape(messages) == [("user", "text"), ("assistant", "text")]
        assert not _mentions_correction(messages)
        assert messages[-1]["content"][0]["text"] == harness.chunks[0]
        assert harness.events("numeric_guard_revision") == [{"figures": ["4,567"], "count": 1}]
        assert harness.events("numeric_guard_unverified") == [{"figures": ["4,567"], "count": 1}]

    def test_max_revisions_env_raises_the_budget(self, monkeypatch):
        monkeypatch.setenv("NUMERIC_GUARD_MAX_REVISIONS", "2")
        harness = _Harness(
            monkeypatch,
            [
                _text("About 4,567."),
                _text("Still 4,567."),
                _text("No figure available from a tool this turn."),
            ],
        )

        outcome, metrics, messages = harness.run("How many signups?")

        assert outcome == "answered"
        assert metrics.api_calls == 3
        assert harness.chunks == ["No figure available from a tool this turn."]
        assert _shape(messages) == [("user", "text"), ("assistant", "text")]
        assert not _mentions_correction(messages)
        assert len(harness.events("numeric_guard_revision")) == 2
        assert harness.events("numeric_guard_unverified") == []

    def test_annotate_mode_skips_revision(self, monkeypatch):
        monkeypatch.setenv("NUMERIC_GUARD_MODE", "annotate")
        harness = _Harness(monkeypatch, [_text("There were 4,567 signups.")])

        outcome, metrics, messages = harness.run("How many signups?")

        assert outcome == "answered"
        assert metrics.api_calls == 1
        assert harness.chunks[0].endswith(
            "_Figures not verified against a tool result: 4,567._"
        )
        assert harness.events("numeric_guard_revision") == []
        assert harness.events("numeric_guard_unverified") == [{"figures": ["4,567"], "count": 1}]

    def test_off_mode_disables_the_guard(self, monkeypatch):
        monkeypatch.setenv("NUMERIC_GUARD_MODE", "off")
        harness = _Harness(monkeypatch, [_text("There were 4,567 signups.")])

        outcome, metrics, _ = harness.run("How many signups?")

        assert outcome == "answered"
        assert metrics.api_calls == 1
        assert harness.chunks == ["There were 4,567 signups."]
        assert harness.events("numeric_guard_revision") == []
        assert harness.events("numeric_guard_unverified") == []

    def test_prior_turn_tool_results_ground_follow_up_answers(self, monkeypatch):
        harness = _Harness(monkeypatch, [_text("As before: 8,765 rows.")])
        prior = [
            {"role": "user", "content": "Count rows."},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t0", "name": "databricks_execute_sql", "input": {"sql": "SELECT COUNT(*) FROM t"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t0", "content": '{"row_count": 1, "sample_rows": [[8765]]}'}
                ],
            },
            {"role": "assistant", "content": "8,765 rows (from `COUNT(*)`)."},
        ]
        messages = prior + [{"role": "user", "content": "Remind me of the count?"}]

        outcome = worker.run_agent(messages, "C1", "thread-1")

        assert outcome == "answered"
        assert harness.chunks == ["As before: 8,765 rows."]
        assert harness.events("numeric_guard_revision") == []


class TestInteractionWithEvidenceGuard:
    def test_evidence_revision_then_numeric_revision_with_tool_call(self, monkeypatch):
        harness = _Harness(
            monkeypatch,
            [
                _tool_use("databricks_list_job_runs", {"date": "2026-08-12"}, tool_id="runs-1"),
                _text("Ops Email ran for 66 minutes, which looks slow."),
                _text("Ops Email ran for 66 minutes."),
                _tool_use(
                    "calculate",
                    {"expression": "duration_ms / 60000", "variables": {"duration_ms": 3960000}},
                    tool_id="calc-1",
                ),
                _text("Ops Email ran for 66 minutes (via `calculate`)."),
            ],
        )

        outcome, metrics, messages = harness.run("Compare today's email jobs.")

        assert outcome == "answered"
        assert metrics.api_calls == 5
        assert harness.chunks == ["Ops Email ran for 66 minutes (via `calculate`)."]
        assert _shape(messages) == [
            ("user", "text"),
            ("assistant", "tool_use"),
            ("user", "tool_result"),
            ("assistant", "tool_use"),
            ("user", "tool_result"),
            ("assistant", "text"),
        ]
        assert messages[1]["content"][0]["name"] == "databricks_list_job_runs"
        assert messages[3]["content"][0]["name"] == "calculate"
        assert not _mentions_correction(messages)
        assert all("looks slow" not in str(message) for message in messages)
        assert harness.saved[-1] == messages
        assert len(harness.events("final_response_guard_revision")) == 1
        assert harness.events("numeric_guard_revision") == [{"figures": ["66"], "count": 1}]
        assert harness.events("final_response_guard_fallback") == []
        assert harness.events("numeric_guard_unverified") == []

    def test_numeric_revision_then_evidence_revision(self, monkeypatch):
        harness = _Harness(
            monkeypatch,
            [
                _text("The table has about 45 columns."),
                _tool_use(
                    "databricks_describe_object",
                    {"catalog": "acme_prod", "schema": "reporting", "name": "signups"},
                    tool_id="desc-1",
                ),
                _text("The table has 45 columns and the Signups job runs daily."),
                _text("The table has 45 columns (`column_count` from describe)."),
            ],
        )

        outcome, metrics, messages = harness.run("Explain the Signups table.")

        assert outcome == "answered"
        assert metrics.api_calls == 4
        assert harness.chunks == ["The table has 45 columns (`column_count` from describe)."]
        assert _shape(messages) == [
            ("user", "text"),
            ("assistant", "tool_use"),
            ("user", "tool_result"),
            ("assistant", "text"),
        ]
        assert not _mentions_correction(messages)
        assert all("runs daily" not in str(message) for message in messages)
        assert all("about 45" not in str(message) for message in messages)
        assert harness.events("numeric_guard_revision") == [{"figures": ["45"], "count": 1}]
        assert len(harness.events("final_response_guard_revision")) == 1

    def test_evidence_fallback_is_exempt_from_numeric_check(self, monkeypatch):
        harness = _Harness(
            monkeypatch,
            [
                _tool_use("databricks_list_job_runs", {"date": "2026-08-12"}, tool_id="runs-1"),
                _text("Ops Email looks normal because it aggregates many campaigns."),
                _text("Ops Email is slow because it aggregates many campaigns."),
            ],
        )

        outcome, _, messages = harness.run("Compare today's email jobs.")

        assert outcome == "answered"
        assert "*Observed job runs — 2026-08-12*" in harness.chunks[0]
        assert "Figures not verified" not in harness.chunks[0]
        assert harness.events("final_response_guard_fallback")
        assert harness.events("numeric_guard_revision") == []
        assert harness.events("numeric_guard_unverified") == []
        assert _shape(messages) == [
            ("user", "text"),
            ("assistant", "tool_use"),
            ("user", "tool_result"),
            ("assistant", "text"),
        ]


class TestBounds:
    def test_revision_counts_against_max_iterations(self, monkeypatch):
        monkeypatch.setattr(worker, "MAX_ITERATIONS", 1)
        harness = _Harness(monkeypatch, [_text("There were 4,567 signups.")])

        outcome, metrics, _ = harness.run("How many signups?")

        # The single allowed iteration was spent on the draft; the revision
        # request was appended but no further model call was made.
        assert outcome == "max_iterations"
        assert metrics.api_calls == 1
        assert harness.events("numeric_guard_revision") == [{"figures": ["4,567"], "count": 1}]
