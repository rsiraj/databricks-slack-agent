"""Evidence requirements for interpretations of Databricks job-run history."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import yaml


_RUN_HISTORY_TOOLS = {
    "databricks_list_job_runs",
    "databricks_get_recent_runs",
    "databricks_get_run_status",
}
_SCHEDULE_WORD = re.compile(r"\b(?:cadence|scheduled?|schedules?)\b", re.IGNORECASE)
# Concrete cadence / time tokens. A bare schedule word only counts as a cadence
# claim when one of these appears in the same sentence.
_CADENCE_TOKEN = re.compile(
    r"\b(?:hourly|daily|nightly|weekly|monthly|intraday|cron)\b"
    r"|\bevery\s+~?(?:\d+|one|two|three|four|few|other)\s+(?:minutes?|hours?|days?|weeks?)\b"
    r"|\bat\s+\d{1,2}:\d{2}\b"
    r"|\bat\s+\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b"
    r"|\bruns?\s+on\b"
    r"|\b(?:is|are|was|were|gets?|been)\s+scheduled\s+(?:to|for|at)\b",
    re.IGNORECASE,
)
# A cadence word: an adverb/adjective ("daily", "nightly") or the two-word
# forms the eval fixtures also forbid ("each day", "every night").
_CADENCE_WORD = (
    r"(?:hourly|daily|nightly|weekly|intraday|"
    r"(?:each|every)[\s-]+(?:day|night|morning|hour|week))"
)
# Cadence assertions that do not use a schedule word at all ("runs daily",
# "runs each day", "runs a daily MERGE", "performs a nightly load", "every two
# hours", "the daily refresh", "a healthy daily refresh"). The run verb may be
# separated from the cadence token by an article or determiner, and the noun
# the cadence modifies may have been an inline code span that
# ``_prose_without_code`` already stripped.
_CADENCE_CLAIM = re.compile(
    r"\b(?:runs?|ran|running|refresh(?:es|ed|ing)?|executes?|executed|executing|"
    r"performs?|performed|performing|triggers?|triggered|fires?|fired)\s+"
    r"(?:(?:a|an|the|its|this|that|one|as\s+a|on\s+a|in\s+a)\s+)?"
    + _CADENCE_WORD
    + r"\b"
    r"|\b" + _CADENCE_WORD + r".{0,30}"
    r"(?:schedule|cadence|job|run|refresh(?:es|ed|ing)?|merge|load|pipeline|batch|snapshot)\b"
    r"|\bevery\s+~?(?:\d+|one|two|three|four)\s+(?:minutes?|hours?|days?)\b",
    re.IGNORECASE,
)
# "there is no schedule configured", "not scheduled", "isn't scheduled",
# "has no cron schedule", "unscheduled" — a negated statement is not a cadence claim.
_CADENCE_NEGATION = re.compile(
    r"\b(?:no|not|never|without|isn['’]t|aren['’]t|wasn['’]t|weren['’]t|"
    r"is\s+not|are\s+not|was\s+not|were\s+not|"
    r"doesn['’]t\s+have|does\s+not\s+have|has\s+no|have\s+no|hasn['’]t)\s+"
    r"(?:\w+\s+){0,2}?(?:schedule[sd]?|cadence)\b"
    r"|\bunscheduled\b",
    re.IGNORECASE,
)
# Instructions and hypotheticals describe what *could* be configured, not what is.
# Bare "should" is only a recommendation when it governs scheduling/running the
# job ("should probably be scheduled nightly"); "days_since_latest should be
# <= 2 for a healthy daily refresh" states an expectation about the job's
# actual cadence and stays a claim.
_CADENCE_HYPOTHETICAL = re.compile(
    r"\b(?:to\s+add|to\s+set(?:\s+up)?|to\s+schedule|to\s+configure|to\s+enable|"
    r"you\s+can|you\s+could|you\s+may|you\s+would|you\s+should|you['’]d|"
    r"if\s+you|how\s+to|would|could|"
    r"should\s+(?:(?:probably|ideally|also|then)\s+)?(?:be\s+)?"
    r"(?:scheduled|configured|set|run|running|refresh(?:ed)?|executed?|triggered?))\b",
    re.IGNORECASE,
)
_BASELINE_CLAIM = re.compile(
    r"\b(?:normal(?:ly)?|typical(?:ly)?|usual(?:ly)?|unusual(?:ly)?|"
    r"abnormal(?:ly)?|anomal(?:y|ies|ous|ously)|consistent(?:ly)?|elevated|"
    r"heavyweight|slow|fast)\b",
    re.IGNORECASE,
)
_CAUSE_CONNECTOR = re.compile(
    r"\b(?:because|due to|could be|may be|might be|likely|probably|"
    r"suggest(?:s|ed|ing)?|indicat(?:e|es|ed|ing)|given)\b",
    re.IGNORECASE,
)
_STRUCTURAL_TOPIC = re.compile(
    r"\b(?:full scan|overwrite|joins?|campaigns?|aggregat(?:e|es|ed|ion))\b"
    r"|count\s*(?:\(\s*)?distinct\b",
    re.IGNORECASE,
)
_RUNTIME_CAUSE_TOPIC = re.compile(
    r"\b(?:data volume|volume|cache|compute|contention|query plan|workload|"
    r"upstream|incremental|missing data|cluster|resources?)\b",
    re.IGNORECASE,
)
_RUNTIME_TREND_CLAIM = re.compile(
    r"\bruntime\b.{0,80}\b(?:creep|increase|grow|degrade|trend)\w*\b",
    re.IGNORECASE,
)
_TIME_CONCEPT = re.compile(
    r"(?:^|[^a-z0-9])(?:date|day|week|month|timestamp|time)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)
_COMPARISON_SQL = re.compile(
    r"\b(?:group\s+by|date_trunc|count_if|case\s+when|lag\s*\(|"
    r"lead\s*\(|over\s*\(|union\s+all)\b",
    re.IGNORECASE,
)


def _result_failed(result: Any) -> bool:
    if not isinstance(result, dict):
        return True
    if result.get("error") or result.get("validation_failure"):
        return True
    return str(result.get("status", "")).upper() in {"FAILED", "CANCELED", "CLOSED"}


_CODE_COMMENT_LINE = re.compile(r"^\s*(?:--|#|//)\s*(.*\S)\s*$")


def _fence_comments(match: re.Match[str]) -> str:
    """The prose a reader still sees inside a fenced code block: its comment lines.

    A SQL comment such as ``-- days_since_latest should be <= 2 for a healthy
    daily refresh`` (an earlier eval run) is a claim about the job, not code;
    the eval fixtures' forbidden patterns match the final text verbatim, so the
    guard must read those lines too. Code itself is still dropped.
    """
    lines = match.group(0).split("\n")[1:-1]
    comments = [
        found.group(1) for found in (_CODE_COMMENT_LINE.match(line) for line in lines) if found
    ]
    return "\n" + "\n".join(comments) + "\n" if comments else " "


def _prose_without_code(text: str) -> str:
    without_fences = re.sub(r"```.*?```", _fence_comments, text, flags=re.DOTALL)
    return re.sub(r"`[^`\n]*`", " ", without_fences)


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?;])\s+|\n+", text)


def _contains_causal_claim(text: str, topic_pattern: re.Pattern[str]) -> bool:
    return any(
        _CAUSE_CONNECTOR.search(segment) and topic_pattern.search(segment)
        for segment in _sentences(text)
    )


def _is_cadence_assertion(sentence: str) -> bool:
    """True when one sentence positively asserts this job's schedule or cadence.

    A schedule word alone ("the schedule lives in the job YAML") is not a
    claim; it needs a concrete cadence/time token in the same sentence.
    Negated statements ("there is no schedule configured") and instructional or
    hypothetical text ("to add a schedule...", "if you want it scheduled
    daily...") are not claims about the job's actual configuration.
    """
    asserted = _CADENCE_CLAIM.search(sentence) or (
        _SCHEDULE_WORD.search(sentence) and _CADENCE_TOKEN.search(sentence)
    )
    if not asserted:
        return False
    if _CADENCE_NEGATION.search(sentence):
        return False
    if _CADENCE_HYPOTHETICAL.search(sentence):
        return False
    return True


def _contains_cadence_claim(text: str) -> bool:
    return any(_is_cadence_assertion(sentence) for sentence in _sentences(text))


_REVISION_REQUESTS = {
    "schedule_configuration_required": (
        "It stated or implied a job schedule/cadence without inspecting current schedule "
        "configuration in this turn. Remove that schedule/cadence claim; keep the rest of "
        "the supported answer."
    ),
    "recent_run_baseline_required": (
        "It characterised a run's duration as normal, typical, unusual, slow, fast, or "
        "consistent without comparing it against that job's recent run history in this "
        "turn. Remove that baseline judgement and report only the observed durations."
    ),
    "structural_logic_required": (
        "It attributed a run's behaviour to the job's logic (a full scan, overwrite, join, "
        "aggregation, or campaign processing) without reading the job's code in this turn. "
        "Remove that structural explanation."
    ),
    "runtime_comparison_required": (
        "It proposed a date-specific runtime cause or trend (data volume, cluster or "
        "compute, upstream data, caching, workload) without comparative time-based data "
        "in this turn. Remove that causal claim."
    ),
}


def _human_duration(duration_ms: int) -> str:
    total_seconds = max(0, round(duration_ms / 1000))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _run_duration_ms(run: dict[str, Any]) -> int | None:
    duration = run.get("duration_ms")
    if duration is not None:
        return int(duration)
    start = run.get("start_time_ms")
    end = run.get("end_time_ms")
    if start and end and int(end) >= int(start):
        return int(end) - int(start)
    return None


def _observed_start_time(run: dict[str, Any], timezone_name: str) -> str | None:
    raw_iso = run.get("start_time")
    if raw_iso:
        try:
            observed = datetime.fromisoformat(str(raw_iso))
        except ValueError:
            observed = None
    else:
        start_ms = run.get("start_time_ms")
        observed = (
            datetime.fromtimestamp(int(start_ms) / 1000, ZoneInfo(timezone_name))
            if start_ms
            else None
        )
    if observed is None:
        return None
    hour = observed.strftime("%I").lstrip("0") or "0"
    return f"{hour}:{observed.strftime('%M %p')}"


class JobRunEvidenceGuard:
    """Track current-turn evidence and reject unsupported run interpretations."""

    def __init__(self) -> None:
        self.run_history_used = False
        self.evidence: dict[str, set[str]] = {
            "schedule_configuration_required": set(),
            "recent_run_baseline_required": set(),
            "structural_logic_required": set(),
            "runtime_comparison_required": set(),
        }
        self.latest_list_runs: dict[str, Any] | None = None
        self.latest_recent_runs: dict[str, Any] | None = None
        self.job_names_by_id: dict[str, str] = {}
        self.jobs: set[str] = set()
        self.notebook_jobs: dict[str, set[str]] = {}
        self.file_contents: dict[str, str] = {}
        self.comparison_queries: list[str] = []

    def _job(self, job_id: Any = None, name: Any = None) -> str | None:
        name = str(name or "").strip()
        if job_id is None and not name:
            return None
        if job_id is None:
            matches = {key for key, value in self.job_names_by_id.items()
                       if value.casefold() == name.casefold()}
            if len(matches) == 1:
                return next(iter(matches))
            key = "name:" + name.casefold()
        else:
            key = str(job_id)
        self.jobs.add(key)
        if name:
            self.job_names_by_id[key] = name
            # Configuration can be read before the numeric job ID is resolved.
            named_key = "name:" + name.casefold()
            if named_key != key and named_key in self.jobs:
                self.jobs.discard(named_key)
                self.job_names_by_id.pop(named_key, None)
                for owners in [*self.evidence.values(), *self.notebook_jobs.values()]:
                    if named_key in owners:
                        owners.discard(named_key)
                        owners.add(key)
        return key

    @staticmethod
    def _path(path: str) -> str:
        # DAB notebook_task paths omit the source-file extension.
        return re.sub(r"\.(?:py|sql|scala|r|ipynb)$", "", path.strip().strip("\"'").lower()).removeprefix("./")

    def _link_notebooks(self, job: str, settings: dict[str, Any]) -> None:
        # YAML folds wrapped scalar paths into their real value. Walk actual
        # task objects (including for_each tasks), not lines or arbitrary keys.
        pending = [settings]
        seen: set[int] = set()
        while pending:
            task = pending.pop()
            if not isinstance(task, dict) or id(task) in seen:
                continue
            seen.add(id(task))
            notebook = task.get("notebook_task")
            if isinstance(notebook, dict):
                path = notebook.get("notebook_path")
                if isinstance(path, str) and path.strip():
                    self.notebook_jobs.setdefault(self._path(path), set()).add(job)
            tasks = task.get("tasks")
            if isinstance(tasks, list):
                pending.extend(tasks)
            loop = task.get("for_each_task")
            if isinstance(loop, dict):
                pending.append(loop.get("task"))

    def _record_file(self, path: str, content: str) -> None:
        self.file_contents[self._path(path)] = content
        if path.lower().startswith("resources/jobs/"):
            try:
                document = yaml.safe_load(content)
            except yaml.YAMLError:
                document = None
            resources = document.get("resources") if isinstance(document, dict) else None
            jobs = resources.get("jobs") if isinstance(resources, dict) else None
            for settings in jobs.values() if isinstance(jobs, dict) else []:
                if not isinstance(settings, dict):
                    continue
                name = settings.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                job = self._job(name=name)
                self._link_notebooks(job, settings)
                if isinstance(settings.get("schedule"), dict) and settings["schedule"]:
                    self.evidence["schedule_configuration_required"].add(job)
        self._refresh_linked_evidence()

    @staticmethod
    def _tables(sql: str) -> set[str]:
        import sqlglot
        from sqlglot import exp
        from sqlglot.optimizer.scope import Scope, traverse_scope

        try:
            statements = sqlglot.parse(sql, read="databricks")
            tables = set()
            for statement in statements:
                if statement is None:
                    continue
                virtual_tables = {
                    id(table)
                    for scope in traverse_scope(statement)
                    for table in scope.tables
                    if isinstance(scope.sources.get(table.alias_or_name), Scope)
                }
                tables.update(
                    ".".join(part for part in (table.catalog, table.db, table.name) if part).lower()
                    for table in statement.find_all(exp.Table)
                    if id(table) not in virtual_tables
                )
            return tables
        except (sqlglot.errors.ParseError, sqlglot.errors.TokenError):
            return set()

    def _refresh_linked_evidence(self) -> None:
        for path, owners in self.notebook_jobs.items():
            content = self.file_contents.get(path)
            if not content:
                continue
            self.evidence["structural_logic_required"].update(owners)
            tables = self._tables(content)
            for sql in self.comparison_queries:
                query_tables = self._tables(sql)
                # Every queried source must belong to this job's inspected code;
                # an unrelated comparison cannot bless a runtime explanation.
                if query_tables and query_tables <= tables:
                    self.evidence["runtime_comparison_required"].update(owners)

    def _sql_job(self, sql: str) -> str | None:
        """Resolve only an unambiguous job_id equality, never an OR/range/list."""
        import sqlglot
        from sqlglot import exp

        try:
            tree = sqlglot.parse_one(sql, read="databricks")
        except (sqlglot.errors.ParseError, sqlglot.errors.TokenError):
            return None
        if tree is None or tree.find(exp.Or) or tree.find(exp.Not):
            return None
        where = tree.args.get("where")
        if where is None:
            return None
        values = set()
        terms = list(where.this.flatten()) if isinstance(where.this, exp.And) else [where.this]
        for comparison in terms:
            if not isinstance(comparison, exp.EQ):
                continue
            for column, literal in ((comparison.left, comparison.right), (comparison.right, comparison.left)):
                if isinstance(column, exp.Column) and column.name.lower() == "job_id" and isinstance(literal, exp.Literal):
                    values.add(literal.this)
        return self._job(next(iter(values))) if len(values) == 1 else None

    def record_tool_result(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        result: Any,
    ) -> None:
        if _result_failed(result):
            return
        if tool_name in _RUN_HISTORY_TOOLS:
            self.run_history_used = True
            self._job(result.get("job_id"), result.get("run_name"))
            for run in result.get("runs", []):
                self._job(run.get("job_id"), run.get("run_name"))

        if tool_name == "databricks_list_job_runs":
            self.latest_list_runs = result
            return
        if tool_name == "databricks_get_recent_runs":
            job = self._job(result.get("job_id", tool_input.get("job_id")))
            summary = result.get("successful_run_summary") or {}
            if job is not None and summary.get("successful_run_count", 0) > 1:
                self.evidence["recent_run_baseline_required"].add(job)
            self.latest_recent_runs = result
            return
        if tool_name == "github_get_file_content":
            self._record_file(str(tool_input.get("path", "")), str(result.get("content", "")))
            return
        if tool_name != "databricks_execute_sql":
            return

        sql = str(tool_input.get("sql", "")).lower()
        columns = [str(column).lower() for column in result.get("columns", [])]
        rows = [dict(zip(columns, row)) for row in result.get("sample_rows", [])]
        if "databricks_workspace_jobs" in sql:
            queried_job = self._sql_job(sql)
            for row in rows:
                job = self._job(row.get("job_id"), row.get("job_name")) or queried_job
                if job is None:
                    continue
                settings = row.get("job_settings_json") or {}
                if isinstance(settings, str):
                    try:
                        settings = json.loads(settings)
                    except ValueError:
                        settings = {}
                if isinstance(settings, dict):
                    self._link_notebooks(job, settings)
                if any(value and ("schedule" in column or "cron" in column) for column, value in row.items()) or (isinstance(settings, dict) and settings.get("schedule")):
                    self.evidence["schedule_configuration_required"].add(job)
            self._refresh_linked_evidence()
            return

        is_metadata = "information_schema" in sql or re.match(r"^\s*(?:describe|desc|show)\b", sql)
        if (not is_metadata and int(result.get("row_count", 0) or 0) > 1
                and _TIME_CONCEPT.search(sql) and _COMPARISON_SQL.search(sql)):
            self.comparison_queries.append(sql)
            # Runtime telemetry keyed by job ID is directly attributable.
            job = self._sql_job(sql)
            if job is not None:
                self.evidence["runtime_comparison_required"].add(job)
            self._refresh_linked_evidence()

    def _claim_jobs(self, sentence: str) -> set[str]:
        # Preserve inline code in identity matching: job names/IDs are commonly
        # formatted as `Coverage` even though code is excluded from claim text.
        explicit_ids = set(re.findall(r"\bjob(?:[ _-]?id)?\s*[:#=]?\s*`?(\d+)\b", sentence, re.I))
        if explicit_ids - self.jobs:
            return set()
        matched = set(explicit_ids)
        for job, name in self.job_names_by_id.items():
            if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", sentence, re.I):
                matched.add(job)
        # An explicitly named subject cannot borrow the known subject's
        # evidence in a compound sentence ("Coverage and UnknownJob are slow").
        remainder = sentence
        for job in matched:
            name = self.job_names_by_id.get(job)
            if name:
                remainder = re.sub(re.escape(name), "KNOWN_JOB", remainder, flags=re.I)
        subjects = re.finditer(
            r"(?:^|\band\s+|\bbut\s+|\bwhile\s+)\s*`?"
            r"([A-Za-z_][\w.-]*(?:\s+[A-Za-z_][\w.-]*){0,3})`?\s+"
            r"(?:is|are|was|were|looks?|seems?|runs?)\b", remainder
        )
        for subject in subjects:
            if not re.fullmatch(
                r"(?:(?:(?:the|this)\s+)?KNOWN_JOB(?:\s+job)?"
                r"(?:\s+and\s+KNOWN_JOB)?|it|this job|the job|the run|today's duration)",
                subject.group(1), re.I,
            ):
                return set()
        if matched:
            return matched
        # Implicit references are safe only for a single known job. Unknown
        # names (including code spans) cannot inherit that job's evidence.
        implicit = re.match(
            r"^[\s*\->]*(?:it\b|its\b|this (?:job|run|duration)\b|"
            r"the (?:job|run|notebook|configured|higher|source|duration|runtime)\b|"
            r"today(?:['’]s)?\b|higher (?:source |data )?volume\b)", sentence, re.I
        )
        if len(self.jobs) == 1 and implicit and not re.search(r"`[^`]+`|\bjob\s+[A-Z0-9]", sentence):
            return set(self.jobs)
        return set()

    def validate(self, text: str) -> list[str]:
        failures: set[str] = set()
        without_fences = re.sub(r"```.*?```", _fence_comments, text, flags=re.DOTALL)
        for sentence in _sentences(without_fences):
            prose = _prose_without_code(sentence)
            claims = {"schedule_configuration_required": _contains_cadence_claim(prose)}
            if self.run_history_used:
                claims.update({
                    "recent_run_baseline_required": bool(_BASELINE_CLAIM.search(prose)),
                    "structural_logic_required": _contains_causal_claim(prose, _STRUCTURAL_TOPIC),
                    "runtime_comparison_required": bool(_contains_causal_claim(prose, _RUNTIME_CAUSE_TOPIC) or _RUNTIME_TREND_CLAIM.search(prose)),
                })
            jobs = self._claim_jobs(sentence)
            for category, claimed in claims.items():
                if claimed and (not jobs or not jobs <= self.evidence[category]):
                    failures.add(category)
        return [category for category in self.evidence if category in failures]

    def revision_instruction(self, violations: list[str]) -> str | None:
        """Return a private correction request covering every violation.

        Every category gets one revision pass; the deterministic fallback in
        ``safe_fallback`` is used only when the revised draft still violates.
        Returns None only when there is nothing to revise.
        """
        requests = [
            _REVISION_REQUESTS[violation]
            for violation in violations
            if violation in _REVISION_REQUESTS
        ]
        if not requests:
            return None
        body = " ".join(requests)
        return (
            "Revise your previous draft before answering the user. "
            f"{body} Keep every other supported statement. "
            "Do not mention this internal correction request."
        )

    def _list_runs_fallback(self, result: dict[str, Any]) -> str:
        runs = list(result.get("runs") or [])
        run_count = len(runs)
        distinct_jobs = {
            str(run.get("job_id") or run.get("run_name"))
            for run in runs
            if run.get("job_id") is not None or run.get("run_name")
        }
        succeeded = sum(run.get("result_state") == "SUCCESS" for run in runs)
        active = sum(
            str(run.get("life_cycle_state", "")).upper()
            in {"PENDING", "QUEUED", "RUNNING", "TERMINATING", "BLOCKED"}
            for run in runs
        )
        other_terminal = sum(
            bool(run.get("result_state")) and run.get("result_state") != "SUCCESS"
            for run in runs
        )
        selected_date = result.get("date") or "the requested period"
        timezone_name = str(result.get("timezone") or "America/New_York")
        lines = [
            f"*Observed job runs — {selected_date}*",
            "",
            f"- {run_count} executions across {len(distinct_jobs)} jobs",
            (
                f"- Results observed: {succeeded} succeeded, {other_terminal} other terminal, "
                f"{active} active"
            ),
        ]
        if result.get("truncated"):
            lines.append("- The tool result was truncated; these counts cover returned runs only")

        names = [
            str(run.get("run_name") or f"job_id {run.get('job_id')}")
            for run in runs
        ]
        repeated = [
            (name, count)
            for name, count in Counter(names).most_common()
            if count > 1
        ][:5]
        if repeated:
            lines.extend(["", "*Repeated executions observed*"])
            for name, count in repeated:
                matching_runs = [
                    run
                    for run in runs
                    if str(run.get("run_name") or f"job_id {run.get('job_id')}") == name
                ]
                observed_times = [
                    observed
                    for observed in (
                        _observed_start_time(run, timezone_name)
                        for run in sorted(
                            matching_runs,
                            key=lambda item: int(item.get("start_time_ms") or 0),
                        )
                    )
                    if observed
                ]
                displayed_times = ", ".join(observed_times[:4])
                if len(observed_times) > 4:
                    displayed_times += f", +{len(observed_times) - 4} more"
                suffix = f": {displayed_times}" if displayed_times else ""
                lines.append(f"- `{name}` — {count} starts{suffix}")

        with_durations = [
            (run, duration)
            for run in runs
            if (duration := _run_duration_ms(run)) is not None
        ]
        if with_durations:
            lines.extend(["", "*Longest observed executions*"])
            for run, duration in sorted(
                with_durations,
                key=lambda item: item[1],
                reverse=True,
            )[:5]:
                name = str(run.get("run_name") or f"job_id {run.get('job_id')}")
                lines.append(f"- `{name}` — {_human_duration(duration)}")

        lines.extend(
            [
                "",
                "*Evidence limit*",
                "- Configured cadence, normal duration, workload, and causes were not inferred.",
            ]
        )
        return "\n".join(lines)

    def _recent_runs_fallback(self, result: dict[str, Any]) -> str:
        summary = result.get("successful_run_summary") or {}
        runs = list(result.get("runs") or [])
        job_id = str(result.get("job_id", "unknown"))
        job_name = self.job_names_by_id.get(job_id)
        label = f"`{job_name}`" if job_name else f"job_id `{job_id}`"
        lines = ["*Observed duration comparison*", "", f"- Job: {label}"]

        latest = runs[0] if runs else {}
        latest_duration = _run_duration_ms(latest)
        if latest_duration is not None:
            lines.append(f"- Latest observed duration: {_human_duration(latest_duration)}")

        successful_count = int(summary.get("successful_run_count", 0) or 0)
        median_ms = summary.get("median_duration_ms")
        min_ms = summary.get("min_duration_ms")
        max_ms = summary.get("max_duration_ms")
        if successful_count:
            lines.append(f"- Recent successful runs compared: {successful_count}")
        if median_ms is not None:
            lines.append(f"- Median: {_human_duration(int(median_ms))}")
        if min_ms is not None and max_ms is not None:
            lines.append(
                f"- Observed range: {_human_duration(int(min_ms))} to "
                f"{_human_duration(int(max_ms))}"
            )
        if latest_duration is not None and median_ms:
            difference_pct = (latest_duration / int(median_ms) - 1) * 100
            direction = "above" if difference_pct >= 0 else "below"
            lines.append(
                f"- Latest vs. median: {abs(difference_pct):.1f}% {direction}"
            )

        lines.extend(
            [
                "",
                "*Evidence limit*",
                "- The retrieved evidence does not establish the cause of the duration.",
            ]
        )
        return "\n".join(lines)

    def safe_fallback(self) -> str:
        if self.latest_recent_runs:
            return self._recent_runs_fallback(self.latest_recent_runs)
        if self.latest_list_runs:
            return self._list_runs_fallback(self.latest_list_runs)
        return (
            "*Observed evidence only*\n\n"
            "The retrieved run evidence did not support the draft's interpretation. I can report "
            "the observed run details or inspect the relevant configuration and history."
        )
