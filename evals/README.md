# Conversation evaluations

This directory contains the required model-in-the-loop PR quality gate and a
harness for comparing models and effort levels on quality, cost, and latency.
It sends the production system prompt and production tool schemas to the
configured Anthropic model, but all GitHub and Databricks responses come from
synthetic fixtures. The evaluator has no implementation path to Slack or live
data tools.

The initial `signups_table_explanation` scenario reproduces the planning
mistakes seen in production. It requires zero user-visible guard rejections and
zero tool errors, exact metadata verification before analytical queries, no SQL
against the build-only `internal_sum` CTE field, and a completed evidence-backed
answer; whether that answer names `_anonymousId` and the browser/device
identity model is reported as soft coverage, not gated. The runner uses the same automatic metadata preflight as production and
records those injected checks as a metric without adding them to model-visible
messages. Valid SQL projections outside the fixture's bounded semantic model
return an explicit non-error synthetic limitation with no rows; the model is
told not to infer zero or production absence, and the limitation is counted in
the evaluation summary. Synthetic metadata and repository searches follow the
same zero-row/not-found and path-filter contracts as production, and the object
catalog includes prompt-documented objects that the scenario may legitimately
inspect. This keeps failures focused on model behavior instead of fixture gaps.

Unsupported schedule/cadence claims receive one private revision request in
both production and the evaluator. Only the revised answer is delivered or
retained; a still-unsupported revision falls back to evidence-only text.

The runner also mirrors the worker's numeric grounding pass
(`utils.numeric_guard`): the system prompt, the user prompt, and every
model-visible tool input/result are grounded; on `end_turn` any figure in the
draft that no grounded number supports triggers one private revision request
(`NUMERIC_GUARD_MODE=revise`, `NUMERIC_GUARD_MAX_REVISIONS=1` by default, same
env as production), then a visible footnote if the revision is still
unverified. The draft and instruction are removed from the recorded
conversation afterwards; the model-visible message sequence is identical to
production. The trace records `numeric_guard_mode`, `numeric_guard_revisions`,
and `numeric_guard_unverified_figures`.

Two further scenarios target numbers specifically. `signups_column_count`
reproduces a production answer that said "36 columns" for a 35-column table
because the model echoed a result-set row count instead of `column_count`.
`signup_percent_change` reproduces in-head arithmetic: two monthly
aggregates come back from SQL and the percent change must come from the
`calculate` tool or from the SQL itself, never from the model. Each fixture's
`notes` field explains the mistake it guards against.

## Layout

| File | Purpose |
|---|---|
| `run.py` | CLI: single run, baseline-judged PR gate (`--baseline`), baseline recording (`--write-baseline`), model/effort/repeat matrix |
| `conversation_runner.py` | One production-shaped conversation; records usage, latency, cost |
| `scoring.py` | Deterministic pass/fail plus a graded score per trace |
| `regression.py` | Pure regression rule: baseline vs current per scenario, verdict table, exit codes |
| `baseline.json` | Committed baseline the PR gate judges against: one entry per (provider, model, effort), each with per-scenario counts pooled from one or more runs (see [The PR gate](#the-pr-gate-a-baseline-relative-regression-check)) |
| `gate_scope.py` | Path filter (stdlib only) that decides whether a PR can change model behavior and so whether the gate runs |
| `aggregate.py` | Per-(provider, model, effort) summary table and percentiles |
| `pricing.py` | List-price snapshot used for cost estimates (see caveat below) |
| `synthetic_tools.py` | Fixture-backed GitHub/Databricks tool dispatcher |
| `fixtures/*.json` | Scenarios; the file stem is the scenario name |

`synthetic_tools.py` still contains scenario-specific branches (signups
projections, lineage shapes, job-history objects). Generalising those is a
separate piece of work; new scenarios may need matching dispatcher support.
Fixture-driven data the dispatcher understands without code changes:

- `objects.<full_name>.columns` — `[name, type, comment]` or
  `[name, type, comment, nullable]` (`true`/`false` or `"YES"`/`"NO"`; omitted
  means nullable). `describe_object` and `information_schema.columns` serve
  the flag; `column_count` is always `len(columns)`.
- `objects.<full_name>.create_sql` — optional; `SHOW CREATE TABLE` derives a
  DDL statement from the column list when absent.
- `sql_results` — ordered list of `{"pattern", "columns", "rows",
  "description"}`. `pattern` is a regex matched with `re.search` (DOTALL)
  against the model's SQL after backtick removal, whitespace collapsing, and
  lower-casing (for example `.*count\\(.*\\).*group by.*month.*`). The first
  match wins and is consulted **before** every scenario-specific branch. Output
  column names come from the model's own projection when its width matches the
  rows, so aliases line up; otherwise the declared `columns` are used. An entry
  with `"unmodeled": true` (no `rows`; optional `result_scope` text) serves the
  `projection_not_modeled` limitation to SQL matching its pattern, so a grain
  the fixture does not model (for example a daily breakdown) is never answered
  with rows meant for another grain. Keep row-serving patterns anchored to the
  grain they model; `signup_percent_change.json` shows the shape.
- `github_file` and `lineage_rows` are optional. Without a `github_file`,
  searches return zero matches and file reads return a
  `no_github_source` error.
- `notes` — free text for humans; ignored by the runtime and the scorer.

## The PR gate: a baseline-relative regression check

The required `Conversation evaluations` check used to run each fixture **once**
and fail the PR on any hard violation. Against a nondeterministic model that is
a coin flip: on an earlier prompt revision
`signup_percent_change` failed about one run in three for a real, known
model behavior (a describe-first miss), so dependency bumps and error-message
changes went red and were rerun until green.
The gate now has three parts, and a red check always says *which* of two
things it is — a behavioral regression or an infrastructure error — so
"rerun until green" is never the right response.

### 1. It only runs when the PR can change model behavior

The `Evaluation scope` job runs `python -m evals.gate_scope --base <sha>
--head <sha>` (stdlib only, no dependency install) over the PR's merge-base
diff and writes `run=true|false`. The `Conversation evaluations` job has `if:
needs.eval-scope.outputs.run == 'true'`; when it is skipped the required check
still counts as passing for branch protection. The gate runs when the PR
touches any of:

| Path | Why |
|---|---|
| `prompts/**` | system prompt, business context, provider addenda |
| `tools/**` | tool schemas and the results the model sees |
| `utils/**` | guards, provider adapters, numeric grounding |
| `worker.py` | the agent loop the harness mirrors |
| `evals/**` (except `*.md`) | fixtures, scoring, runner, baseline — the measuring instrument; this README is not |
| `.github/workflows/pr-check.yml` | the gate itself |
| `pyproject.toml` / `uv.lock` | **only** when the `anthropic` or `openai` constraint or locked version differs between base and head (`gate_scope.model_sdk_pins`); a `boto3` bump does not run it |

Everything else (docs, tests, `scripts/`, `infrastructure/`, `receiver.py`,
other workflows) skips the gate. The decision and its reason are printed in the
job log and step summary. The existing Dependabot skip is unchanged (see
[Dependabot PRs](#dependabot-prs)).

### 2. It judges 3 repeats per scenario against a committed baseline

The job runs

```bash
python -m evals.run --repeats 3 --concurrency 3 --baseline evals/baseline.json --output-dir eval-results
```

so every fixture is run three times (nine conversations today) with the
production model and effort, and the result is judged by the rule in
`regression.py` against the entry in `evals/baseline.json` for that
(provider, model, effort), **not** by "did any conversation fail". Cost is
roughly **$1–2.50 and 5–8 minutes per gated PR** (about $0.26 per conversation
at Sonnet 4.6 / high, the longest fixture takes up to five minutes). The
verdict table is appended to the job summary and written to
`eval-results/verdict.md` / `verdict.json` alongside the usual per-run traces.

> **This repository ships without a recorded `evals/baseline.json`.** Baselines
> are pass rates measured from paid model runs, so whoever deploys the bot
> records their own (see [Re-baselining is deliberate](#re-baselining-is-deliberate)).
> Until the file exists, the PR workflow skips the gate with a notice and the
> tests that validate the committed file are skipped. The numbers and hashes
> below are **illustrative**.

`evals/baseline.json` (`schema_version` 2) holds one entry per
`<provider>/<model>/<effort>` cell; each entry has one record per fixture whose
repeats may be pooled from several runs, with the per-run split kept in
`sources` so a reviewer can see where 4/6 came from, and a `prompt_version`
naming the commit and system-prompt hash the samples measured:

```json
"baselines": {
  "anthropic/claude-sonnet-4-6/high": {
    "provider": "anthropic", "model": "claude-sonnet-4-6", "effort": "high",
    "recorded_at": "2026-09-08T15:02:11+00:00",
    "prompt_version": {
      "git_commit": "0123456789abcdef0123456789abcdef01234567",
      "system_prompt_sha256": "0123456789ab"
    },
    "source_run_urls": [
      "https://github.com/.../actions/runs/<run-id>",
      "https://github.com/.../actions/runs/<run-id>"
    ],
    "scenarios": {
      "signups_table_explanation": {
        "repeats": 6, "passed": 4, "pass_rate": 0.6667, "mean_score": 0.9167,
        "hard_violation_codes": {"avoidable_tool_failure": 2},
        "sources": [
          {"source_run_url": "https://github.com/.../runs/<run-id>", "repeats": 3, "passed": 2},
          {"source_run_url": "https://github.com/.../runs/<run-id>", "repeats": 3, "passed": 2}
        ]
      }
    }
  },
  "anthropic/claude-sonnet-5/high": { "...": "same two runs, R = 6: 6/6, 6/6, 6/6" }
}
```

`prompt_version.git_commit` is the commit whose prompts the runs measured
(`--prompt-commit`, else `GITHUB_SHA`, else `git rev-parse HEAD` of the
checkout that ran `--write-baseline`) and `system_prompt_sha256` is the first
12 hex characters of the sha256 of `load_system_prompt(provider)` from that
checkout. The verdict header prints both and says whether the run under test
sends the same prompt; a mismatch is informational — the gate never fails on
it, and entries recorded before the field existed are judged exactly as
before (the header reads `Baseline prompt: not recorded`).

The gate selects the entry whose key matches the cell of the run being judged.
That is what lets a candidate model's baseline be committed **before** the
switch: its entry is unused until `DEFAULT_MODEL` moves, and then it is what
the gate compares against. A run whose cell has no entry
cannot be judged; the verdict is `no_model_baseline`, the headline reads `NO
BASELINE FOR THIS MODEL (cannot judge)`, the runner prints `NO BASELINE: ...`
naming the missing key and the keys the file does have, and the process exits
**2** (the "cannot judge" code shared with infrastructure errors — never 1,
because nothing regressed).

Once the file exists, `tests/test_eval_regression.py` asserts it parses, has an entry for the
production default model/effort, that every entry covers exactly the fixtures
on disk with at least three repeats per scenario, consistent `sources` and a
`prompt_version`, and that the production entry pools at least two runs so no
scenario has a `0.000` threshold.

### 3. The rule

For each scenario, with `N` current repeats and `R` baseline repeats:

```
margin = max(1/N, 1/sqrt(N·R))
```

The scenario is a **regression** when either

1. **the pass-rate rule** — `current_pass_rate < baseline_pass_rate − margin`.
   This is the primary instrument; or
2. **the new-code rule** — a hard violation code that has **zero occurrences
   in every scenario of the baseline entry** (entry-wide, not just this
   scenario's record) occurs in **at least 2 of the N current repeats** of a
   scenario, **or in at least 50 % of them** (`NEW_CODE_MIN_REPEATS = 2`,
   `NEW_CODE_MIN_FRACTION = 0.5` in `regression.py`; N = 3 needs 2 of 3, N = 2
   fires at 1 of 2, N = 6 needs 3 of 6).

Why the new-code rule is entry-wide and needs recurrence:

- A code already recorded *anywhere* in the entry is a **known failure mode**
  of this model and prompt. How often it happens is exactly what the pass-rate
  margin governs; seeing it in a scenario whose own baseline happened not to
  hit it is not a new kind of behavior. An earlier eval run showed the
  problem: `signup_percent_change` (6/6 in the baseline) failed one
  repeat with `avoidable_tool_failure`, which the same entry had already
  recorded twice under `signups_table_explanation`, and the old rule
  called that a regression at 2/3 — above its own 0.667 threshold.
- **A small clean baseline sample cannot prove a code is absent.** With
  `R = 6` a failure mode with a true rate of about 10 % has
  `0.9⁶ ≈ 53 %` chance of zero observations in the baseline, so under the old
  rule its first later occurrence would have read as "new" about half the
  time with the PR under test doing nothing wrong. The same run's
  `signups_table_explanation` did exactly that: `unsupported_answer_claim`
  once in three repeats, at 2/3 passes against a 0.333 threshold. Requiring
  two repeats (or half of them) turns a single sighting into information, not
  a verdict.

Codes absent from the *scenario's own* baseline are still shown per row in
the **`Unseen here`** column with the number of current repeats they occurred
in (`` `unsupported_answer_claim` 1/3 ``), and an `Informational:` line below
the table says why they did not fail the row. That column never changes a
verdict.

Any run that ends `status: "errored"` (provider 5xx/429/timeout after the
runner's retries, missing key, missing package) makes the verdict **errored**
instead: the remaining samples cannot be trusted and the cause is not the PR.
The process exits **2** (`EXIT_ERRORED`), distinct from **1** for a
regression, and the workflow annotates the check `Conversation evaluations:
infrastructure error` rather than `behavioral regression`. The errored row's
reason is built from each run record's structured `error` — exception class,
API `code`/`type`, HTTP status, redacted message and request id (for example
`AuthenticationError/invalid_api_key (401) x3 ... Provider said:
AuthenticationError/invalid_api_key (401): Incorrect API key provided: sk-…
[request_id=req_…]`) — so a 401 for a wrong key, a missing scope or no access
to the model is diagnosable from the verdict table without opening the artifact.

Arithmetic with the gate's `N = 3` and a pooled baseline of `R = 6`
(illustrative numbers):

```
margin = max(1/3, 1/sqrt(18)) = max(0.333, 0.236) = 0.333
```

| Baseline | Threshold (`baseline − 0.333`) | Passes | Regresses |
|---|---|---|---|
| 6/6 | 0.667 | 3/3, 2/3 | 1/3, 0/3 |
| 4/6 (a hard code recorded twice, a known failure mode) | 0.333 | 3/3, 2/3, 1/3 | 0/3 |

A code already recorded in the entry is a known failure mode, so the pass-rate
margin governs how often it may happen. Once a better model or prompt is
recorded with no hard code at all, that same code recurring in 2 of 3 repeats
**is** a new code and a regression, and the threshold rises (0.333 to 0.667 in
the table above).

Seeding an entry from a single 3-repeat run can leave a scenario that happened
to fail once with a `0.000` threshold: it could drop to 0/3 forever without the
gate noticing. Pooling a second run into the same entry gives it a real
threshold, which is why every entry should be built from two artifacts and why
the test suite insists on `R ≥ 6` and a positive threshold for every production
scenario once a baseline is committed.

The `1/N` term means the tolerance is never smaller than one conversation, so
a single flip can never fail the gate on its own; the `1/sqrt(N·R)` term is
the standard-error scale of a proportion estimated from `N·R` samples and
takes over when both sides have many repeats (`N = R = 10` gives 0.1). A
scenario whose known failure rate is already high has a low threshold — that
failure is *known* and is the prompt work's job to fix — but it still
regresses if a new *kind* of hard failure (one the whole entry has never
recorded) shows up in two of its repeats. Individual failed conversations
within tolerance pass.

Three more verdicts exist for a baseline that cannot judge the run, each with
an actionable reason: `no_baseline` (exit 1: the entry has no record for a
fixture — record one), `baseline_mismatch` (exit 1: the run spans several
provider/model/effort cells — judge one at a time), and `no_model_baseline`
(exit 2: the file has no entry at all for the run's cell — add one in the same
PR as the model/effort change; see above).

The verdict table looks like (the `Baseline entry` line names the cell the run
was matched on):

```
- Baseline: `evals/baseline.json`
- Baseline entry: `anthropic/claude-sonnet-5/high`
- Baseline prompt: commit `0123456789ab`, system prompt sha256 `0123456789ab` (same prompt as this run)

| Scenario | Baseline pass | Current pass | Threshold | New codes | Unseen here | Verdict |
|---|---|---|---|---|---|---|
| `signups_column_count` | 6/6 (100%) | 3/3 (100%) | 0.667 | - | - | **pass** |
| `signups_table_explanation` | 4/6 (67%) | 2/3 (67%) | 0.333 | - | `unsupported_answer_claim` 1/3 | **pass** |
| `signup_percent_change` | 6/6 (100%) | 1/3 (33%) | 0.667 | `unsupported_answer_claim` | `unsupported_answer_claim` 2/3 | **regression** |

- `signups_table_explanation`: Informational: `unsupported_answer_claim` is not in any scenario of the baseline entry but occurred in only 1/3 repeat(s), below the 2-repeat / 50% bar; the pass-rate rule governs it.
- `signup_percent_change`: pass rate 1/3 is below 0.667 (baseline 6/6 minus margin 0.333); hard violation code(s) never recorded in any scenario of the baseline entry recurred: `unsupported_answer_claim` in 2/3 repeats.
```

`New codes` is the new-code rule's output (it fails the row); `Unseen here`
is informational (codes this scenario's own baseline lacks, with how many of
the current repeats hit them).

When the PR changes the prompt without re-baselining, the third line instead
reads `— **differs from the prompt under test** (`<sha>`)`; that is a hint to
re-baseline in the PR, not a failure.

### Re-baselining is deliberate

The rule judges a PR against the baseline **on the PR branch**. So a PR that
changes prompts, tools, fixtures, scoring, or the model/effort must also
update the entry it wants to be judged against, in the same PR — and
reviewers see the before/after pass rates and violation codes in the
`evals/baseline.json` diff. Do not edit the numbers by hand. Either:

- run the **Model comparison** workflow on the branch (`gh workflow run
  model-comparison.yml --ref <branch> -f models=anthropic:claude-sonnet-5
  -f efforts=high -f repeats=3`), download the artifact, and record it without
  spending again:

  ```bash
  gh run download <run-id> -n model-comparison-results -D comparison
  python -m evals.run --from-results comparison \
    --models anthropic:claude-sonnet-5 --efforts high \
    --write-baseline evals/baseline.json \
    --source-run-url https://github.com/example-org/databricks-slack-agent/actions/runs/<run-id> \
    --prompt-commit <sha the comparison ran on>
  ```

- or run it locally with the eval key:

  ```bash
  MODEL_EVAL_ANTHROPIC_API_KEY=... python -m evals.run --repeats 3 --concurrency 3 \
    --write-baseline evals/baseline.json --output-dir eval-results
  ```

`--write-baseline` refuses runs with infrastructure errors (a baseline records
clean samples only) and requires a single (provider, model, effort) cell. It
writes (or replaces) **that cell's entry** and keeps every other entry already
in the file, so recording one model never disturbs another model's entry; to drop an entry, delete the file and
re-record the ones you want. A file it cannot parse is an error, not
overwritten. Use at least three repeats; more repeats tighten the tolerance
(`1/sqrt(N·R)`). Commit the file with the change it measures.

**Recording which prompt was measured.** The writer stamps each entry with
`prompt_version`: `git_commit` (from `--prompt-commit`, else `GITHUB_SHA`,
else `git rev-parse HEAD`) and the sha256 prefix of `load_system_prompt()`
**from the checkout you run the command on**. With `--from-results`, run it on
the commit the comparison ran on (or one whose `prompts/` are identical) so
the hash describes the prompt the samples actually saw, and pass
`--prompt-commit` explicitly when your working branch is not that commit.
Without any commit id (no git, no `GITHUB_SHA`, no flag) the entry is written
without the field and the command says so.

**Pooling several runs into one entry.** Give `--from-results` more than once
(and `--source-run-url` once per directory, in the same order — or once for
all of them). Records from every tree are pooled per scenario, so two 3-repeat
comparisons give `R = 6`, and each scenario's `sources` records the split. A pooled entry is built like this from two comparison runs (each `high` × 3
repeats):

```bash
gh run download <run-id> -n model-comparison-results -D cmp3
gh run download <run-id> -n model-comparison-results -D cmp4
python -m evals.run \
  --from-results cmp3 --from-results cmp4 \
  --models anthropic:claude-sonnet-4-6 --efforts high \
  --write-baseline evals/baseline.json \
  --source-run-url https://github.com/example-org/databricks-slack-agent/actions/runs/<run-id> \
  --source-run-url https://github.com/example-org/databricks-slack-agent/actions/runs/<run-id> \
  --prompt-commit <sha>
# e.g. -> signups_column_count 6/6, signups_table_explanation 4/6, signup_percent_change 6/6 (R = 6)
```

**Adding an entry for another model.** The same command with another
`--models`/`--efforts` selection adds a second entry to the same file and
keeps the first. A second model's entry can come from the same two artifacts:

```bash
python -m evals.run \
  --from-results cmp3 --from-results cmp4 \
  --models anthropic:claude-sonnet-5 --efforts high \
  --write-baseline evals/baseline.json \
  --source-run-url https://github.com/example-org/databricks-slack-agent/actions/runs/<run-id> \
  --source-run-url https://github.com/example-org/databricks-slack-agent/actions/runs/<run-id> \
  --prompt-commit <sha>
# e.g. -> 6/6, 6/6, 6/6 (R = 6); the first entry is kept
```

Only the entry matching `DEFAULT_MODEL` / `DEFAULT_EFFORT` is exercised by the
gate; an entry for another model is ready for when the default moves.

**Where the files may live.** `--baseline` must name an existing `.json` file
and may not contain `..` components; `--write-baseline` must name a `.json`
path under the repository root or under `--output-dir`, again without `..`
(`evals/baseline.json` from the repository root satisfies both). Anything else
is rejected before any model is called.

If a PR *legitimately* lowers a pass rate (say, a stricter fixture), the same
procedure applies: re-baseline, and let the reviewer judge the drop in the diff
rather than the gate.

### What a red check means

- **`Conversation evaluations: behavioral regression`** (exit 1) — read the
  verdict table in the job summary: a scenario dropped below its threshold or
  produced a violation code the baseline never had. Fix the prompt/tool
  change, or, if the new behavior is intended, re-baseline in the PR. Do not
  rerun to get a different sample.
- **`Conversation evaluations: infrastructure error`** (exit 2) — a provider
  5xx/429/timeout survived the retries, or the key was unavailable. This one
  *is* rerunnable once the provider recovers; it says nothing about the PR.
  The same exit code (and check title) is used when `evals/baseline.json` has
  **no entry for the model/effort under test** (`NO BASELINE FOR THIS MODEL`
  in the verdict): nothing regressed, the gate simply cannot judge. The fix is
  to commit an entry, not to rerun.
- The check is **skipped** when the PR cannot change model behavior; that
  counts as passing.

## Local run

Use a dedicated, spend-capped evaluation key—not the bot's production secret:

```bash
MODEL_EVAL_ANTHROPIC_API_KEY=... \
  python -m evals.run --output-dir eval-results
```

With no flags the command runs one model (`$MODEL_EVAL_MODEL` or the
production default), one effort (`$MODEL_EVAL_EFFORT` or `high`), every
fixture in `evals/fixtures/` (currently three) once, and exits 1 on any hard
(behavioral) violation; soft content-coverage violations are reported without
failing (see [Scoring](#scoring)). That is a quick smoke test, not the gate:
to reproduce what CI decides, add `--repeats 3 --baseline evals/baseline.json`.
It writes `summary.md` and `summary.json` under `eval-results/` (plus
`trace.json` / `result.json` for a single run, and `verdict.md` /
`verdict.json` with `--baseline`), and the per-run file described below. The
trace contains only the synthetic conversation, visible assistant text, tool
inputs, and fixture results; thinking blocks and credentials are not persisted.

The standard `pytest` suite exercises fixture dispatch, every scoring rule, the
regression rule and baseline schema, the path filter, the matrix runner, retry
classification, and the usage report without an API key or network access.

## CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--provider anthropic\|openai` | `anthropic` | Provider for *bare* model ids. `openai` needs `$MODEL_EVAL_OPENAI_API_KEY` (the `openai` package is a locked runtime dependency); without the key every OpenAI cell is recorded as an infrastructure error, never a behavioral failure. Traces and run records carry `provider`. See `docs/PROVIDERS.md`. |
| `--models a,b` | `$MODEL_EVAL_MODEL` or the provider default (`claude-sonnet-5` / `gpt-5.6-terra`, the production models) | Comma-separated model ids, each optionally provider-qualified as `<provider>:<model>` (`anthropic:claude-sonnet-4-6,openai:gpt-5.6-terra`). Bare ids use `--provider`; an unknown prefix (`opnai:`) is rejected. One SDK client is built per provider named in the matrix. |
| `--efforts low,medium,high,max` | `$MODEL_EVAL_EFFORT` or `high` | Also accepts `xhigh` on models that support it |
| `--repeats N` | `1` | Runs per (model, effort, scenario) |
| `--scenarios a,b` | all fixtures | `--scenario x` remains as a single-scenario alias |
| `--thinking adaptive\|disabled\|none` | `adaptive` | `none` omits the parameter entirely |
| `--max-tokens N` | `24000` | Per-call `max_tokens` (production `MAX_TOKENS`: `recommended_max_tokens("claude-sonnet-5", "high")`, the value Terraform sets; above 21,000 the provider streams, exactly as in production) |
| `--max-api-calls N` | fixture `max_api_calls`, else `25` | Per-conversation model-call budget for comparison runs; the default is production's `worker.MAX_ITERATIONS` (see [Budgets mirror production](#budgets-mirror-production)) |
| `--concurrency N` | `2` | Thread pool size; keep low to respect the eval key's rate limit |
| `--output-dir PATH` | `eval-results` | |
| `--fail-on-violation` / `--no-fail-on-violation` | fail | Without `--baseline`, any failed conversation exits 1; ignored when `--baseline` or `--write-baseline` is given |
| `--baseline PATH` | | Judge the run against the committed baseline's entry for the run's (provider, model, effort) with the regression rule; exit 0 pass / 1 regression / 2 infrastructure error or no entry for this cell. Writes `verdict.md` and `verdict.json`. Needs a single cell. Must be an existing `.json` file without `..` components. The PR gate passes `evals/baseline.json`. |
| `--write-baseline PATH` | | Record the run's per-scenario pass counts and hard violation codes as the baseline entry for its cell, keeping other cells already in the file. Refuses runs with infrastructure errors (exit 2). Must be a `.json` path under the repository or `--output-dir`, without `..`. Mutually exclusive with `--baseline`. |
| `--from-results DIR` | | No model calls: load `rep*.json` records from an existing output tree (e.g. a downloaded comparison artifact) and apply `--baseline` or `--write-baseline`; repeat the flag to pool several trees into one entry (repeats add up per scenario); `--models`/`--efforts` pick one cell of a multi-cell tree. Records that carry a full trace are **re-scored with this checkout's `scoring.py`** before judging or recording (the stored `result` reflects the rules at recording time), so a severity change reaches existing artifacts; the command prints how many it re-scored |
| `--source-run-url URL` | Actions run URL when in CI, else null | Provenance recorded in the baseline (`source_run_urls` and per-scenario `sources`). Repeatable: one per `--from-results` directory in order, or one for all |
| `--prompt-commit SHA` | `GITHUB_SHA` in CI, else `git rev-parse HEAD` | With `--write-baseline` only: the commit whose prompts the recorded runs measured, stored as `prompt_version.git_commit` beside the sha256 prefix of `load_system_prompt()` from this checkout. 7–40 hex characters |
| `--list-scenarios` | | Print fixture names and exit |

Exit codes: `0` pass, `1` behavioral failure or regression (or a baseline that
cannot judge the run), `2` infrastructure error. `2` is used for every
infrastructure outcome — errored conversations, no buildable client, a
refused `--write-baseline` — so a workflow can label it distinctly.

Outputs: `<out>/<model>/<effort>/<scenario>/rep<k>.json` (trace + score +
status per run, each carrying `provider`), `<out>/summary.json`, and
`<out>/summary.md` with one row per (provider, model, effort) — the `Provider`
column comes first, so a head-to-head such as
`anthropic:claude-sonnet-4-6,openai:gpt-5.6-terra` is one table: pass rate, mean graded score, wall-clock mean/p50/p95, API-call
p50/p95, API calls mean/p95 and the number of runs that hit the API-call
budget, mean tool calls, mean tokens by class (input, cache write, cache read,
output), mean estimated cost, a violation histogram, and the
infrastructure-error count. The per-run lines show `api_calls=<used>/<budget>`
and `hit_budget=yes|no` so a comparison can see how close each model runs to
the production bound.

## Budgets mirror production

The harness must never bound a conversation tighter than the worker does;
otherwise a failure measures the harness, not the model (an earlier CI run
cut `signups_table_explanation` off at the old 12-call cap with
`empty_answer`, on a conversation production would have let finish).

| Runner constant | Default | Production source |
|---|---|---|
| `MAX_API_CALLS` | `25` | `worker.MAX_ITERATIONS`: the bot posts "Hit max iterations without completing" after this many model calls |
| `MAX_TOKENS_PER_CALL` | `24000` | `worker.MAX_TOKENS`: what `_resolve_max_tokens` yields for the default `claude-sonnet-5` / `high` with `ANTHROPIC_MAX_TOKENS` unset (`recommended_max_tokens`), and the value Terraform sets in production |
| `MAX_TOTAL_OUTPUT_TOKENS` | `MAX_TOKENS_PER_CALL × MAX_API_CALLS` = `600000` | Production has no separate per-conversation output-token cap; this is its effective bound. The runner derives the live value from the effective `--max-tokens` and API-call budget. |
| `MAX_FINAL_REVISIONS` | `1` | `worker.MAX_FINAL_REVISIONS` |

`tests/test_conversation_eval_scoring.py` compares these to `worker.py` by AST
(`worker` cannot be imported by the eval job: it wires boto3 and Slack clients
at import time), so the two cannot drift silently. Exhausting the API-call
budget is still a hard `api_call_budget` violation: at that point production
would have posted "Hit max iterations" to the user, which is a behavioral
failure, not a harness artefact. Each trace records `api_calls`,
`hit_api_call_budget`, and the effective `budget` (`max_api_calls`,
`max_api_calls_source` of `default`/`fixture`/`override`, `max_output_tokens`).

To run a scenario under a tighter bound, set `"max_api_calls": N` in its fixture
JSON, or pass `--max-api-calls N` for a whole comparison run; the CLI flag wins
over the fixture. Neither changes what counts as a failure.

## Infrastructure errors vs behavioral failures

Each conversation is wrapped. A `429`, any `5xx`, or an SDK connection/timeout
error re-runs the *whole* conversation up to two more times with jittered
backoff (on top of the SDK's own per-request retries). If it still fails, the
run is recorded with `status: "errored"` and the exception class name. Errored
runs are reported separately and never counted in the pass-rate denominator.
The process exits **2** when any run errored, with an `INFRASTRUCTURE ERROR`
message that is distinct from `BEHAVIORAL FAILURE` / `BEHAVIORAL REGRESSION`
(exit 1); with `--baseline` the verdict is `ERRORED (infrastructure)` and the
PR check is annotated as such. Non-transient errors (for example a `400` for
an unsupported effort level) are recorded as errored immediately without
retry.

## Scoring

`scoring.py` is deterministic and makes no model calls. Every violation carries
a `severity`. `passed` is true only with zero **hard** violations; the
regression rule counts passes and hard violation codes per scenario, so only
behavioral failures can move the gate; **soft** violations are reported but
never block. `score = 1 - weighted_violations` (hard 0.25, soft
0.05, clamped to `[0, 1]`) is reported alongside `passed` so repeated runs and
model comparisons can be compared on a graded scale; soft violations do lower
the score, so a coverage regression is visible in `summary.md` (`hard=N
soft=N` on every run line, `[hard]`/`[soft]` on every violation, and a
`Violations: hard N (block the gate), soft N (reported only …)` headline) even
though it does not fail the job.

What blocks (hard by default):

| Code | Raised by |
|---|---|
| `avoidable_tool_failure` | any tool error, including guard rejections, **except the `duplicate_sql` and `wrong_github_source` categories** (see the category table below); the message carries the guard's `missing_columns` and a one-line SQL snippet so the CI log alone explains the failure, and the violation carries the guard's `category` |
| `unsupported_answer_claim` | `forbidden_final_patterns` (schedule/cadence/baseline claims) |
| `required_number_missing`, `forbidden_number_present` | `required_final_numbers`, `forbidden_final_numbers` |
| `ungrounded_approximation` | `numeric_grounding`: a hedged figure no tool supports |
| `forbidden_sql_identifier` | `forbidden_identifier_scope` / `forbidden_sql_identifiers` |
| `metadata_ordering`, `missing_analytical_evidence`, `missing_tool_evidence` | `required_descriptions_before_query`, `required_analytical_objects`, `required_successful_tools` |
| `required_tool_missing` | `required_tool_calls` |
| `incomplete_conversation`, `empty_answer` | stop reason other than `end_turn`; no final text |
| `api_call_budget`, `output_token_budget` | the conversation used its whole API-call budget (production's `MAX_ITERATIONS`) without finishing, or exceeded the output-token bound; see [Budgets mirror production](#budgets-mirror-production) |

What is reported only (soft by default):

| Code | Raised by | Why soft |
|---|---|---|
| `missing_answer_fact` | `required_final_patterns` | Content coverage: a correct, grounded answer can phrase a fact differently or omit it without misusing a tool or inventing anything. |
| `ungrounded_number` | `numeric_grounding` | Mirrors the production numeric guard, which annotates rather than blocks. |
| `avoidable_tool_failure` with category `duplicate_sql` | `SqlRunGuard` rejecting a byte-identical re-run | A repeated identical query is a wasted call, not a wrong or unsafe action: the result was already in the conversation and production's cost is one round trip. An earlier eval run re-ran a `COUNT(*)/MAX(signup_date)` query to "re-verify" a figure after a numeric-guard revision request. |
| `avoidable_tool_failure` with category `wrong_github_source` | the fixture refusing a repository file it does not model (the stale `Signups MA - Merge.sql`) | A wasted read, not a wrong answer: in production the read simply succeeds. Comparison runs <run-id> and <run-id> each read the MA notebook once, after the current file, and still explained the table from the current file. *Using* the stale file is caught separately and hard: `signups_table_explanation.json` forbids presenting the MA notebook as the current or build source. |

**Tool-failure categories.** `avoidable_tool_failure` covers every tool
error; its severity is decided per *category* — the guard's
`validation_failure` (or a fixture's `fixture_failure` / `synthetic_failure`),
recorded on the violation as `category`. The table is
`AVOIDABLE_FAILURE_CATEGORY_SEVERITY` in `scoring.py`; categories it does not
list default to hard.

| Category | Severity | Why |
|---|---|---|
| `duplicate_sql` | **soft** | wasted round trip; same data, nothing wrong or unsafe executed |
| `column_not_in_live_schema` | hard | invented a column the live schema does not have |
| `object_metadata_required`, `system_object_metadata_required` | hard | analytical query before the object's exact metadata was verified |
| `invalid_struct_access_syntax` | hard | JSON-style `:` access on a live STRUCT column |
| `wrong_github_source` | **soft** | read a stale/unmodelled repository file; wasted round trip that succeeds in production. Presenting that file as the build source is a hard `unsupported_answer_claim` via the fixture's `forbidden_final_patterns` |
| `unparseable_sql` | hard | SQL the parser rejects |
| `write_statement`, `denied_function` | hard | blocked unsafe action |
| anything else (`schema_inspection_required`, `object_circuit_breaker`, `no_github_source`, `tool_error`, ...) | hard | default |

A fixture may move any code in either direction with
`"violation_severity": {"<code>": "soft" | "hard"}`, and a tool-failure
category on its own with `"avoidable_tool_failure:<category>"` (the
category-qualified key wins over the bare code, so
`{"avoidable_tool_failure": "hard", "avoidable_tool_failure:duplicate_sql":
"soft"}` hardens every tool error but the duplicate, and
`{"avoidable_tool_failure:wrong_github_source": "hard"}` makes a scenario that
is specifically about picking the right file block on the read). Promote
`missing_answer_fact` to `hard` only when the required pattern *is* the
behavioral point of the scenario rather than incidental coverage:
`signup_percent_change.json` does this for its "a percent figure is
stated" pattern, because an answer that never states the percent change has
not answered the question. `signups_column_count.json` leaves its
`(?i)nullable` pattern soft (coverage) while its `required_final_numbers`
check is already hard, and `signups_table_explanation.json` leaves its
`_anonymousId` / identity-model patterns soft; each fixture's `notes` says
why. `synthetic_limitation` counts stay informational metrics.

Fixture fields used by the scorer (all optional):

- `required_descriptions_before_query`, `required_analytical_objects`,
  `required_successful_tools`
- `required_tool_calls` — list of `{"tool": "calculate", "min": 1,
  "input_contains": "pct_change"}` (`min` defaults to 1; `input_contains` is
  an optional case-insensitive substring of the JSON-serialised tool input).
  Only non-error calls count. Violation: `required_tool_missing`.
- `forbidden_identifier_scope` — `{"table": "cat.sch.tbl", "identifiers": [..]}`
  (or `"tables": [...]`, or a list of such objects). Identifiers are only
  forbidden in SQL that reads the scoped table(s). The legacy top-level
  `forbidden_sql_identifiers` list still works and applies to every relation.
- `required_final_patterns`, `forbidden_final_patterns` — regexes over the
  final text (`missing_answer_fact`, soft by default; `unsupported_answer_claim`,
  hard).
- `required_final_numbers`, `forbidden_final_numbers` — figures as strings
  (`"35"`, `"12480"`). Matched as whole numeric tokens in the final text:
  thousands separators are normalised (`12,480` is `12480`), `35.0` equals
  `35`, and digits inside identifiers or longer digit runs (`col_36`, `1236`,
  `2.36`, `36th`, `us-east-1`) never match. Violations:
  `required_number_missing`, `forbidden_number_present`.
- `numeric_grounding` — when `true`, `utils.numeric_guard.NumericEvidenceGuard`
  is re-run over the trace: the user prompt plus every tool input and result
  are grounded (not the system prompt, so the check is at least as strict as
  production) and the final text is validated. Violation: `ungrounded_number`
  (soft by default) listing the figures. Hedged figures (`~12%`,
  `approximately 12%`, `about 1,500`) are judged against the same evidence:
  `~28.6%` for a `calculate` result of `28.587...` is rounding presentation and
  passes, while a hedged figure no tool supports is an in-head estimate and
  raises `ungrounded_approximation` (hard by default). Do not add blanket
  `~`/"approximately" regexes to `forbidden_final_patterns` for this. The
  metric `numeric_guard_revisions` reports how many private numeric revisions
  the runner issued.
- `violation_severity`

The runner records an output-token budget breach without masking a completed
answer: a final `end_turn` is still captured as `final_text` and the scorer
raises `output_token_budget` separately rather than `incomplete_conversation`.
The budget itself is production's effective bound (per-call `max_tokens` times
the API-call budget), not a tighter harness cap.

## Cost and latency capture

Each trace records, per model call, `latency_ms`, the `model` reported by the
API, `stop_reason`, and the full usage block (`input_tokens`,
`cache_creation_input_tokens`, `cache_read_input_tokens`, `output_tokens`); per
tool dispatch, `tool_name` and `latency_ms`; and per conversation, usage
totals, API/tool/wall-clock time, and `estimated_cost_usd`. OpenAI usage is
normalized to the same accounting (`input_tokens` excludes cached tokens,
which are reported as `cache_read_input_tokens`; there is no cache write).

`pricing.py` holds a list-price snapshot keyed by model id (USD per MTok) with
cache multipliers of 1.25x input for cache writes and 0.1x input for cache
reads; rows may instead carry an explicit `cached_input` price and
`cache_write_multiplier` (the `gpt-5.6-terra` row does).
**The table is a convenience for relative comparisons. Re-verify every
row against the official pricing page before quoting a number, and add a row
before trialling a new model id.** Unknown ids produce `estimated_cost_usd:
null` rather than a guess. Costs cover the evaluation's own API spend, not the
production bill; use the production usage report in
`scripts/export_sessions.py` for that.

## How to run a model comparison

1. Measure production first. `scripts/export_sessions.py` writes
   `model_usage_report.md` from the worker's `model_call` / `run_summary`
   audit events (per model and effort: calls per run, cache hit rate, tokens per
   run, estimated cost per run and month-to-date, p50/p95 latency, outcome and
   stop-reason histograms). That is the baseline any candidate must beat.
2. Run the matrix with a spend-capped key:

   ```bash
   MODEL_EVAL_ANTHROPIC_API_KEY=... python -m evals.run \
     --models claude-sonnet-5,claude-sonnet-4-6 \
     --efforts medium,high \
     --repeats 3 \
     --concurrency 2 \
     --no-fail-on-violation \
     --output-dir eval-results/model-comparison
   ```

   For a cross-vendor comparison qualify each id with its provider and supply
   both evaluation keys; the OpenAI provider sends the same system prompt
   (plus `prompts/providers/openai.md`) and tool schemas through the Responses
   API translation in `utils/llm/openai_provider.py`:

   ```bash
   MODEL_EVAL_ANTHROPIC_API_KEY=... MODEL_EVAL_OPENAI_API_KEY=... python -m evals.run \
     --models anthropic:claude-sonnet-4-6,openai:gpt-5.6-terra \
     --efforts high --repeats 3 --concurrency 2 \
     --no-fail-on-violation --output-dir eval-results/model-comparison
   ```

3. Read `summary.md`. Compare pass rate and mean score first, then cost per run
   and p95 wall-clock. Check the violation histogram for *which* rules a
   candidate breaks, and confirm `Infra errors` is zero before trusting a cell.
   Effort labels are vendor-specific (`high` means different compute on each
   side), so compare the usage and latency columns, not the label. OpenAI
   usage is already normalised to the Anthropic columns (`Input` excludes
   cached tokens, `Cache read` is `cached_tokens`, `Cache write` is always 0).

Noise floor: with `n` scenarios and `r` repeats a pass rate has a standard
error of roughly `1/sqrt(n·r)`. One fixture at three repeats cannot distinguish
anything smaller than about a 50-point swing; twenty scenarios at three repeats
gets you to roughly ±13 points. The three-fixture set in this directory is a
regression gate (with the tolerance in [The rule](#3-the-rule) sized for
exactly that noise), **not** enough evidence to decide a model or effort
change.
Before deciding, seed 20–30 scenarios from exported real sessions
(`scripts/export_sessions.py` -> `sessions_export.md`), each with its own
fixture objects, required tool evidence, and forbidden claims, and extend
`synthetic_tools.py` where the dispatcher lacks the shapes those sessions use.

## Running a comparison from GitHub

`.github/workflows/model-comparison.yml` runs the same matrix on demand so
nobody needs the eval key locally. Actions → **Model comparison** → *Run
workflow* and fill in:

| Input | Default | Passed as |
|---|---|---|
| `models` | `anthropic:claude-sonnet-5,anthropic:claude-sonnet-4-6` (production first) | `--models`; entries may be `<provider>:<model>`, e.g. `anthropic:claude-sonnet-4-6,openai:gpt-5.6-terra` for a head-to-head |
| `efforts` | `medium,high` | `--efforts` |
| `repeats` | `3` | `--repeats` |
| `scenarios` | empty (every fixture) | `--scenarios` only when non-empty |
| `provider` | `anthropic` | `--provider`; only applies to bare (unqualified) model ids |
| `concurrency` | `2` | `--concurrency` |

Or from the CLI:

```bash
gh workflow run model-comparison.yml --ref <branch> \
  -f models='anthropic:claude-sonnet-4-6,openai:gpt-5.6-terra' \
  -f efforts='high' -f repeats='3' -f concurrency='2'
gh run list --workflow model-comparison.yml --limit 1
gh run download <run-id> -n model-comparison-results -D comparison
```

The job installs the runtime dependency set from `uv.lock`
(`uv sync --locked --no-dev`; the `openai` SDK is part of it), sets the same
dummy AWS variables as the PR eval job, exposes both evaluation-only secrets
(`MODEL_EVAL_ANTHROPIC_API_KEY`, `MODEL_EVAL_OPENAI_API_KEY`) and requires at
least one of them, and runs
`python -m evals.run … --no-fail-on-violation --output-dir eval-results`, so
behavioral failures never fail the run — the point is to compare cells, not to
gate. A provider named in `models` whose key is missing has its cells recorded
as infrastructure errors (the step exits 1 with `INFRASTRUCTURE ERROR`, the
other provider's cells are still scored and uploaded). `eval-results/summary.md` is appended to the workflow run summary and
the whole `eval-results/` tree (per-run traces, `summary.json`) is uploaded as
the `model-comparison-results` artifact for 14 days. The workflow has
`permissions: contents: read`, a 90-minute timeout, and a `model-comparison`
concurrency group so two comparisons never run at once (they would skew each
other's latency numbers and share one rate limit).

With the defaults and one fixture that is 2 × 2 × 3 = 12 conversations; each
additional fixture adds 12 more. Read the result with the noise-floor caveat
above.

## GitHub setup

1. Add `MODEL_EVAL_ANTHROPIC_API_KEY` as a repository Actions secret. Give the
   key its own Anthropic workspace/project, low spend ceiling, and narrow rate
   limit. `MODEL_EVAL_OPENAI_API_KEY` (an evaluation-only OpenAI project key,
   same constraints) enables `openai:` cells in the **Model comparison**
   workflow; it is not read by the PR gate and is not the production app
   secret.
2. Require the `Conversation evaluations` status check in `main` branch
   protection. It is skipped (which counts as passing) on PRs that cannot
   change model behavior and on Dependabot PRs; when it runs it applies the
   baseline-relative rule above.
3. Keep the workflow on the normal `pull_request` event. Never change it to
   `pull_request_target` while executing PR code.

GitHub does not expose repository secrets to fork PRs. This repository therefore
needs trusted same-repository PR branches for the required check. If untrusted
forks are introduced, use a maintainer-created branch or an external trusted
evaluation service rather than exposing the evaluation key.

### Dependabot PRs

Dependabot-triggered `pull_request` runs also receive no repository secrets, so
`MODEL_EVAL_ANTHROPIC_API_KEY` is empty and the "Require dedicated evaluation
key" step can never pass. `pr-check.yml` therefore skips the
`Conversation evaluations` job when `github.actor == 'dependabot[bot]'`; a
skipped required check counts as passing for branch protection, so a
Dependabot PR is gated only by `Unit tests`, `Verify Docker build`,
`Validate Terraform`, and the Cycode scans.

Consequence: a Dependabot bump that can change model behaviour must still be
evaluated with the key before it is merged. That is any bump of `anthropic` (or
a transitive dependency of it such as `httpx`/`pydantic`), and any change to
`prompts/`, `tools/`, or model configuration. Do one of:

- push the same change from a maintainer branch (the eval runs on that PR), or
- run `pr-check.yml` for the Dependabot branch from a maintainer context, e.g.
  `gh workflow run "PR Check" --ref <dependabot-branch>` is not possible for a
  `pull_request`-only workflow, so use the manual **Model comparison** workflow
  instead: `gh workflow run "Model comparison" --ref <dependabot-branch>
  -f models=<production model> -f efforts=high -f repeats=1`, and read the
  `summary.md` it publishes. A `workflow_dispatch` run started by a maintainer
  has the secrets even though the branch was created by Dependabot.

Routine bumps of unrelated packages (`boto3`, `requests`, `sqlglot`, test
tooling, the Lambda base-image digest) do not need the eval; the unit suite
covers them. For maintainer-pushed PRs the `Evaluation scope` job makes the
same distinction automatically: `pyproject.toml`/`uv.lock` changes only run
the gate when the `anthropic` or `openai` pin differs.

Transport-level failures are retried by the Anthropic SDK and, for the whole
conversation, by `evals.run` (see above); if they still fail the check is
labelled an infrastructure error (exit 2). Behavioral failures are not
retried: an unresolved guard rejection, bad tool call, unsupported fixture
request, budget overrun (at production's own `MAX_ITERATIONS`), or hard
scoring violation counts against the scenario's pass rate and violation codes,
and the check fails only when the regression rule says the scenario got worse
than the committed baseline. Every run leaves its sanitized trace and the
verdict as a 14-day workflow artifact. Soft (content-coverage) violations
appear in the same summary and artifact without affecting the verdict.
