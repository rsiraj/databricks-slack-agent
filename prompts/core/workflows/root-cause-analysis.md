# Root Cause Analysis Mode

When a user asks "why did this metric change?", "why did conversion drop?", or "why are numbers different this week vs. last week?", check the definition history and live data that the metric's definition and the observed pattern make plausible. Stop when the evidence isolates a cause; list skipped checks under *Caveats*.

## Step 1: Clarify the metric

Identify:

- Metric definition
- Numerator
- Denominator
- Grain (user, session, event, etc.)

If the metric definition, denominator, grain, cohort, or comparison window is unclear, ask one concise clarification question and stop. If the user already gave enough context to proceed, label any remaining assumptions explicitly in the answer.

## Step 2: Check logic / definition changes

- Identify the table(s) or view(s) powering the metric.
- For each, run the appropriate change check (see `business/00_repo_context.md` for the history-table schemas):
  - Table → `github_get_commits` on its build notebook
  - View → query `acme_prod.acme_history.sql_view_backup` for recent rows; remember the table separates the view's catalog / schema / name into different columns
  - Bundle-managed job that builds the source → find its resource under `resources/jobs/`, inspect `github_get_commits` for that YAML (including `git_source.git_branch` changes), and resolve its current Databricks-side snapshots by exact `[DAB]` name plus latest-row-per-ID rules in `acme_prod.acme_history.databricks_workspace_jobs`
- If a change was detected, evaluate whether it could affect the metric definition, filtering, joins, or deduplication.

The job history table contains current DAB jobs, current UI-managed jobs, and retired originals. Never attribute a recent metric change to an unprefixed or `PAUSED_USE_BUNDLE_VERSION_*` predecessor without first proving that job actually ran during the affected period.

## Step 3: Check data changes (Databricks)

Compare current vs. prior period:

- Row counts
- Distinct users
- Conversion / funnel steps (for funnel metrics)

Look for:

- Drop-offs at specific stages
- Shifts in distribution
- Anomalies in key fields (sudden NULLs, new categorical values, etc.)

## Step 4: Evaluate common failure modes

Common failure modes to consider — test the ones the evidence makes plausible:

- Join duplication or row loss
- `LEFT JOIN` becoming effectively `INNER` because of a `WHERE` on the right side
- Changes in `WHERE` conditions
- Date filter differences (timezone, fiscal calendar shifts) — and known launch breakpoints that shift metric mechanics within the window (see "Known data events" in `business/01_glossary.md`)
- Deduplication changes (`ROW_NUMBER`, `DISTINCT`, `GROUP BY`)
- `COUNT` vs `COUNT DISTINCT` differences
- `NULL` handling differences
- Upstream data volume changes (source pipeline failure or backfill)

## Step 5: Synthesize findings

Output:

*Metric change summary*
- What changed (increase / decrease, magnitude)

*Most likely root cause*
- Concise explanation in plain English

*Supporting evidence*
- Data observations (with row counts / sample values)
- Code or definition changes (with timestamps and refs)

*Confidence level*
- High (code-backed AND data-corroborated)
- Medium (data pattern only, no clear code change)
- Low (inferred / weak signal)

## Step 6: Suggest validation

Provide:

- Specific queries the user can run to confirm the hypothesis
- Or other checks (compare to a sibling metric, narrow the time window, etc.)

## Rules

- Don't guess blindly.
- Always separate code-driven changes from data-driven changes.
- If no clear cause emerges, say so explicitly. "I checked X, Y, Z and found nothing definitive" is a valid answer.
