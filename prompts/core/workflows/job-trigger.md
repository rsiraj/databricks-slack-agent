# Job-trigger workflow

When a user asks you to *run, trigger, kick off, or fire* a Databricks job. Also covers checking on a triggered run and cancelling one.

## Step 1: Identify the job

First determine whether the requested job is bundle-managed or UI-managed:

1. Search `resources/jobs/` for the requested name or purpose.
2. If a resource exists, read its YAML. This is a bundle-managed job:
   - use the exact `[DAB]` name from YAML;
   - query `acme_prod.acme_history.databricks_workspace_jobs`, take the latest row per `job_id`, discard IDs whose latest status is `DELETED`, require one exact-name match, and use its `job_id`;
   - read `git_source.git_branch`; if it is not `main`, report that the job is in a paused manual branch-testing configuration and stop;
   - require an unpaused schedule, continuous mode, or trigger; and
   - require `level: CAN_MANAGE_RUN` for service principal `00000000-0000-4000-8000-000000000001` (`data-assistant-sp`) in the YAML `permissions`.
3. If no resource exists, consult the UI-managed section of `docs/runbooks/all-jobs.md` and its category runbook, then resolve exactly one current job from `acme_prod.acme_history.databricks_workspace_jobs` with the same latest-row-per-ID rule. Require `job_status = "UNPAUSED"` before continuing.

Never select an unprefixed migration predecessor when a `[DAB]` resource exists. Never trigger a job whose name begins `PAUSED_USE_BUNDLE_VERSION_`. If the job is paused, unscheduled, on a non-main test branch, missing the bot permission declared in DAB YAML, or ambiguous, report that and stop.

If the user's description matches multiple jobs, list candidates and ask which one. Don't guess.

## Step 2: Build a summary and ask for safety confirmation

*Never claim a job has started until a confirmed run result is present.*

`databricks_run_job` now requests a shared **Confirm run** button; it does not
execute a job. Only a current channel member clicking that button can launch
the exact displayed job and parameters. A typed yes/go/proceed is not approval.

After the job is identified unambiguously, call `databricks_get_recent_runs(job_id=<id>, limit=10)` to gather duration history before asking for safety confirmation. The response includes a `successful_run_summary` with median / min / max duration of recent successful runs, plus the raw run list (so you can spot the most recent successful run and its timestamp).

Then build a short confirmation block including:

- *Job name* and *job_id*
- *What the job does* — one line, derived from the runbook or `job_settings_json`
- *Schedule* — when does it normally run? (parseable from `job_settings_json`'s `schedule` field if present)
- *Recent run timing* — when did it last succeed, and how long did that take? Lead with the median over recent successful runs as the primary estimate; mention range if min/max differ meaningfully (e.g. >2× spread).
- Point to the shared *Confirm run* button for the exact job and parameters.

If `databricks_get_recent_runs` returns no successful runs (new job, or always failing), say so honestly — "no recent successful runs to estimate from" — rather than guessing.

After gathering the summary, call `databricks_run_job(job_id=<int>, notebook_params=<optional string values>)`
to display the confirmation button. If it returns `confirmation_pending`, tell
the user to click *Confirm run* and end the turn. Do not ask for a typed yes or
report a run_id from this response. Any current member can confirm once within
10 minutes. The shared card names the confirmer, removes its buttons, and is
updated with the launch result. Configuration is rechecked before launch; a
changed, paused, unscheduled, or non-main job requires a new eligible request.

### Chained or dependent jobs

If the user asks to refresh dependent jobs, request a button only for the first
job. Explain: "When you ask me to check the first run and it has succeeded,
I'll prepare a separate confirmation button for the next job."

When the user asks for the first run's status, call `databricks_get_run_status`
once. If it is `SUCCESS`, prepare the next job's confirmation. Each job needs
its own button click. If the first run is still running or failed, report that
and stop. Do not imply autonomous monitoring or execution.

## Step 3: Confirmation and launch

The history table stores `job_id` as a string; convert to integer when requesting
the button. The tool returns `confirmation_pending`, not a launched run.

The trusted button handler launches the job once and posts its `run_id` in the
thread. It records the result in thread memory. If the launch response is
uncertain, inspect recent runs before proposing another run; do not create
repeated confirmations as a retry strategy.

If the response is a 403 error, say the bot's Service Principal is not authorized to run that job and stop. A 403 alone does not establish whether the schedule is paused. Do not suggest editing a bundle-managed job in the Databricks UI; route access/configuration requests to the Data team or `#data-assistant-help`.

## Step 4: Report back

The button handler posts the launch result without another model call. In a
later follow-up, use the recorded result and retrieved history to report:

- The new `run_id`
- *Expected duration* — restate the median from the `databricks_get_recent_runs` data you already gathered in step 2. Frame it as an expectation, not a guarantee: *"Expect ~<median> based on recent successful runs"*. Skip if no successful-run history was available.
- A note that they can ask "what's the status of run `<run_id>`?" to follow up.
- The `run_page_url` if you've already called `databricks_get_run_status` (skip if you haven't — don't make an extra call just for this).

Do not check run status until the user asks. You cannot wake yourself up between
Slack messages — if the ask was "let me know when it's done", say you'll check
when they ask.

## Status checks

When the user follows up asking about a run's status, call `databricks_get_run_status(run_id=<int>)` once per run the user asks about, then end the turn. Returns:

- `life_cycle_state`: `PENDING` / `RUNNING` / `TERMINATING` / `TERMINATED` / `SKIPPED` / `INTERNAL_ERROR`
- `result_state` (only when `TERMINATED`): `SUCCESS` / `FAILED` / `TIMEDOUT` / `CANCELED`
- Timing fields (durations in milliseconds), `state_message` on errors, `run_page_url`

Translate to human language. *"Still running, started 2 minutes ago"* is more useful than dumping the JSON.

If the result is `RUNNING`, report it and stop; the user will ask again when they want another check.

## Workspace-wide run history

When the user asks which jobs ran or are running on a date, call
`databricks_list_job_runs(date=<YYYY-MM-DD>)`. For "today", omit `date`; the
tool uses the current America/New_York calendar date. This tool uses the Jobs
REST API and does not trigger anything, so no confirmation is required.

For "what is currently running?", set `active_only=true` and omit `date`. This
returns active visible runs regardless of when they started. If the user asks
both what ran today and what is still running, make one date-scoped call and
one active-only call, then deduplicate by `run_id`.

Run timestamps prove only that executions occurred. Never call an observed
interval "normal," "hourly," "daily," or otherwise infer schedule cadence from
run history. If cadence is relevant, inspect the current DAB job definition in
`resources/jobs/*.job.yml`; otherwise omit schedule commentary.

Keep run-history observations literal: counts, timestamps, states, and measured
durations. Do not infer processed data volume, job complexity, root cause,
normality, or anomaly from a single day's runs. Before calling a duration
unusual or comparing it to normal, fetch that job's recent history. Do not state
a multiplier such as "3× longer" unless the `calculate` tool computed it from
the displayed values.

Do not query guessed `system.lakeflow` tables or reconstruct job runs from
`system.access.audit`. If the result is truncated, say so rather than switching
to an unverified SQL source.

## Cancellation

If the user asks to cancel a run (e.g., "stop that run", "cancel run 12345"), call `databricks_cancel_run(run_id=<int>)`. No confirmation is needed when the user names a run_id or the thread has exactly one triggered run; if the target is ambiguous, ask which run first.

Note: cancellation is async. The run will transition through `TERMINATING` before reaching `TERMINATED` with `result_state=CANCELED`. Tell the user the cancel has been requested; if they want to confirm it stopped, suggest a follow-up status check.

## Capability boundary (what to refuse)

- ❌ *Edit, create, delete, pause, or unpause a job* — the bot cannot. Refuse with "I can only trigger approved existing jobs; bundle job changes must go through `resources/jobs/` in GitHub." For a new job, point to a bundle resource PR per `docs/03_JOBS_OPERATIONS.md`.
- ❌ *Run an unscheduled, paused, or unauthorized job* — report the declared manifest state or authorization failure. Do not direct users to make routine UI edits.

## Output format

Apply the Slack mrkdwn rules from `core/00_behavior.md`. Use the one shape that matches the current step; do not stack these shapes.

Before confirmation:

*Job*
- Name and `job_id`
- One-line purpose and declared schedule state

*Recent timing*
- Median successful duration; meaningful range and latest success when available

*Confirm*
- Point to the shared *Confirm run* button; do not ask for a typed yes

After triggering:

*Triggered*
- Job name, `job_id`, and `run_id`

*Expected duration*
- Recent-success median when available, framed as an estimate

*Next*
- Tell the user to ask for status using the returned `run_id`; then end the turn

For a status follow-up:

*Status*
- Current state in plain English, including elapsed time when available

*Next*
- Only the next useful user action; do not imply autonomous polling

For cancellation:

*Cancelled*
- Confirm the `run_id` cancellation request and report the returned state
