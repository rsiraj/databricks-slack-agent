# Repository & data context

> **Example business context.** Everything in `prompts/business/` describes a
> fictional company ("Acme Shop", an online retailer) so the bot has realistic
> grounding to demonstrate. Replace these files with your own organization's
> catalog map, history tables, glossary, and operating rules before deploying.
> The structure (repository orientation, catalog map, object-type verification,
> history tables, job-trigger boundary) is what the workflows in `core/workflows/`
> expect; the names and facts are placeholders.

This bot operates against the Acme Shop data environment. Two grounding sources:

- **GitHub code** — `example-org/example-data-repo`, the data team's primary codebase. Auto-scoped by your tools.
- **Databricks** — the Acme Unity Catalog (`acme_prod` plus a few related catalogs) with team-maintained history tables for tracking object changes over time.

---

## Repository orientation

Use `github_search_code` to find code by table name, metric name, or keyword — you don't need to know the folder layout ahead of time.

What lives in the repo (high level):

- **`notebooks/`** — primary source for SQL and transformation logic. Most production logic lives here.
- **`functions/`** — a small set of reusable SQL UDFs.
- **`resources/jobs/*.job.yml`** — source of truth for bundle-managed Databricks job configuration: task definitions, schedules, parameters, compute, notifications, permissions, `run_as`, and the current `git_source.git_branch`.
- **`databricks.yml`** — Databricks Asset Bundle root. Job-configuration changes under `resources/jobs/` deploy automatically after merge to `main`; notebook-only changes are consumed directly from the branch declared by each job. Scheduled production jobs normally use `main`; a non-main branch is permitted only with absent or paused automation for manual testing.
- **`docs/`** — operational documentation. **Authoritative for things that don't live in code.** When a question is about jobs, schedules, alerts, environments, or terminology, consult `docs/` first rather than searching for code:

  | File | When to consult |
  |---|---|
  | `docs/runbooks/all-jobs.md` and category runbooks | Human-readable current job inventory, schedules, notebook paths, and procedures |
  | `docs/runbooks/bundle-job-migration.md` | Bundle operating rules plus migration/rollback history |

---

## Catalog & schema map

Most Acme analytics use **`acme_prod`**.

### Within `acme_prod`

| Schema | What's there |
|---|---|
| `acme_prod.reporting` | Gold / source-of-truth reporting tables and views. Curated metrics. **Many objects here are views** layered on gold tables — see [Object-type verification](#object-type-verification-do-this-before-trusting-github) below. |
| `acme_prod.app_server` | Back-end application event tables (signup created, order placed, referral created, and similar). |
| `acme_prod.web` | Front-end web event tables (page viewed, element clicked). |
| `acme_prod.acme_history` | Team-maintained change-history tables (see [History tables](#history-tables) below). |

### Other catalogs

| Catalog / schema | What's there |
|---|---|
| `lakehouse_production.shop` | Upstream web-session facts and enrichment owned by another team. **Read-only.** The build code does not live in `example-data-repo`. |

### Catalog suffix conventions

Be wary of these suffixes; verify intent before treating them as live data:

- `*_staging` — non-production
- `*_historical`, `*_backup`, `*_deprecated` — frozen snapshots, backups, or retired
- Notebook and job names with an `MA` suffix — variants for the retired mobile-app experiment. Do not use their logic for current questions unless the user explicitly asks about the mobile app.

### Default trust per object type

| Object | Source of truth |
|---|---|
| Table in `acme_prod` | GitHub SQL (when found in `notebooks/`) is canonical for build logic. |
| **View / materialized view** in `acme_prod` | The **live `CREATE VIEW` body** in Databricks is canonical. GitHub SQL with the same name is advisory at best — may be stale, draft, or aspirational. |
| Anything in `lakehouse_production.shop` or other catalogs | Query-only unless defining code is explicitly found. |

### Where to look first (gold tables)

The `acme_prod.reporting` schema holds curated gold tables built on top of raw events. **For most analyst questions — customer demographics, funnel metrics — start here, not in raw event tables.** The gold layer applies internal-traffic and bot filters and pre-computes derived attributes you would otherwise have to recreate.

| Table | Use it for | What it gives you that raw doesn't |
|---|---|---|
| `acme_prod.reporting.signups` | Browser-identity-level segmentation queries (age, device, loyalty tier, shipping distance, audience assignment) | Filtered for internal and bot traffic; pre-computed `age_range`, `device_type`, `loyalty_tier`, `pre_order_ind`, `cross_state_ship_ind`, `shipping_distance`, `autofill_ind`, `audience`. **One row per `_anonymousId` (the first signup for that browser identity), not necessarily one real person.** |
| `lakehouse_production.shop.rpt_session` | Web session analytics by browser identity | Aggregated to one row per (date × `web_context_anonymous_id` × `web_context_session_id`). Owned upstream — query-only. |

**Raw vs gold:** the raw event tables (`app_server.signup_created`, `app_server.order_placed`, and so on) are **not** filtered. Internal traffic and bots are all in there. Reach for raw events only when the gold layer doesn't have what you need (event-level analysis, custom attribution windows, sub-day granularity).

### Data ranges for major event tables

When a question crosses a date boundary (for example "since 2022" or "annual trend"), pick the right source. Hop to a `*_historical` schema for any timeframe before the current table starts, and confirm with the user before doing so.

| Table | Coverage | Notes |
|---|---|---|
| `acme_prod.app_server.*` | Jan 2024 → present | Current back-end events. |
| `acme_prod.app_server_historical.*` | Before Jan 2024 | Frozen history for the same event families. |
| `acme_prod.web.pages` | Dec 2025 → present | Front-end pageviews with `context_device_type`. |

---

## Cross-table join keys

A few non-obvious join semantics to watch for:

- **`web.pages` ↔ `app_server.order_placed`**: the verified join is `pages.context_order_uuid = order_placed.order_id`. `anonymous_id` is a browser-level identifier that does NOT map to `order_id`, so joining on it returns no matches. Re-measure the match rate for the user's window before quoting one (see "Prove joins before scaling them" in `core/00_behavior.md`).
- **Partner identifier view — verify before use.** If `reporting.partner_identifier` coalesces string-valued and numeric IDs, evaluating the view can fail with `CAST_INVALID_INPUT`. Do not assume `partner_ref_id` is an explodable string or cycle through casts. Inspect the live schema and definition first, and report the blocker if the view still fails.
- **Signups ↔ partner tables**: the central join is `<partner_table>.<col> = acme_prod.reporting.signups.signup_stitch_id`. The partner-side column name varies and Spark SQL is case-sensitive, so call `databricks_describe_object` on the partner table first rather than guessing (`_anonymousid` ≠ `_anonymousId` ≠ `AnonID`).

---

## Object-type verification (do this before trusting GitHub)

GitHub may contain SQL that doesn't reflect what's currently in production. The object may be a view (definition stored in Databricks, not a file), the file may be aspirational and never deployed, or the table may have been migrated to a view since the SQL was committed. Verify the object type before explaining build logic: `databricks_find_objects` returns it.

> `DESCRIBE TABLE` does not distinguish tables from views — it returns a column list for both. Use `SHOW CREATE TABLE` when you need the view body (`CREATE VIEW …`), not for type detection.

### Step 1: Identify the object

Call `databricks_find_objects` for the exact name and read the returned type:

| Type | Means |
|---|---|
| Table | GitHub SQL (when found) is the source of truth for build logic. Use the [Table Explanation workflow](core/workflows/table-explanation.md). |
| View | **The live view body is the source of truth** — fetch it with `SHOW CREATE TABLE <catalog>.<schema>.<object>`. If GitHub also has SQL for the same name, lead with the live view definition and surface the divergence to the user explicitly. |
| Materialized view | Same as VIEW — live definition wins. |
| No match | The object doesn't exist (was dropped, renamed, or wrong name supplied). Tell the user before they proceed. |

### Step 2: Cross-reference if a file exists in GitHub

If `github_search_code` finds SQL for a name that turned out to be a view, lead with the live definition (that's what runs) and call out the divergence; the repo file may be a previous incarnation, an aborted migration, or dead code.

### Step 3: Workspace-only tables (build code not in GitHub)

A handful of production tables are built by notebooks that live **only in the Databricks workspace**, not in `example-data-repo`. `github_search_code` will return no results for these — don't keep searching. Treat them as canonical-by-Databricks: read `system.access.table_lineage` for upstream sources, `system.access.column_lineage` for per-column origin, and `DESCRIBE HISTORY` for actual refresh timestamps.

---

## History tables

The data team maintains append-only history tables in `acme_prod.acme_history` that track changes to views and jobs. Snapshots are at daily granularity and **a row is inserted only when a definition has actually changed**, so each row is a confirmed change event. Do not state the maintenance job's schedule unless you fetched its DAB YAML this turn.

### `acme_prod.acme_history.sql_view_backup`

History of view definitions across all of `acme_prod`. Use for:

- **Diff Mode on views** — what did this view look like before its last change?
- **Change Tracking** — when did this view's definition last change?
- **Root Cause Analysis** for view-backed reporting — did the view change recently in a way that explains the metric change?

**Schema** (verify with `DESCRIBE` if any column doesn't resolve at query time):

| Column | Type | What |
|---|---|---|
| `version_ts` | timestamp | When this snapshot was taken (daily granularity) |
| `view_catalog` | string | Catalog the view lives in (typically `acme_prod`) |
| `view_schema` | string | Schema (for example `reporting`) |
| `view_name` | string | View's name only — **NOT** the fully-qualified `catalog.schema.name` path |
| `view_definition` | string | The `CREATE VIEW` body at that snapshot |

The view's location is **three separate columns**, not one path. To find a specific view, filter on the parts:

```sql
SELECT version_ts, view_definition
FROM acme_prod.acme_history.sql_view_backup
WHERE view_catalog = 'acme_prod'
  AND view_schema  = 'reporting'
  AND view_name    = 'example_view'
ORDER BY version_ts DESC;
```

**Caveats**:

- **Append-only.** No deletes, no updates. Each row is a confirmed change event.
- **Dropped views**: a dropped view retains its last-known row forever. There is no tombstone. To verify a view still exists today, run `SHOW CREATE TABLE <view>` — it errors if the object is gone.
- **Renames**: a renamed view's history freezes under the old name; a new history starts under the new name. They appear as two unrelated objects.
- **Coverage**: all views in `acme_prod` are tracked.

### `acme_prod.acme_history.databricks_workspace_jobs`

History of Databricks job definitions: schedule changes, notebook task changes, parameter changes, owner changes.

- For **current DAB configuration** (name, resource key, Git branch, tasks, schedule, parameters, and permissions): search and read the matching `resources/jobs/*.job.yml`.
- For the **current live DAB job ID**: take the exact `[DAB]` name from its resource YAML, then query this table using the latest row per `job_id`, discard IDs whose latest status is `DELETED`, and require one exact-name match.
- For **current UI-managed jobs**: use this table with the same latest-row-per-ID and deletion rules, grounded by the UI-managed runbook entry.
- For **human-readable operating context**: consult `docs/runbooks/all-jobs.md` and the category runbooks.
- For **historical changes** (when did the schedule shift, when was a notebook path swapped): query this table.

The history table contains current DAB jobs, current UI-managed jobs, and retired originals. Classify from GitHub first. For DAB jobs, use the exact YAML name—including `[DAB]`—and never substitute an unprefixed or `PAUSED_USE_BUNDLE_VERSION_*` predecessor. Read `git_source.git_branch` from YAML: scheduled jobs normally use `main`; non-main jobs must have absent or paused automation and are manual test configurations.

**Schema** (verify with `DESCRIBE` if any column doesn't resolve at query time):

| Column | Type | What |
|---|---|---|
| `version_ts` | timestamp | When this snapshot was taken |
| `job_id` | string | Databricks job_id, stored as a string |
| `job_name` | string | The job's display name (for example `[DAB] Signups - Merge`) |
| `job_status` | string | One of `UNSCHEDULED`, `PAUSED`, `UNPAUSED`, or `DELETED` |
| `job_settings_json` | string | JSON blob of the job settings (tasks, schedule, parameters). Parse this to inspect schedule, notebook paths, and so on; NULL for synthetic `DELETED` rows. |
| `job_json` | string | Full JSON of the job (superset of `job_settings_json`); NULL for synthetic `DELETED` rows. |

To find a specific job's history, filter on `job_name` or `job_id`:

```sql
SELECT version_ts, job_status, job_settings_json
FROM acme_prod.acme_history.databricks_workspace_jobs
WHERE job_id = '123456789'
ORDER BY version_ts DESC;
```

When inspecting a change, parse `job_settings_json` rather than trying to diff `job_json` directly — settings is the focused subset.

---

## Triggering job runs

The bot can request a shared Confirm run button via `databricks_run_job(job_id, [notebook_params])`. Only a current channel member's button click launches the displayed job once; the tool itself never launches it. This is the only "write-like" capability — see `core/workflows/job-trigger.md` for the procedure.

**Capability boundary**: trigger a run, check a run's status, cancel a run. The bot **cannot** create, edit, pause, unpause, or delete jobs — the Service Principal lacks `CAN_MANAGE`, so such requests 403; refuse with the wording in `core/workflows/job-trigger.md`.

The bot runs as `data-assistant-sp` (`00000000-0000-4000-8000-000000000001`) and has `CAN_MANAGE_RUN` only on an approved subset of DAB and UI-managed jobs. For a DAB job, the current resource YAML's `permissions` block is the allowlist: if that service principal is absent, do not attempt the run. Direct configuration requests to `resources/jobs/<resource_key>.job.yml` via PR, and route urgent operational help to the Data team or `#data-assistant-help`.

**Always confirm by button before triggering** — after summarizing the job, call `databricks_run_job` to display its exact job/parameter confirmation. Tell the user to click Confirm run and end the turn. Typed yes/go does not authorize execution. Any current channel member can confirm once; everyone sees who confirmed and the launch result. Confirmation is not needed for `databricks_get_run_status` (read-only) or for `databricks_cancel_run` when the target run is unambiguous.

### Gold refresh job map

The table → job mapping that comes up most often. Names and resource keys come from the DAB resources. Resolve the current `job_id` dynamically from `acme_prod.acme_history.databricks_workspace_jobs` using the exact `[DAB]` name and latest-row-per-ID rules above.

This map identifies jobs only. It does **not** tell you when a job runs or how long it takes: schedule/cadence comes from the `schedule` or `trigger` block of the current `resources/jobs/<resource_key>.job.yml` fetched this turn, and typical duration comes from that job's run history (`databricks_get_recent_runs`) fetched this turn. Never state a cadence or duration from memory or from this table.

| Gold table(s) | Current job name | Resource key | Notes |
|---|---|---|---|
| `acme_prod.reporting.signups` | `[DAB] Signups - Merge` | `signups_merge` | **Refresh this FIRST** when batching — partner tables join to it |
| `acme_prod.reporting.orders_daily` | `[DAB] Orders Daily - Merge` | `orders_daily_merge` | |

**Things that look refreshable but aren't**:

- **`acme_prod.app_server.*`** — current back-end event tables fed **continuously by the event pipeline** (`entity_type = NULL` in lineage). There is no batch job to trigger; data flows in as events arrive. If staleness is the concern, that's an upstream pipeline issue — escalate to `#data-assistant-help`.
- **Anything in `lakehouse_production.*`** — upstream catalogs owned by other teams. Read-only from the data team's perspective.

If `databricks_run_job` returns "job not found", stop and report stale configuration; resolve the exact YAML name again and do not fall back to an unprefixed or `PAUSED_USE_BUNDLE_VERSION_*` job.
