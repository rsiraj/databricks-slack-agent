# Change Tracking & code history

When asked "when was this updated?", "when did this logic change?", or "what changed recently?":

## Step 1: Identify the file or object

- For *tables*: locate the notebook or SQL file responsible for the build logic via `github_search_code`.
- For *views*: don't rely on GitHub. Query `acme_prod.acme_history.sql_view_backup`. *See `business/00_repo_context.md` for the table's schema and the correct filter shape* — note especially that catalog, schema, and view name are stored in separate columns (don't try to match a fully-qualified path against a single column).
- For *jobs*: search `resources/jobs/` first. A matching resource is bundle-managed; its YAML is current configuration, including the Git branch. If none matches, use the UI-managed runbook entry and resolve the current job from `acme_prod.acme_history.databricks_workspace_jobs`.

## Step 2: Retrieve the change history

- For files: `github_get_commits` on the path. The repo's default branch (`main`) is auto-applied.
- For bundle-managed jobs: use `github_get_commits` on `resources/jobs/<resource_key>.job.yml` for exact reviewed configuration history. Use `github_compare_commits` when the relevant base/head refs are known.
- For views and Databricks-side job snapshots: query the appropriate history table. For jobs, resolve the exact current ID using the latest row per `job_id`, discard deleted IDs, and use the exact `[DAB]` YAML name for bundle jobs. The table also contains retired originals. Each row is a confirmed change event; `acme_history` snapshots are at daily granularity, so the actual Databricks-side change happened *at or before* the row's `version_ts`. Do not state the snapshot job's schedule unless you fetched its DAB YAML this turn.

## Step 3: Output

*Last updated*:
- For files and bundle job configuration: most recent commit timestamp + SHA
- For views and Databricks-side job snapshots: the `version_ts` of the most recent history row

*Change summary*:
- For files and bundle job configuration: derived from commit messages and diffs
- For views and Databricks-side job snapshots: described in terms of what differs between the two most recent rows

*Confidence*:
- High when commit/diff data is concrete
- Medium when only metadata is available

## Rules

- Never guess change history.
- If commit / history data is unavailable, explicitly say so.
- For views and jobs: don't conflate the snapshot date with the actual change date — the change happened *at or before* the row's timestamp, and the snapshots' daily granularity is the limit of what `version_ts` can tell you.
