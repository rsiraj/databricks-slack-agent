# Diff Mode (code change comparison)

When asked to compare versions of a table's build logic, a view's definition, or a job's configuration over time.

## Step 1: Identify the object

Determine table vs. view vs. job — see the `Object-type verification` section in `business/00_repo_context.md`. The right diff source depends on what type it is.

## Step 2: Retrieve both versions

- *Table* (defined in GitHub): use `github_compare_commits` with the relevant commit SHAs, or fetch two specific commits' file contents and compare.
- *View*: query `acme_prod.acme_history.sql_view_backup` for the view's last two history rows and compare their `view_definition` values. See `business/00_repo_context.md` for the table's schema (the view's location is stored as separate `view_catalog` / `view_schema` / `view_name` columns).
- *Bundle-managed job*: search `resources/jobs/` and use `github_get_commits` / `github_compare_commits` on the matching resource YAML for the exact reviewed configuration diff, including branch changes. If the user specifically wants Databricks-side snapshots, resolve the current ID from `acme_prod.acme_history.databricks_workspace_jobs` using the exact `[DAB]` YAML name and latest-row-per-ID rules, then compare `job_settings_json`. Never mix in an unprefixed or `PAUSED_USE_BUNDLE_VERSION_*` predecessor.

## Step 3: Identify differences

Look for changes in:

- Filters (`WHERE`, `HAVING`)
- Joins (type, keys)
- Aggregations
- `CASE` logic
- Deduplication logic (`ROW_NUMBER`, `DISTINCT`)
- Added or removed columns / conditions
- For jobs: schedule, notebook task path, parameters

## Step 4: Output

*What changed*:
- Exact logic differences

*Impact*:
- How results would be affected (row counts, column values, grain)

*Risk*:
- Potential downstream impact on dependent tables / metrics / dashboards

## Rules

- Only describe differences supported by actual diff data.
- Do not hallucinate diffs.
- If only one version exists in history (e.g., the object was never modified after creation), say so — don't manufacture a comparison.
