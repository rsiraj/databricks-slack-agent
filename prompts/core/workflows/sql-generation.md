# SQL Generation for Metrics

When asked to generate SQL for a metric or analytical question.

## Step 1: Validate inputs

- Identify the correct table(s) — prefer `acme_prod.reporting` gold tables / views when one fits
- Treat live Unity Catalog table and column comments as the primary documentation for an object's purpose, grain, and column semantics. For each table or view used in analytical SQL, inspect its live comments once per conversation before writing the final query, unless that metadata was already retrieved earlier in the thread.
- Use `databricks_find_objects` to resolve an uncertain name; use direct `information_schema` SQL only for focused metadata questions the typed tools cannot express. For very wide event tables, reason from the relevant returned columns instead of requesting raw data samples.
- Use comments as documentation, not as proof of implementation. Reconcile them with the live view definition, GitHub build logic, and observed schema; call out material conflicts rather than silently choosing one.
- Confirm the grain matches the metric definition
- Use GitHub-backed logic when available; for views, base on the live definition
- If the question references a metric defined in `business/01_glossary.md`, anchor to that definition
- If the requested result must match a published reporting table, inspect that table's defining SQL before choosing raw sources. Carry forward its exact source events, dedup key, date column, join path, and filters.

## Step 2: Build the query safely

- Prefer `COUNT(DISTINCT ...)` for user-level metrics
- Respect deduplication logic from the source (`ROW_NUMBER` filters, `GROUP BY`)
- Avoid `SELECT *` — name the columns you need
- Apply appropriate date filters when relevant
- When the year is omitted, apply the current calendar year and state it; do not ask solely for the year. Interpret "last week" as the previous completed Monday–Sunday week, not a rolling seven days.
- Before the first analytical query against any object in a conversation, call `databricks_describe_object` on it and use only the column names it returned. Similarly named tables use different conventions (for example `web_element_text` / `event_date` on one source vs `webelement_text` / `_date` on another), and event-pipeline columns such as `original_timestamp` exist only on raw `app_server.*` event tables, not on curated `reporting.*` tables.
- If a first query errors with `UNRESOLVED_COLUMN`, stop and inspect the schema before trying another guessed variant. If it errors with `MISSING_GROUP_BY`, fix the aggregation shape before running adjacent exploratory queries.
- The tool performs a small, high-confidence syntax preflight. If it rejects `LIMIT` on metadata SQL or a window function directly in `WHERE`, fix that exact shape; do not work around the check with a semantically different query.

## Step 3: Match the business definition

Ensure the SQL aligns with:

- The metric's definition (numerator, denominator, grain)
- The aggregation level the user asked about
- Filters used in the source logic (e.g., `is_internal_ip = false`)

### Strict funnel semantic check (opt-in)

Use this only when the user, glossary, or inspected source logic explicitly defines a funnel where the numerator population must be a subset of the denominator at the same grain. Alias the executed aggregate columns as:

- `funnel_numerator`
- `funnel_denominator`
- `funnel_rate` for a 0–1 ratio, or `funnel_rate_pct` for a 0–100 percentage

Those exact aliases activate a result-level bounds check. If it returns `WARNING`, reconcile grain, deduplication, and subset filters before presenting the number.

The aliases must survive into the **outermost final SELECT and returned column names**. Do not rename them to labels such as `form_starts` or `form_submits`; translate them into human-friendly labels only in the written answer. If `funnel_rate` or `funnel_rate_pct` is returned without both required count aliases, the tool treats the semantic contract as incomplete and returns `WARNING`.

Do not use these aliases or claim a strict subset check for exploratory event sequences, independently counted cohorts, loosely related stages, or any funnel whose subset semantics have not been explicitly defined. The check does not infer business meaning.

## Step 4: Output format

Provide:

*SQL query*
- Clean, runnable, read-only
- Put every query you intend the user to run in a language-labeled `sql` fenced
  code block; use plain fences for illustrative fragments and metadata probes.
  The delivery layer automatically combines the `sql` blocks into a `.sql`
  attachment; do not claim that attachment was uploaded because delivery happens
  after the response is generated.

*Explanation*
- What it calculates
- Grain
- Assumptions

*Warnings* (when relevant)
- Potential double-counting
- Missing filters that could matter
- Data freshness limitations

## Step 5: Safety

The SQL safety rules in `core/00_behavior.md` apply: read-only, `LIMIT` on raw rows, prefer aggregates, and hand the user runnable SQL for full-dataset requests instead of paginating results back.
