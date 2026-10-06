# Explaining tables & logic

## Step 0: Verify object type

Resolve the object with `databricks_find_objects` (returns its type, or no match) and read columns and comments with `databricks_describe_object`. Run `SHOW CREATE TABLE` only when you need a view's body. See `business/00_repo_context.md` → "Object-type verification" for the dispatch.

## Step 1: Read live documentation

Explain from the table comment and the column names, types, and comments `databricks_describe_object` returned; query `<catalog>.information_schema.tables` / `<catalog>.information_schema.columns` only for a filtered metadata question the typed tool cannot express.

- Treat Unity Catalog comments as the primary documentation for purpose, grain, and column semantics.
- For very wide event tables, query columns relevant to the user's question instead of relying on an unfiltered result, because tool output is sampled.
- Validate comments against the live definition and GitHub build logic. If they disagree, explain the conflict and prefer the live definition for implemented behavior.
- Build a persisted-column inventory from the live metadata before writing validation SQL. A name that appears only inside a notebook CTE, window expression, filter, or MERGE source is a build-time alias, not a queryable target column. Never reference such a name against the final table unless it also appears in the live persisted-column inventory.
- Before submitting validation SQL, compare every target-table column reference with that inventory. This includes fields used only in `WHERE`, `CASE`, `JOIN`, `GROUP BY`, and `ORDER BY`, not just the selected output columns.
- Notebook code proves transformation logic, not job schedule or runtime. Do not state cadence, schedule, or typical duration unless you inspected the current `resources/jobs/*.job.yml` configuration or live job evidence in this turn.
- When the user explicitly asks for lineage, retrieve it before optional profiling,
  sample queries, history, or job discovery. Describe `system.access.table_lineage` first
  in the current turn, then query it for the requested object's upstream and downstream
  relationships.
- Do not spend tool calls locating job configuration merely to explain a table.
  Only inspect a refresh job, schedule, or runtime when the user explicitly asks
  for that operational detail; table build logic comes from the defining SQL.

## Table-level explanation

If the object is a *table*:

1. Identify the catalog.
2. If the defining notebook / SQL exists in GitHub:
   - Use GitHub + Databricks query results as evidence
   - Lead with the GitHub-backed logic, supported by sample query results
3. Otherwise:
   - Use SQL query results only
   - Explicitly state that the build logic is not confirmed from code

If the object is a *view*:

- The `CREATE VIEW` body returned by `SHOW CREATE TABLE` is the source of truth — explain that.
- If GitHub also has SQL with the same name, mention the divergence and prefer the live definition.

## Output structure

- *What it is* — one-line summary in plain English
- *How it's built* (if known) — source tables, key transformations
- *Key logic* — joins, filters, deduplication, aggregations
- *Upstream sources* — list with their catalog/schema
- *Unknowns* — anything you couldn't verify from code or query

Always label evidence:
- *(from GitHub)* — confirmed from code
- *(from query)* — confirmed from a Databricks query
- *(from Unity Catalog comment)* — documented in live table or column metadata
- *(inferred)* — your interpretation, not directly backed
