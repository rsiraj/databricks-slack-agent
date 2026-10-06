# Query Comparison Mode

When a user gives you two SQL queries and asks why they produce different results.

## Step 1: Compare structure

Check for differences in:

- Source tables (and whether they're tables or views — see `Object-type verification` in `business/00_repo_context.md`)
- Joins, join types, join keys
- Filters, date ranges (watch for windows spanning known launch breakpoints — see "Known data events" in `business/01_glossary.md`)
- Grouping level
- Deduplication logic
- Aggregations
- `COUNT` vs `COUNT DISTINCT`
- `WHERE` vs `HAVING`
- `CASE` logic
- `NULL` handling

## Step 2: Identify grain

For each query, explain:

- Row-level grain before aggregation
- Final output grain
- Whether one query is more duplicated or more filtered than the other
- Duplication risk (e.g., a one-to-many join on the right side without dedup)

## Step 3: Explain the discrepancy drivers

Call out the most likely reasons results differ. Common drivers:

- Duplicate rows introduced by a join
- Different denominator definitions
- Different date filters (timezone, fiscal calendar)
- User-level vs event-level counting
- Filtered NULLs
- `LEFT JOIN` becoming effectively `INNER` due to a `WHERE` on the right side
- Inconsistent dedup logic
- Grain mismatch
- One query reading a table while the other reads a view layered on it (with extra filters)

## Step 4: Output format

*Query A summary*
- What it's doing in plain English

*Query B summary*
- Same

*Key differences*
- Exact logic differences that affect results

*Most likely reason results differ*
- Concise explanation in plain English, ranked by confidence

*How to validate*
- Suggested checks or narrower test queries

## Step 5: Evidence handling

- If code-backed logic exists in GitHub for either query's source object, use it.
- If comparing ad hoc user queries, rely on the literal text and Databricks validation.
- Don't guess silently. Label uncertainty clearly.
