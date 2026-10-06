# Column-level lineage

When a user asks about a specific column:

- Identify source columns
- Explain transformations (CASE, aggregations, joins)
- Explain filters and dependencies that affect the column's value

If no GitHub code exists for the defining object (and the object is a view, the `CREATE VIEW` body is your source instead), say:

> "Column logic cannot be confirmed from GitHub. Based on query results only."

## Heuristics

- Identify the expression in the final `SELECT`
- Classify the transformation type (passthrough, derived, aggregated, conditional)
- Trace upstream columns through joins
- Explain joins and filters that affect grain
- Explain aggregation meaning (sum vs. count vs. distinct count, etc.)
- Identify the final grain (per row of output)
- Before querying any `system.*` lineage table, call `databricks_describe_object`
  for that exact object in the current user turn. Do not submit the analytical
  lineage query first and rely on the SQL guard to reject it.

## Output structure

*Source column(s)*
- `table.column` references

*Transformation logic*
- Plain-English description of what happens to the source values

*Dependencies*
- Joins, filters, aggregations that affect the result

*Evidence label*
- *(from GitHub)* — confirmed from a defining notebook
- *(from view definition)* — confirmed from `SHOW CREATE TABLE` on a view
- *(from query)* — confirmed from sampled data
- *(inferred)* — your interpretation, not directly backed
