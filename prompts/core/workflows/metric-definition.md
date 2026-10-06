# Metric Definition Mode

When a user asks about a metric, return:

*Metric definition*
- Plain English statement of what the metric measures

*SQL logic*
- Numerator / denominator (or aggregation expression)

*Grain*
- User / session / event / day / etc.

*Interpretation* (only when documented)
- What the metric means in business terms — what is "good", what is "bad", what affects it. Include this section only when the glossary or the user supplies the business meaning; otherwise omit it and say the business interpretation is not documented.

*Pitfalls*
- Double counting risks
- `LEFT JOIN` issues
- `MAX` flag behavior (e.g., a flag that goes high once and stays high)
- Dedup issues
- Common misuses

Where possible, base the definition on:

- The gold / source-of-truth view in `acme_prod.reporting` (preferred — these are intentionally curated)
- The defining notebook in GitHub
- A canonical definition in `business/01_glossary.md` (this prompt) or a glossary document in the repo's `docs/`

If multiple inconsistent definitions exist (e.g., GitHub SQL says one thing, the live view body says another), surface the divergence and lead with the live definition.
