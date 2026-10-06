# Glossary (example)

> **Example glossary.** These are fictional Acme Shop terms. Replace this file
> with your organization's metric and terminology definitions. Keep each entry
> to a definition, a grain, and the object that implements it so the bot can
> anchor SQL to it (see `core/workflows/sql-generation.md` and
> `core/workflows/metric-definition.md`).

| Term | Definition | Grain | Implemented in |
|---|---|---|---|
| Signup | First account-creation event for a browser identity that passes internal-traffic and bot filters. | One row per `_anonymousId` | `acme_prod.reporting.signups` |
| Autofill | A signup where the visitor accepted saved details at checkout (`autofill_ind`). Not the same as `identification_method = 'anonymous'`. | Signup | `acme_prod.reporting.signups.autofill_ind` |
| Pre-order signup | A signup that happens before the customer's first order (`pre_order_ind = 1` when `order_date_differential < 0`). Do not reverse the sign from the column name; read the defining expression. | Signup | `acme_prod.reporting.signups.pre_order_ind` |
| Cross-state shipment | Billing state differs from shipping state. | Signup | `acme_prod.reporting.signups.cross_state_ship_ind` |
| Internal traffic | Sessions from office IPs or bots; excluded from gold tables, present in raw event tables. | Session | `acme_prod.reporting.signups` (filter) |

## Known data events

Dated launches that change how a metric behaves inside a comparison window belong
here, so the bot can flag them before attributing a change to something else.

- **2026-08-03 — `identification_method = 'anonymous'` begins** (fictional example). Rows before this date never carry that value, and it is **not** equivalent to non-autofill signups. Flag it for any trend, year-to-date, or month-over-month comparison that crosses the date.
- **2026-09 — storefront redesign launch** (fictional example). Weekly signups fell roughly ~-36% the week after launch, then recovered; treat the dip as a launch effect to confirm, not a data defect.

## Masked columns (example)

The bot's service principal sees some protected columns masked. A masked value (for example a `dob` placeholder) proves only that the queried protected column is masked; it says nothing about separately persisted or derived columns in the same table, such as `age` or `age_range`. Check those columns directly before declaring them unusable.
