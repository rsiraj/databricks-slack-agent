# Using the Databricks Slack Agent

The Databricks Slack Agent answers data questions in Slack using live Databricks
metadata and results plus the SQL, notebooks, and job definitions in
the configured GitHub data repository (`GITHUB_REPO`).

## Start a conversation

The bot operates only in approved private Slack channels that include at least
one allowlisted bot admin. If the channel is not approved or the last admin
leaves, the bot leaves the channel; ask the bot's administrators for access. Mention it with your question:

> `@Databricks Slack Agent how many customers signed up before their first order last week, broken out by device?`

The bot replies in a thread. Mention it again on every follow-up, including
inside that thread:

> `@Databricks Slack Agent can you split that by loyalty tier too?`

Keeping follow-ups in the same thread preserves the prior question, SQL, tool
results, and assumptions.

## Questions that fit the bot well

These examples use the fictional Acme Shop data in `prompts/business/`.
They are illustrative, not saved queries or guaranteed metric definitions.

### Analyze a metric

- `@Databricks Slack Agent how many pre-order and post-order signups did we have last week, split by mobile and desktop?`
- `@Databricks Slack Agent show the weekly checkout conversion rate for the last eight weeks.`
- `@Databricks Slack Agent how many unique users viewed the product page yesterday, excluding internal traffic?`
- `@Databricks Slack Agent write the SQL for signups by audience and loyalty tier for the current month.`

Include the time period, desired grain, and important filters when you know
them. The bot will inspect the live schema and repository logic before using
unfamiliar fields or rebuilding a published metric.

### Understand or reconcile data

- `@Databricks Slack Agent why doesn't a count of raw signup_created events match reporting.signups?`
- `@Databricks Slack Agent what does pre_order_ind mean in reporting.signups, and where is it defined?`
- `@Databricks Slack Agent what is the correct join path between web page views and placed orders?`
- `@Databricks Slack Agent trace the checkout conversion metric from its source event through the final reporting view.`

These questions are especially useful when similarly named events have
different grains, identifiers, date fields, or deduplication rules.

### Investigate repository logic

- `@Databricks Slack Agent find the SQL that builds reporting.signups and explain how device_type is assigned.`
- `@Databricks Slack Agent where is the Orders merge defined, and what tables does it update?`
- `@Databricks Slack Agent compare the recent changes to the Signups merge logic and summarize the metric impact.`
- `@Databricks Slack Agent read the full Orders merge file and explain its deduplication logic.`

The bot can search by code content or filename and can read up to 60,000
characters from the selected GitHub file. It reports when a larger file was
truncated and does not treat the omitted remainder as inspected.

### Check or run Databricks jobs

- `@Databricks Slack Agent what jobs ran today, and are any still running?`
- `@Databricks Slack Agent when did the Orders merge last succeed and how long does it usually take?`
- `@Databricks Slack Agent run the Orders merge job.`
- `@Databricks Slack Agent what is the status of run 123456?`

Before triggering a job, the bot explains which live job it resolved, what it
does, and its recent duration, then displays a **Confirm run** button with the
exact job and parameters. Any current channel member may click it once within
10 minutes. The shared message names the confirmer, removes the button, and
shows the resulting run ID. Typing “yes” does not launch a job. The bot rechecks
the configuration at confirmation and refuses paused, unscheduled, non-main,
or changed jobs. Each dependent job needs its own confirmation. It can
check or cancel runs, but it cannot create, edit, pause, unpause, or delete job
configuration. Configuration changes belong in
the data repository's `resources/jobs/` and go through a pull request.

Run history shows observed executions, not configured cadence or root cause.
The bot checks job configuration before describing a schedule, recent runs
before calling a duration normal or unusual, and comparative data before
attributing runtime changes to volume or other causes. If those claims are not
supported, it returns an evidence-only summary instead.

## What to expect

- SQL execution is read-only. The bot cannot modify Databricks tables or views.
- Answers should distinguish live Databricks evidence from repository logic
  and documented business context.
- Slack responses are normalized for Slack formatting, including tabular output
  and links.
- Long answers are split into numbered `Part N of M` messages. If Slack cannot
  confirm every part, the bot stops, marks delivery as incomplete, and does not
  ask for feedback on the partial answer.
- Raw query results are capped. If output is sampled or truncated, the bot
  discloses that and avoids treating the sample as the full population.
- The bot may ask for clarification when the metric, population, or source is
  genuinely ambiguous.
- Images can be attached to a question. Other Slack file types are ignored.

If a tool or data object is unavailable, the bot should report the specific
blocker rather than guess. For access or operational help, contact the bot's
administrators. If the bot reports an error reference such as
`agent-0123456789ab`, include that reference when asking for help; it identifies
the matching internal error without exposing technical details in Slack.
