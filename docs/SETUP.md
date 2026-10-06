# Set up a new bot instance

This guide deploys a copy of the bot into an AWS account and connects it
to an existing Databricks workspace. AWS infrastructure is created by Terraform;
Slack app configuration remains a short manual step.

Allow roughly 20–30 minutes after prerequisites are available.

## 1. Prerequisites

You need:

- A fork or copy of this repository.
- Terraform 1.10+, Docker, AWS CLI v2, and Python 3.
- AWS access for IAM, ECR, Lambda, Secrets Manager, DynamoDB, CloudWatch, and
  API Gateway.
- An account-wide GitHub Actions OIDC provider in AWS. The Terraform can create
  one if the account does not already have it.
- An existing AWS Secrets Manager secret containing Databricks credentials:

  ```json
  {
    "host": "https://your-workspace.cloud.databricks.com",
    "token": "dapi..."
  }
  ```

- A Serverless Databricks SQL warehouse ID is recommended so interactive bot
  queries do not wait through the longer startup time of a stopped classic or
  pro warehouse. The service principal behind the token needs `CAN_USE` on the
  warehouse and read access to the catalogs the bot may query.
- Slack workspace permission to create/install an app.
- An Anthropic API key and a fine-grained GitHub token with read access to the
  data repository.

Long-lived Databricks credentials should be owned and rotated by the platform
Terraform that manages the workspace. This repository consumes only the
existing secret ARN; it creates no Databricks resources and never reads the
secret value during Terraform operations.

## 2. Customize the bot repository

Before deploying:

1. Replace `prompts/business/` with your organization's catalog map, metric
   glossary, and operational context. The shipped files describe a fictional
   online shop and are only a template; leaving them in a real deployment will
   produce confidently incorrect answers. `tests/test_metadata_prompt_guidance.py`
   and `tests/test_conversation_learning_guidance.py` assert on that example
   text, so update them alongside your own content.
2. Replace the placeholder IDs in `config/admins.json` with the Slack user IDs allowed to administer the bot.
3. Replace the placeholder ID in `config/allowed_channels.json` with the private Slack channel IDs in which
   the bot may operate.
4. Review `prompts/core/` for remaining team-specific language.

Both config lists fail closed: empty admin/channel configuration prevents the
bot from operating.

## 3. Create the Slack app

Create an app at `api.slack.com/apps` from scratch, but leave Event
Subscriptions disabled until Terraform has produced the request URL.

Add these bot token scopes:

- `app_mentions:read`
- `chat:write`
- `files:read` (image attachments on questions)
- `files:write` (attach generated SQL files to answer threads)
- `groups:read`
- `groups:write`
- `groups:history`

Install the app to the workspace, then retain:

- the Bot User OAuth Token (`xoxb-...`);
- the Signing Secret from Basic Information.

When you later subscribe to `app_mention`, `member_joined_channel`, and
`member_left_channel`, Slack force-adds `channels:read`, `channels:history`,
and `channels:manage` to the bot token; they cannot be removed while those
events are subscribed. The application still refuses to operate in public
channels through its channel allowlist and runtime privacy checks. Uncheck
"Required" on those scopes in the app config so reinstalls do not re-request
them.

## 4. Configure Terraform

From the repository root:

```bash
cp infrastructure/terraform.tfvars.example infrastructure/terraform.tfvars
```

Fill in every required value:

- `app_name`: unique lowercase/dash resource prefix;
- `bot_github_repository`: this fork in `owner/name` form;
- `data_github_repository`: repository searched by the GitHub tools;
- `databricks_secret_arn`: full ARN of the existing `host`/`token` secret,
  including the 6-character suffix Secrets Manager appends (`...name-AbCdEf`;
  copy it from `aws secretsmanager describe-secret --query ARN`);
- `databricks_warehouse_id`;
- `partner` and `team`: values for the cost-allocation tags.

Optional worker tuning variables are exported to the worker Lambda as
environment variables. Defaults:

| Variable | Env var | Default | Purpose |
|---|---|---|---|
| `llm_provider` | `LLM_PROVIDER` | `anthropic` | Model provider (`anthropic` or `openai`; see `docs/PROVIDERS.md`) |
| `llm_model` | `LLM_MODEL` | `""` | Provider-neutral model override; empty uses the provider default |
| `anthropic_model` | `ANTHROPIC_MODEL` | `claude-sonnet-5` | Anthropic model used by the agent loop |
| `anthropic_effort` | `ANTHROPIC_EFFORT` | `high` | Reasoning effort (`low`/`medium`/`high`/`xhigh`/`max`) |
| `anthropic_max_tokens` | `ANTHROPIC_MAX_TOKENS` | `24000` | Output-token cap per model call (`recommended_max_tokens` for the default model at `high`; above 21,000 the worker streams) |
| `anthropic_timeout_seconds` | `ANTHROPIC_TIMEOUT_SECONDS` | `180` | Per-request client timeout |
| `agent_min_seconds_per_iteration` | `AGENT_MIN_SECONDS_PER_ITERATION` | `90` | Remaining Lambda time required before starting another tool iteration |

Capacity controls: `receiver_reserved_concurrent_executions` /
`worker_reserved_concurrent_executions` (default `null`, unreserved) and
`api_throttling_burst_limit` / `api_throttling_rate_limit` (default 50 / 25
requests) on the Slack events route.

The AWS account ID is discovered from the active credentials. Standard tags are
applied to all supported resources:

- `Project = app_name`
- `Partner = partner`
- `Team = team`
- `IsProduction = is_production`
- `Provisioner = terraform://github.com/<bot repository>` unless overridden

### State

Local state works for a trial, but configure remote state before a shared
production deployment. `infrastructure/backend.tf.example` contains an S3
example. Terraform Cloud can use `infrastructure/` as its working directory
without a checked-in backend block.

Never commit `terraform.tfvars`, state files, plans, or credentials.

### GitHub OIDC

Many AWS accounts already contain the account-wide provider
`token.actions.githubusercontent.com`; leave
`create_github_oidc_provider = false` in that case.

Set it to `true` only when the provider is absent. Exactly one Terraform stack
should own this account-wide object.

## 5. Bootstrap AWS infrastructure

Authenticate to the intended AWS account. If you use SSO:

```bash
aws sso login --profile YOUR_PROFILE
export AWS_PROFILE=YOUR_PROFILE
```

Run:

```bash
./scripts/bootstrap-infrastructure.sh
```

The helper:

1. initializes Terraform;
2. creates the ECR repository first;
3. builds the repository's Docker image for Lambda `linux/amd64`;
4. pushes the configured bootstrap tag;
5. applies the full Terraform configuration;
6. prints the app secret ARN, GitHub deployment values, and Slack event URL.

Terraform uses targeting only for the initial ECR bootstrap dependency. The
final un-targeted apply reconciles the complete configuration.

The script passes additional arguments to both applies. For unattended sandbox
setup, for example:

```bash
./scripts/bootstrap-infrastructure.sh -auto-approve
```

To also configure GitHub repository variables with an authenticated `gh` CLI:

```bash
CONFIGURE_GITHUB_VARIABLES=true ./scripts/bootstrap-infrastructure.sh
```

Without that flag, the helper prints the exact `gh variable set` commands. The
deployment workflow needs:

- `APP_NAME`
- `AWS_REGION`
- `AWS_DEPLOY_ROLE_ARN`
- `DEPLOY_ENVIRONMENT`

The required PR conversation evaluation also needs a repository Actions secret
named `MODEL_EVAL_ANTHROPIC_API_KEY`. Use an evaluation-only Anthropic
workspace/project with a low spend ceiling; do not reuse the production app
secret. With an authenticated GitHub CLI:

```bash
gh secret set MODEL_EVAL_ANTHROPIC_API_KEY
```

In the `main` branch protection or ruleset, require the exact
`Conversation evaluations` status check. The check only runs when a PR touches
files that can change model behavior (`prompts/`, `tools/`, `utils/`,
`worker.py`, `evals/`, the workflow, or an `anthropic`/`openai` pin); otherwise
it is skipped, which counts as passing. When it runs it evaluates every fixture
three times and blocks only on a regression relative to the committed
`evals/baseline.json`; infrastructure errors are labelled separately (exit
code 2) and never read as regressions. The normal `pull_request` workflow can
use this secret for trusted branches in the same repository; GitHub withholds it
from fork-originated pull requests. See `evals/README.md` ("The PR gate") for
the synthetic-tool scope, the regression rule, and re-baselining.

## 6. Populate the app credentials secret

Run:

```bash
./scripts/configure-app-secret.sh
```

The helper securely prompts for:

- Anthropic API key;
- Slack bot token;
- Slack signing secret;
- GitHub token.

It writes a temporary mode-`0600` JSON file, sends it to Secrets Manager, and
deletes it. Values are not put in Terraform configuration/state or command
history.

The resulting secret shape is:

```json
{
  "anthropic_api_key": "...",
  "slack_bot_token": "xoxb-...",
  "slack_signing_secret": "...",
  "github_token": "github_pat_..."
}
```

## 7. Enable Slack events and interactivity

In the Slack app:

1. Open Event Subscriptions and enable events.
2. Paste the `slack_events_url` printed by the bootstrap helper (or run
   `terraform -chdir=infrastructure output -raw slack_events_url`).
3. Wait for Slack to verify the URL.
4. Subscribe to:
   - `app_mention`
   - `member_joined_channel`
   - `member_left_channel`
5. Save and reinstall the app if Slack requests it.
6. Open **Interactivity & Shortcuts**, enable interactivity, and paste the same
   `slack_events_url` as the Request URL. This endpoint handles the answer
   feedback buttons and private follow-up modals.
7. Make sure at least one Slack user listed in `config/admins.json` is already
   a member of the target private channel. The channel-access guard runs when
   the bot joins and leaves immediately if no allowlisted admin is present.
8. Invite the bot to one of the private channels in
   `config/allowed_channels.json`.

If verification fails, inspect the receiver log, and the API Gateway access
log (`/aws/apigateway/YOUR_APP_NAME`, enabled by default via
`api_access_logs_enabled`) to see whether Slack's request reached the API and
what status the integration returned:

```bash
aws logs tail /aws/lambda/YOUR_APP_NAME-receiver \
  --follow \
  --region YOUR_AWS_REGION
aws logs tail /aws/apigateway/YOUR_APP_NAME \
  --follow \
  --region YOUR_AWS_REGION
```

## 8. Test

Run the automated regression coverage first:

```bash
pytest tests/test_slack_formatting.py \
  tests/test_slack_delivery.py \
  tests/test_run_status.py \
  tests/test_error_reporting.py \
  tests/test_response_template_contract.py
```

Before opening or merging a prompt/tool-policy PR, also run the model evaluation
with its dedicated key, the way CI does (three repeats judged against the
committed baseline):

```bash
MODEL_EVAL_ANTHROPIC_API_KEY=... python -m evals.run \
  --repeats 3 --concurrency 3 --baseline evals/baseline.json
```

This calls the configured production model but uses only repository-owned
synthetic GitHub and Databricks fixtures. It exercises the same automatic
metadata preflight as production. Unresolved guard rejections, fixture/tool
errors, metadata-ordering mistakes, unsupported claims, and ungrounded figures
are hard violations that count against a scenario's pass rate; the check fails
only when a scenario's pass rate drops beyond the noise tolerance of the
baseline or a new kind of hard violation appears (exit 1), and it reports
provider/infrastructure errors separately (exit 2). Whether the answer covered
a specific content fact is reported as a soft violation without affecting the
verdict. A prompt or tool-policy PR that intentionally changes behavior must
also regenerate the entry for the deployed model in `evals/baseline.json`
in the same PR (other models' entries in the file are kept). A fresh clone
ships without a baseline, so record one first; until then the PR workflow
skips the gate:

```bash
MODEL_EVAL_ANTHROPIC_API_KEY=... python -m evals.run \
  --repeats 3 --concurrency 3 --write-baseline evals/baseline.json
```

`evals/README.md` ("Re-baselining is deliberate") shows how to build the entry
from one or more Model comparison artifacts instead of a live run.

Valid projections outside the bounded fixture model return an explicit
synthetic limitation with no rows or fabricated values. Unsupported cadence
claims receive the same private one-pass revision used in production.

In an allowed private channel:

1. Mention the bot with a question whose answer is known.
2. Verify it replies in a thread.
3. Mention it again inside the thread and verify session continuity.
4. Confirm an item appears in the `APP_NAME-sessions` DynamoDB table.
5. Submit each feedback rating, verify the two lower ratings request a private
   comment, and confirm records appear in `APP_NAME-feedback`.
6. Have a second channel member rate the same answer and confirm both users
   have distinct feedback records.
7. Ask for a sufficiently detailed answer to produce multiple parts. Confirm
   the messages are labeled `Part N of M`, code fences render independently,
   and the feedback panel appears once after the final part.
8. Ask a question that requires several tools. Confirm one progress message is
   edited from `Working on it...` to `Checked` rather than posting a new status
   message for every tool.

Tail runtime logs when troubleshooting:

```bash
aws logs tail /aws/lambda/YOUR_APP_NAME-receiver --follow --region YOUR_AWS_REGION
aws logs tail /aws/lambda/YOUR_APP_NAME-worker --follow --region YOUR_AWS_REGION
```

When Slack displays an `agent-<id>` reference, search the worker output for the
exact ID to correlate it with the full internal exception. Incomplete multipart
delivery is logged as `Slack answer delivery incomplete` and emits the
`answer_delivery_failed` audit event.

## 9. Ongoing changes

- Terraform owns infrastructure and runtime configuration.
- `.github/workflows/deploy.yml` owns container image releases after the first
  apply. Terraform intentionally ignores `image_uri` drift.
- Update infrastructure with:

  ```bash
  terraform -chdir=infrastructure plan
  terraform -chdir=infrastructure apply
  ```

- Merges to `main` run the unit tests, then build/push `main-<short-sha>`,
  update both Lambdas' code, and fail the run if either function does not
  report `State=Active` / `LastUpdateStatus=Successful` on the new image.
- Python dependencies are declared in `pyproject.toml` and resolved once into
  `uv.lock` (Python 3.12, the Lambda base image's minor). Nothing else is
  committed: CI runs `uv sync --locked` (which fails if the lock is stale) and
  the Dockerfile exports the runtime subset of `uv.lock` as a hashed
  requirements file at build time and installs it with `pip --require-hashes`.
  After changing a constraint in `pyproject.toml`, update the lock and commit
  both files:

  ```bash
  uv lock                 # or: uv lock --upgrade-package <name>
  uv sync                 # local .venv with the dev group for pytest
  ```

  Dependabot (`package-ecosystem: uv`) updates `pyproject.toml` and `uv.lock`
  together, so its PRs need no manual regeneration step.

- Rotate app credentials by rerunning `scripts/configure-app-secret.sh`.
- Databricks token rotation remains the responsibility of the workspace's
  platform IaC; Lambdas pick up the external secret's new value automatically.

Review every destroy plan carefully. ECR `force_delete` and DynamoDB deletion
protection are conservative/configurable so teardown must be intentional.
