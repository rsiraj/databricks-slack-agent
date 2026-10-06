# Terraform infrastructure

This directory is a standalone deployment for a new copy of the bot. It wraps
the reusable `modules/data-assistant` AWS module with provider, tag, and GitHub
OIDC configuration.

It creates:

- ECR repository and image-retention policy;
- app-credentials secret container (no secret value);
- DynamoDB session, answer-feedback, and rate-limit tables (sessions and
  feedback have point-in-time recovery and, by default, deletion protection);
- Lambda/API Gateway CloudWatch logs;
- separate least-privilege receiver and worker execution roles, plus a
  code-only GitHub Actions deploy role;
- receiver and worker image Lambdas with async-invoke retries disabled
  (`maximum_retry_attempts = 0`; the worker also drops events older than 60s)
  so a Slack turn is never replayed by Lambda;
- HTTP API Gateway `POST /slack/events` with route throttling.

It does not create or modify Databricks resources. Supply the **full** ARN of
an existing AWS Secrets Manager secret containing Databricks `host` and
`token`, including the 6-character suffix Secrets Manager appends
(`...:secret:name-AbCdEf`). An ARN without the suffix is wildcarded in IAM.

Requires Terraform 1.10 or newer (the S3 backend example uses `use_lockfile`).

## Quick start

```bash
cp infrastructure/terraform.tfvars.example infrastructure/terraform.tfvars
# edit infrastructure/terraform.tfvars
./scripts/bootstrap-infrastructure.sh
./scripts/configure-app-secret.sh
```

See `docs/SETUP.md` for prerequisites, Slack configuration, remote-state
guidance, and verification.

## Bootstrap image

Lambda requires its ECR image to exist before Terraform can create the
functions. The bootstrap helper safely resolves this ordering:

1. targeted apply for only `module.data_assistant.aws_ecr_repository.this`;
2. local Docker build and ECR push;
3. complete, un-targeted Terraform apply.

After creation, `image_uri` is ignored by Terraform because the GitHub Actions
deployment workflow owns code releases. Everything else about the functions
(handler `CMD`, environment variables, memory, timeout, roles) stays owned by
Terraform; the deploy role deliberately lacks
`lambda:UpdateFunctionConfiguration`.

## Runtime knobs

The worker's model and agent-loop settings are module variables so operators
can tune them without a code change:

| Variable | Env var | Default |
|---|---|---|
| `llm_provider` | `LLM_PROVIDER` | `anthropic` |
| `llm_model` | `LLM_MODEL` | `""` (provider default) |
| `anthropic_model` | `ANTHROPIC_MODEL` | `claude-sonnet-5` |
| `anthropic_effort` | `ANTHROPIC_EFFORT` | `high` |
| `anthropic_max_tokens` | `ANTHROPIC_MAX_TOKENS` | `24000` |
| `anthropic_timeout_seconds` | `ANTHROPIC_TIMEOUT_SECONDS` | `180` |
| `agent_min_seconds_per_iteration` | `AGENT_MIN_SECONDS_PER_ITERATION` | `90` |

Capacity controls: `receiver_reserved_concurrent_executions` and
`worker_reserved_concurrent_executions` (default `null` = unreserved), and
`api_throttling_burst_limit` / `api_throttling_rate_limit` (default 50 / 25).

## Using the child module elsewhere

Organizations that keep application infrastructure in a central Terraform
repository can copy or source `modules/data-assistant` directly. The caller must
provide:

- AWS provider configuration;
- a GitHub OIDC provider ARN;
- bot and data repository names;
- an existing Databricks credential-secret ARN and warehouse ID;
- an ECR image tag that already exists;
- desired resource tags.

The root configuration in this directory is the supported easiest path for a
new fork.
