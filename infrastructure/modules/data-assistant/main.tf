data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  receiver_function_name = "${var.app_name}-receiver"
  worker_function_name   = "${var.app_name}-worker"
  sessions_table_name    = "${var.app_name}-sessions"
  feedback_table_name    = "${var.app_name}-feedback"
  ratelimit_table_name   = "${var.app_name}-ratelimit"
  lambda_image_uri       = "${aws_ecr_repository.this.repository_url}:${var.lambda_image_tag}"

  receiver_function_arn = "arn:${data.aws_partition.current.partition}:lambda:${var.aws_region}:${data.aws_caller_identity.current.account_id}:function:${local.receiver_function_name}"
  worker_function_arn   = "arn:${data.aws_partition.current.partition}:lambda:${var.aws_region}:${data.aws_caller_identity.current.account_id}:function:${local.worker_function_name}"

  github_oidc_subjects = distinct([
    "repo:${var.github_repository}:environment:${var.github_environment}",
    "repo:${var.github_repository}:ref:refs/heads/${var.github_deploy_branch}",
  ])

  receiver_environment = {
    APP_SECRET_ARN       = aws_secretsmanager_secret.app_credentials.arn
    RATELIMIT_TABLE_NAME = aws_dynamodb_table.ratelimit.name
    WORKER_FUNCTION_NAME = local.worker_function_name
  }

  worker_environment = merge(
    var.worker_environment_variables,
    {
      AGENT_MIN_SECONDS_PER_ITERATION = tostring(var.agent_min_seconds_per_iteration)
      ANTHROPIC_EFFORT                = var.anthropic_effort
      ANTHROPIC_MAX_TOKENS            = tostring(var.anthropic_max_tokens)
      ANTHROPIC_MODEL                 = var.anthropic_model
      ANTHROPIC_TIMEOUT_SECONDS       = tostring(var.anthropic_timeout_seconds)
      APP_SECRET_ARN                  = aws_secretsmanager_secret.app_credentials.arn
      DATABRICKS_ROW_LIMIT            = tostring(var.databricks_row_limit)
      DATABRICKS_SECRET_ARN           = var.databricks_secret_arn
      DATABRICKS_WAREHOUSE_ID         = var.databricks_warehouse_id
      FEEDBACK_TABLE_NAME             = aws_dynamodb_table.feedback.name
      GITHUB_DEFAULT_REF              = var.github_default_ref
      GITHUB_REPO                     = var.data_github_repository
      LLM_MODEL                       = var.llm_model
      LLM_PROVIDER                    = var.llm_provider
      RATELIMIT_CHANNEL_PER_HOUR      = tostring(var.ratelimit_channel_per_hour)
      RATELIMIT_SQL_PER_DAY           = tostring(var.ratelimit_sql_per_day)
      RATELIMIT_TABLE_NAME            = aws_dynamodb_table.ratelimit.name
      RATELIMIT_USER_PER_DAY          = tostring(var.ratelimit_user_per_day)
      RATELIMIT_USER_PER_HOUR         = tostring(var.ratelimit_user_per_hour)
      SESSIONS_TABLE_NAME             = aws_dynamodb_table.sessions.name
      WORKER_FUNCTION_NAME            = local.worker_function_name
    },
  )

  # Secrets Manager appends a random 6-character suffix ("-AbCdEf") to every
  # secret ARN, and IAM matches resource ARNs exactly. A full ARN (with suffix)
  # is used verbatim; an ARN supplied without the suffix gets a "-*" wildcard so
  # the policy still resolves. Callers are expected to pass the full ARN.
  app_secret_resource_arns = [
    for arn in [aws_secretsmanager_secret.app_credentials.arn] :
    can(regex("-[A-Za-z0-9]{6}$", arn)) ? arn : "${arn}-*"
  ]
  databricks_secret_resource_arns = [
    for arn in [var.databricks_secret_arn] :
    can(regex("-[A-Za-z0-9]{6}$", arn)) ? arn : "${arn}-*"
  ]
  secret_resource_arns = distinct(concat(
    local.app_secret_resource_arns,
    local.databricks_secret_resource_arns,
  ))
}

resource "aws_ecr_repository" "this" {
  name                 = var.app_name
  image_tag_mutability = "MUTABLE"
  force_delete         = var.ecr_force_delete

  encryption_configuration {
    encryption_type = "AES256"
  }

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = var.tags
}

resource "aws_ecr_lifecycle_policy" "this" {
  repository = aws_ecr_repository.this.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = var.ecr_untagged_expiration_days
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Keep recent main branch images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["main-"]
          countType     = "imageCountMoreThan"
          countNumber   = var.ecr_keep_image_count
        }
        action = {
          type = "expire"
        }
      },
    ]
  })
}

resource "aws_secretsmanager_secret" "app_credentials" {
  name                    = "${var.app_name}/app-credentials"
  description             = "Anthropic, Slack, and GitHub credentials for ${var.app_name}; value is populated outside Terraform."
  kms_key_id              = var.app_credentials_kms_key_id
  recovery_window_in_days = var.app_secret_recovery_window_days

  tags = var.tags
}

resource "aws_dynamodb_table" "sessions" {
  name                        = local.sessions_table_name
  billing_mode                = "PAY_PER_REQUEST"
  hash_key                    = "thread_ts"
  deletion_protection_enabled = var.dynamodb_deletion_protection_enabled

  attribute {
    name = "thread_ts"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled = true
  }

  tags = var.tags
}

resource "aws_dynamodb_table" "feedback" {
  name                        = local.feedback_table_name
  billing_mode                = "PAY_PER_REQUEST"
  hash_key                    = "answer_key"
  range_key                   = "user_id"
  deletion_protection_enabled = var.dynamodb_deletion_protection_enabled

  attribute {
    name = "answer_key"
    type = "S"
  }

  attribute {
    name = "user_id"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled = true
  }

  tags = var.tags
}

# Rate-limit counters are short-lived (per-window TTL) and safe to recreate, so
# this table is not protected against deletion and has no point-in-time recovery.
resource "aws_dynamodb_table" "ratelimit" {
  name                        = local.ratelimit_table_name
  billing_mode                = "PAY_PER_REQUEST"
  hash_key                    = "key"
  deletion_protection_enabled = false

  attribute {
    name = "key"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  server_side_encryption {
    enabled = true
  }

  tags = var.tags
}

resource "aws_cloudwatch_log_group" "receiver" {
  name              = "/aws/lambda/${local.receiver_function_name}"
  retention_in_days = var.log_retention_days

  tags = var.tags
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/aws/lambda/${local.worker_function_name}"
  retention_in_days = var.log_retention_days

  tags = var.tags
}

resource "aws_cloudwatch_log_group" "api_gateway" {
  count = var.api_access_logs_enabled ? 1 : 0

  name              = "/aws/apigateway/${var.app_name}"
  retention_in_days = var.log_retention_days

  tags = var.tags
}

data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

# ---------------------------------------------------------------------------
# Receiver execution role: verify Slack signatures, dedupe event IDs, and hand
# off to the worker. It never touches Databricks credentials or session state.
# ---------------------------------------------------------------------------

resource "aws_iam_role" "receiver" {
  name               = "${var.app_name}-receiver-lambda"
  description        = "Runtime role for the ${var.app_name} receiver Lambda."
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "receiver_basic_execution" {
  role       = aws_iam_role.receiver.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "receiver_runtime" {
  statement {
    sid       = "InvokeWorker"
    actions   = ["lambda:InvokeFunction"]
    resources = [local.worker_function_arn]
  }

  statement {
    sid       = "ReadAppSecret"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = local.app_secret_resource_arns
  }

  # Slack event-id dedupe uses the same conditional UpdateItem pattern as the
  # rate limiter, on the rate-limit table.
  statement {
    sid       = "DedupeSlackEvents"
    actions   = ["dynamodb:UpdateItem"]
    resources = [aws_dynamodb_table.ratelimit.arn]
  }

  dynamic "statement" {
    for_each = length(var.kms_decrypt_key_arns) > 0 ? [1] : []

    content {
      sid       = "DecryptAppSecret"
      actions   = ["kms:Decrypt"]
      resources = var.kms_decrypt_key_arns
    }
  }
}

resource "aws_iam_role_policy" "receiver_runtime" {
  name   = "${var.app_name}-receiver-runtime"
  role   = aws_iam_role.receiver.id
  policy = data.aws_iam_policy_document.receiver_runtime.json
}

# ---------------------------------------------------------------------------
# Worker execution role: the agent loop, Databricks/GitHub/Anthropic access,
# session persistence, feedback, and rate limiting.
# ---------------------------------------------------------------------------

resource "aws_iam_role" "worker" {
  name               = "${var.app_name}-worker-lambda"
  description        = "Runtime role for the ${var.app_name} worker Lambda."
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "worker_basic_execution" {
  role       = aws_iam_role.worker.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "worker_runtime" {
  statement {
    sid       = "InvokeWorker"
    actions   = ["lambda:InvokeFunction"]
    resources = [local.worker_function_arn]
  }

  statement {
    sid       = "ReadCredentialSecrets"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = local.secret_resource_arns
  }

  statement {
    sid = "ReadWriteSessions"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
    ]
    resources = [aws_dynamodb_table.sessions.arn]
  }

  statement {
    sid       = "WriteAnswerFeedback"
    actions   = ["dynamodb:PutItem"]
    resources = [aws_dynamodb_table.feedback.arn]
  }

  statement {
    sid       = "UpdateRateLimits"
    actions   = ["dynamodb:UpdateItem"]
    resources = [aws_dynamodb_table.ratelimit.arn]
  }

  dynamic "statement" {
    for_each = length(var.kms_decrypt_key_arns) > 0 ? [1] : []

    content {
      sid       = "DecryptCredentialSecrets"
      actions   = ["kms:Decrypt"]
      resources = var.kms_decrypt_key_arns
    }
  }
}

resource "aws_iam_role_policy" "worker_runtime" {
  name   = "${var.app_name}-worker-runtime"
  role   = aws_iam_role.worker.id
  policy = data.aws_iam_policy_document.worker_runtime.json
}

# ---------------------------------------------------------------------------
# GitHub Actions deployment role (OIDC).
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "gha_assume_role" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    effect  = "Allow"

    principals {
      type        = "Federated"
      identifiers = [var.github_oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = local.github_oidc_subjects
    }
  }
}

resource "aws_iam_role" "gha_deploy" {
  name               = "gha-${var.app_name}"
  description        = "GitHub Actions deployment role for ${var.github_repository}."
  assume_role_policy = data.aws_iam_policy_document.gha_assume_role.json

  tags = var.tags
}

data "aws_iam_policy_document" "gha_deploy" {
  statement {
    sid       = "ECRLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "ECRPushPull"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:DescribeImages",
      "ecr:DescribeRepositories",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:ListImages",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]
    resources = [aws_ecr_repository.this.arn]
  }

  # Code-only deploys. Function configuration (handler CMD, env vars, memory,
  # timeout, role) is owned by Terraform, so the workflow deliberately lacks
  # lambda:UpdateFunctionConfiguration. The Get* actions back the post-deploy
  # wait/verify step.
  statement {
    sid = "LambdaUpdateCode"
    actions = [
      "lambda:GetFunction",
      "lambda:GetFunctionConfiguration",
      "lambda:UpdateFunctionCode",
    ]
    resources = [
      local.receiver_function_arn,
      local.worker_function_arn,
    ]
  }
}

resource "aws_iam_role_policy" "gha_deploy" {
  name   = "${var.app_name}-deploy"
  role   = aws_iam_role.gha_deploy.id
  policy = data.aws_iam_policy_document.gha_deploy.json
}

# ---------------------------------------------------------------------------
# Lambda functions.
# ---------------------------------------------------------------------------

resource "aws_lambda_function" "receiver" {
  function_name                  = local.receiver_function_name
  description                    = "Acknowledges and verifies Slack events for ${var.app_name}."
  role                           = aws_iam_role.receiver.arn
  package_type                   = "Image"
  image_uri                      = local.lambda_image_uri
  architectures                  = ["x86_64"]
  memory_size                    = var.receiver_memory_size
  timeout                        = var.receiver_timeout_seconds
  reserved_concurrent_executions = var.receiver_reserved_concurrent_executions

  image_config {
    command = ["receiver.lambda_handler"]
  }

  environment {
    variables = local.receiver_environment
  }

  depends_on = [
    aws_cloudwatch_log_group.receiver,
    aws_iam_role_policy.receiver_runtime,
    aws_iam_role_policy_attachment.receiver_basic_execution,
  ]

  lifecycle {
    ignore_changes = [image_uri]
  }

  tags = var.tags
}

resource "aws_lambda_function" "worker" {
  function_name                  = local.worker_function_name
  description                    = "Runs the Databricks Slack agent for ${var.app_name}."
  role                           = aws_iam_role.worker.arn
  package_type                   = "Image"
  image_uri                      = local.lambda_image_uri
  architectures                  = ["x86_64"]
  memory_size                    = var.worker_memory_size
  timeout                        = var.worker_timeout_seconds
  reserved_concurrent_executions = var.worker_reserved_concurrent_executions

  image_config {
    command = ["worker.lambda_handler"]
  }

  environment {
    variables = local.worker_environment
  }

  depends_on = [
    aws_cloudwatch_log_group.worker,
    aws_iam_role_policy.worker_runtime,
    aws_iam_role_policy_attachment.worker_basic_execution,
  ]

  lifecycle {
    ignore_changes = [image_uri]
  }

  tags = var.tags
}

# Asynchronous-invocation policy. Lambda's default for Event invocations is two
# automatic retries with events kept for up to six hours. A Slack turn is not
# idempotent: a retried worker would re-run the agent loop, post duplicate
# answers into the thread, and could trigger the same Databricks job twice. A
# retried receiver would re-dispatch the worker. Both functions therefore get
# zero retries. The worker also drops events older than 60 seconds, because a
# reply that Lambda could not even start within a minute of the @-mention is
# already stale from the user's point of view (Slack itself retries the
# original HTTP delivery within seconds, and the receiver dedupes those).
resource "aws_lambda_function_event_invoke_config" "receiver" {
  function_name          = aws_lambda_function.receiver.function_name
  maximum_retry_attempts = 0
}

resource "aws_lambda_function_event_invoke_config" "worker" {
  function_name                = aws_lambda_function.worker.function_name
  maximum_retry_attempts       = 0
  maximum_event_age_in_seconds = 60
}

# ---------------------------------------------------------------------------
# HTTP API for Slack Events / Interactivity.
# ---------------------------------------------------------------------------

resource "aws_apigatewayv2_api" "this" {
  name          = var.app_name
  description   = "Slack Events API for ${var.app_name}."
  protocol_type = "HTTP"

  tags = var.tags
}

resource "aws_apigatewayv2_integration" "receiver" {
  api_id                 = aws_apigatewayv2_api.this.id
  integration_type       = "AWS_PROXY"
  integration_method     = "POST"
  integration_uri        = aws_lambda_function.receiver.invoke_arn
  payload_format_version = "2.0"
  timeout_milliseconds   = 30000
}

resource "aws_apigatewayv2_route" "slack_events" {
  api_id    = aws_apigatewayv2_api.this.id
  route_key = "POST /slack/events"
  target    = "integrations/${aws_apigatewayv2_integration.receiver.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.this.id
  name        = "$default"
  auto_deploy = true

  # Slack delivers one POST per event; these limits are far above any legitimate
  # workspace's traffic and cap the blast radius of a replayed/flooded endpoint.
  default_route_settings {
    throttling_burst_limit = var.api_throttling_burst_limit
    throttling_rate_limit  = var.api_throttling_rate_limit
  }

  dynamic "access_log_settings" {
    for_each = var.api_access_logs_enabled ? [1] : []

    content {
      destination_arn = aws_cloudwatch_log_group.api_gateway[0].arn
      format = jsonencode({
        requestId               = "$context.requestId"
        routeKey                = "$context.routeKey"
        status                  = "$context.status"
        integrationStatus       = "$context.integrationStatus"
        integrationErrorMessage = "$context.integrationErrorMessage"
        responseLength          = "$context.responseLength"
        requestTime             = "$context.requestTime"
        httpMethod              = "$context.httpMethod"
        path                    = "$context.path"
        protocol                = "$context.protocol"
        sourceIp                = "$context.identity.sourceIp"
        userAgent               = "$context.identity.userAgent"
        responseLatency         = "$context.responseLatency"
        integrationLatency      = "$context.integrationLatency"
      })
    }
  }

  tags = var.tags
}

resource "aws_lambda_permission" "allow_apigateway" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.receiver.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.this.execution_arn}/*/POST/slack/events"
}
