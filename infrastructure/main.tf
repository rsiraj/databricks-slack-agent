provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile

  default_tags {
    tags = local.common_tags
  }
}

data "aws_caller_identity" "current" {}

locals {
  common_tags = merge(
    var.additional_tags,
    {
      IsProduction = tostring(var.is_production)
      Partner      = var.partner
      Project      = var.app_name
      Provisioner  = coalesce(var.provisioner, "terraform://github.com/${var.bot_github_repository}")
      Team         = var.team
    },
  )
}

resource "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 1 : 0

  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = var.github_oidc_thumbprints

  tags = local.common_tags
}

data "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 0 : 1

  url = "https://token.actions.githubusercontent.com"
}

locals {
  github_oidc_provider_arn = var.create_github_oidc_provider ? (
    aws_iam_openid_connect_provider.github[0].arn
    ) : (
    data.aws_iam_openid_connect_provider.github[0].arn
  )
}

module "data_assistant" {
  source = "./modules/data-assistant"

  app_name                 = var.app_name
  aws_region               = var.aws_region
  github_repository        = var.bot_github_repository
  github_deploy_branch     = var.github_deploy_branch
  github_environment       = var.github_environment
  github_oidc_provider_arn = local.github_oidc_provider_arn

  data_github_repository = var.data_github_repository
  github_default_ref     = var.github_default_ref

  databricks_secret_arn   = var.databricks_secret_arn
  databricks_warehouse_id = var.databricks_warehouse_id
  lambda_image_tag        = var.lambda_image_tag

  receiver_memory_size                    = var.receiver_memory_size
  receiver_timeout_seconds                = var.receiver_timeout_seconds
  receiver_reserved_concurrent_executions = var.receiver_reserved_concurrent_executions
  worker_memory_size                      = var.worker_memory_size
  worker_timeout_seconds                  = var.worker_timeout_seconds
  worker_reserved_concurrent_executions   = var.worker_reserved_concurrent_executions

  api_throttling_burst_limit = var.api_throttling_burst_limit
  api_throttling_rate_limit  = var.api_throttling_rate_limit

  log_retention_days                   = var.log_retention_days
  api_access_logs_enabled              = var.api_access_logs_enabled
  dynamodb_deletion_protection_enabled = var.dynamodb_deletion_protection_enabled

  app_secret_recovery_window_days = var.app_secret_recovery_window_days
  app_credentials_kms_key_id      = var.app_credentials_kms_key_id
  kms_decrypt_key_arns            = var.kms_decrypt_key_arns

  ecr_force_delete             = var.ecr_force_delete
  ecr_keep_image_count         = var.ecr_keep_image_count
  ecr_untagged_expiration_days = var.ecr_untagged_expiration_days

  databricks_row_limit         = var.databricks_row_limit
  ratelimit_user_per_hour      = var.ratelimit_user_per_hour
  ratelimit_user_per_day       = var.ratelimit_user_per_day
  ratelimit_channel_per_hour   = var.ratelimit_channel_per_hour
  ratelimit_sql_per_day        = var.ratelimit_sql_per_day
  worker_environment_variables = var.worker_environment_variables

  llm_provider                    = var.llm_provider
  llm_model                       = var.llm_model
  anthropic_model                 = var.anthropic_model
  anthropic_effort                = var.anthropic_effort
  anthropic_max_tokens            = var.anthropic_max_tokens
  anthropic_timeout_seconds       = var.anthropic_timeout_seconds
  agent_min_seconds_per_iteration = var.agent_min_seconds_per_iteration

  tags = local.common_tags
}
