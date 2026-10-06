variable "aws_region" {
  description = "AWS region that hosts the bot runtime."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Optional local AWS CLI profile. Leave null in CI/Terraform Cloud."
  type        = string
  default     = null
  nullable    = true
}

variable "app_name" {
  description = "Stable prefix used for all bot resources."
  type        = string

  validation {
    condition     = length(var.app_name) <= 50 && can(regex("^[a-z0-9]+(?:-[a-z0-9]+)*$", var.app_name))
    error_message = "app_name must be at most 50 characters and contain only lowercase letters, numbers, and single dashes."
  }
}

variable "bot_github_repository" {
  description = "Fork/copy of this bot repository, in owner/name form."
  type        = string

  validation {
    condition     = can(regex("^[^/]+/[^/]+$", var.bot_github_repository))
    error_message = "bot_github_repository must use owner/name format."
  }
}

variable "data_github_repository" {
  description = "Data repository searched by the bot, in owner/name form."
  type        = string

  validation {
    condition     = can(regex("^[^/]+/[^/]+$", var.data_github_repository))
    error_message = "data_github_repository must use owner/name format."
  }
}

variable "github_deploy_branch" {
  description = "Branch allowed to deploy through GitHub OIDC."
  type        = string
  default     = "main"
}

variable "github_environment" {
  description = "GitHub Environment bound to the deployment workflow."
  type        = string
  default     = "production"
}

variable "github_default_ref" {
  description = "Default ref used by the bot's GitHub tools."
  type        = string
  default     = "main"
}

variable "create_github_oidc_provider" {
  description = "Create the account-wide GitHub OIDC provider. Keep false when the AWS account already has one."
  type        = bool
  default     = false
}

variable "github_oidc_thumbprints" {
  description = "TLS thumbprints used only when create_github_oidc_provider is true."
  type        = list(string)
  default     = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

variable "databricks_secret_arn" {
  description = "Existing Secrets Manager secret ARN containing Databricks host/token. Use the full ARN including the 6-character suffix Secrets Manager appends."
  type        = string
}

variable "databricks_warehouse_id" {
  description = "Databricks SQL warehouse ID used by the worker."
  type        = string
}

variable "lambda_image_tag" {
  description = "Initial ECR image tag. The bootstrap helper pushes this before creating Lambda."
  type        = string
  default     = "bootstrap"
}

variable "partner" {
  description = "Business-unit or cost-center code used for the Partner cost-allocation tag."
  type        = string
}

variable "team" {
  description = "Owning team name used for the Team tag."
  type        = string
}

variable "is_production" {
  description = "Whether this deployment is production."
  type        = bool
  default     = true
}

variable "provisioner" {
  description = "Provisioner tag. Defaults to terraform://github.com/<bot repository>."
  type        = string
  default     = null
  nullable    = true
}

variable "additional_tags" {
  description = "Additional tags. Standard Project/Partner/Team/IsProduction/Provisioner values take precedence."
  type        = map(string)
  default     = {}
}

variable "receiver_memory_size" {
  description = "Receiver Lambda memory in MB."
  type        = number
  default     = 1024
}

variable "receiver_timeout_seconds" {
  description = "Receiver Lambda timeout in seconds."
  type        = number
  default     = 10
}

variable "worker_memory_size" {
  description = "Worker Lambda memory in MB."
  type        = number
  default     = 2048
}

variable "worker_timeout_seconds" {
  description = "Worker Lambda timeout in seconds."
  type        = number
  default     = 900
}

variable "receiver_reserved_concurrent_executions" {
  description = "Reserved concurrency for the receiver Lambda. null leaves it unreserved."
  type        = number
  default     = null
  nullable    = true
}

variable "worker_reserved_concurrent_executions" {
  description = "Reserved concurrency for the worker Lambda. null leaves it unreserved."
  type        = number
  default     = null
  nullable    = true
}

variable "api_throttling_burst_limit" {
  description = "API Gateway default-route throttling burst limit (requests)."
  type        = number
  default     = 50
}

variable "api_throttling_rate_limit" {
  description = "API Gateway default-route steady-state throttling rate (requests per second)."
  type        = number
  default     = 25
}

variable "log_retention_days" {
  description = "CloudWatch retention for Lambda and API Gateway logs."
  type        = number
  default     = 365
}

variable "api_access_logs_enabled" {
  description = "Whether to write API Gateway access logs to CloudWatch."
  type        = bool
  default     = true
}

variable "dynamodb_deletion_protection_enabled" {
  description = "Protect the session and answer-feedback tables from accidental deletion."
  type        = bool
  default     = true
}

variable "app_secret_recovery_window_days" {
  description = "Secrets Manager recovery window for the app credentials secret."
  type        = number
  default     = 30
}

variable "app_credentials_kms_key_id" {
  description = "Optional KMS key ID/ARN used to encrypt the app credentials secret."
  type        = string
  default     = null
  nullable    = true
}

variable "kms_decrypt_key_arns" {
  description = "Customer-managed KMS key ARNs Lambda may decrypt for app/Databricks secrets."
  type        = list(string)
  default     = []
}

variable "ecr_force_delete" {
  description = "Allow Terraform to delete a non-empty ECR repository."
  type        = bool
  default     = false
}

variable "ecr_keep_image_count" {
  description = "Number of main branch images retained by the ECR lifecycle policy."
  type        = number
  default     = 10
}

variable "ecr_untagged_expiration_days" {
  description = "Age in days after which untagged ECR images expire."
  type        = number
  default     = 1
}

variable "databricks_row_limit" {
  description = "Maximum rows requested from Databricks per SQL statement."
  type        = number
  default     = 1000
}

variable "ratelimit_user_per_hour" {
  description = "Maximum bot mentions per user per hour."
  type        = number
  default     = 60
}

variable "ratelimit_user_per_day" {
  description = "Maximum bot mentions per user per day."
  type        = number
  default     = 200
}

variable "ratelimit_channel_per_hour" {
  description = "Maximum bot mentions per channel per hour."
  type        = number
  default     = 200
}

variable "ratelimit_sql_per_day" {
  description = "Maximum Databricks SQL submissions across the bot per day."
  type        = number
  default     = 500
}

variable "llm_provider" {
  description = "Model provider for the worker: anthropic (default) or openai (LLM_PROVIDER)."
  type        = string
  default     = "anthropic"
}

variable "llm_model" {
  description = "Provider-neutral model override; empty uses the selected provider's default (LLM_MODEL)."
  type        = string
  default     = ""
}

variable "anthropic_model" {
  description = "Anthropic model ID used by the worker (ANTHROPIC_MODEL)."
  type        = string
  default     = "claude-sonnet-5"
}

variable "anthropic_effort" {
  description = "Effort level: low, medium, high, xhigh, or max (ANTHROPIC_EFFORT)."
  type        = string
  default     = "high"
}

variable "anthropic_max_tokens" {
  description = "Maximum output tokens per Anthropic request (ANTHROPIC_MAX_TOKENS)."
  type        = number
  default     = 24000
}

variable "anthropic_timeout_seconds" {
  description = "Per-request Anthropic client timeout in seconds (ANTHROPIC_TIMEOUT_SECONDS)."
  type        = number
  default     = 180
}

variable "agent_min_seconds_per_iteration" {
  description = "Minimum remaining Lambda seconds required before another agent iteration starts (AGENT_MIN_SECONDS_PER_ITERATION)."
  type        = number
  default     = 90
}

variable "worker_environment_variables" {
  description = "Additional worker environment variables."
  type        = map(string)
  default     = {}
}
