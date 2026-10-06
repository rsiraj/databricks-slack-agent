variable "app_name" {
  description = "Stable prefix used for all bot resources."
  type        = string

  validation {
    condition     = length(var.app_name) <= 50 && can(regex("^[a-z0-9]+(?:-[a-z0-9]+)*$", var.app_name))
    error_message = "app_name must be at most 50 characters and contain only lowercase letters, numbers, and single dashes."
  }
}

variable "aws_region" {
  description = "AWS region hosting the bot resources."
  type        = string

  validation {
    condition     = trimspace(var.aws_region) != ""
    error_message = "aws_region must not be empty."
  }
}

variable "github_repository" {
  description = "Repository that runs the deploy workflow, in owner/name form."
  type        = string

  validation {
    condition     = can(regex("^[^/]+/[^/]+$", var.github_repository))
    error_message = "github_repository must use owner/name format."
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

variable "github_oidc_provider_arn" {
  description = "ARN of the account-wide GitHub Actions OIDC provider."
  type        = string
}

variable "data_github_repository" {
  description = "Data repository searched by the bot, in owner/name form."
  type        = string

  validation {
    condition     = can(regex("^[^/]+/[^/]+$", var.data_github_repository))
    error_message = "data_github_repository must use owner/name format."
  }
}

variable "github_default_ref" {
  description = "Default ref used by the bot's GitHub tools."
  type        = string
  default     = "main"
}

variable "databricks_secret_arn" {
  description = "Existing Secrets Manager secret ARN containing Databricks host/token. Pass the full ARN including the 6-character Secrets Manager suffix (e.g. ...:secret:databricks/token/team-AbCdEf); an ARN without the suffix is wildcarded."
  type        = string

  validation {
    condition     = can(regex("^arn:[^:]+:secretsmanager:[^:]+:[0-9]{12}:secret:", var.databricks_secret_arn))
    error_message = "databricks_secret_arn must be a full AWS Secrets Manager secret ARN."
  }
}

variable "databricks_warehouse_id" {
  description = "Databricks SQL warehouse ID used by the worker."
  type        = string

  validation {
    condition     = trimspace(var.databricks_warehouse_id) != ""
    error_message = "databricks_warehouse_id must not be empty."
  }
}

variable "lambda_image_tag" {
  description = "ECR image tag used for initial Lambda creation. GitHub Actions owns later image updates."
  type        = string
  default     = "bootstrap"

  validation {
    condition     = trimspace(var.lambda_image_tag) != ""
    error_message = "lambda_image_tag must not be empty."
  }
}

variable "receiver_memory_size" {
  description = "Receiver Lambda memory in MB."
  type        = number
  default     = 1024

  validation {
    condition     = var.receiver_memory_size >= 128 && var.receiver_memory_size <= 10240
    error_message = "receiver_memory_size must be between 128 and 10240 MB."
  }
}

variable "receiver_timeout_seconds" {
  description = "Receiver Lambda timeout in seconds."
  type        = number
  default     = 10

  validation {
    condition     = var.receiver_timeout_seconds >= 1 && var.receiver_timeout_seconds <= 900
    error_message = "receiver_timeout_seconds must be between 1 and 900."
  }
}

variable "worker_memory_size" {
  description = "Worker Lambda memory in MB."
  type        = number
  default     = 2048

  validation {
    condition     = var.worker_memory_size >= 128 && var.worker_memory_size <= 10240
    error_message = "worker_memory_size must be between 128 and 10240 MB."
  }
}

variable "worker_timeout_seconds" {
  description = "Worker Lambda timeout in seconds."
  type        = number
  default     = 900

  validation {
    condition     = var.worker_timeout_seconds >= 1 && var.worker_timeout_seconds <= 900
    error_message = "worker_timeout_seconds must be between 1 and 900."
  }
}

variable "receiver_reserved_concurrent_executions" {
  description = "Reserved concurrency for the receiver Lambda. null leaves it unreserved (account pool)."
  type        = number
  default     = null
  nullable    = true

  validation {
    condition     = var.receiver_reserved_concurrent_executions == null || try(var.receiver_reserved_concurrent_executions >= 0, false)
    error_message = "receiver_reserved_concurrent_executions must be null or a non-negative number."
  }
}

variable "worker_reserved_concurrent_executions" {
  description = "Reserved concurrency for the worker Lambda. null leaves it unreserved (account pool)."
  type        = number
  default     = null
  nullable    = true

  validation {
    condition     = var.worker_reserved_concurrent_executions == null || try(var.worker_reserved_concurrent_executions >= 0, false)
    error_message = "worker_reserved_concurrent_executions must be null or a non-negative number."
  }
}

variable "api_throttling_burst_limit" {
  description = "API Gateway default-route throttling burst limit (requests)."
  type        = number
  default     = 50

  validation {
    condition     = var.api_throttling_burst_limit >= 1
    error_message = "api_throttling_burst_limit must be at least 1."
  }
}

variable "api_throttling_rate_limit" {
  description = "API Gateway default-route steady-state throttling rate (requests per second)."
  type        = number
  default     = 25

  validation {
    condition     = var.api_throttling_rate_limit > 0
    error_message = "api_throttling_rate_limit must be greater than 0."
  }
}

variable "log_retention_days" {
  description = "CloudWatch retention for Lambda and API Gateway logs."
  type        = number
  default     = 365

  validation {
    condition = contains(
      [1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1096, 1827, 2192, 2557, 2922, 3288, 3653],
      var.log_retention_days,
    )
    error_message = "log_retention_days must be a CloudWatch Logs-supported retention value."
  }
}

variable "api_access_logs_enabled" {
  description = "Whether to write API Gateway access logs to CloudWatch."
  type        = bool
  default     = true
}

variable "dynamodb_deletion_protection_enabled" {
  description = "Protect the session and answer-feedback tables from accidental deletion. The rate-limit table holds transient counters and is never protected."
  type        = bool
  default     = true
}

variable "app_secret_recovery_window_days" {
  description = "Secrets Manager recovery window for the app credentials secret."
  type        = number
  default     = 30

  validation {
    condition     = var.app_secret_recovery_window_days == 0 || (var.app_secret_recovery_window_days >= 7 && var.app_secret_recovery_window_days <= 30)
    error_message = "app_secret_recovery_window_days must be 0 or between 7 and 30."
  }
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

  validation {
    condition     = var.ecr_keep_image_count >= 1
    error_message = "ecr_keep_image_count must be at least 1."
  }
}

variable "ecr_untagged_expiration_days" {
  description = "Age in days after which untagged ECR images expire."
  type        = number
  default     = 1

  validation {
    condition     = var.ecr_untagged_expiration_days >= 1
    error_message = "ecr_untagged_expiration_days must be at least 1."
  }
}

variable "databricks_row_limit" {
  description = "Maximum rows requested from the Databricks Statement Execution API."
  type        = number
  default     = 1000

  validation {
    condition     = var.databricks_row_limit >= 1
    error_message = "databricks_row_limit must be at least 1."
  }
}

variable "ratelimit_user_per_hour" {
  description = "Maximum bot mentions per user per hour."
  type        = number
  default     = 60

  validation {
    condition     = var.ratelimit_user_per_hour >= 1
    error_message = "ratelimit_user_per_hour must be at least 1."
  }
}

variable "ratelimit_user_per_day" {
  description = "Maximum bot mentions per user per day."
  type        = number
  default     = 200

  validation {
    condition     = var.ratelimit_user_per_day >= 1
    error_message = "ratelimit_user_per_day must be at least 1."
  }
}

variable "ratelimit_channel_per_hour" {
  description = "Maximum bot mentions per channel per hour."
  type        = number
  default     = 200

  validation {
    condition     = var.ratelimit_channel_per_hour >= 1
    error_message = "ratelimit_channel_per_hour must be at least 1."
  }
}

variable "ratelimit_sql_per_day" {
  description = "Maximum Databricks SQL submissions across the bot per day."
  type        = number
  default     = 500

  validation {
    condition     = var.ratelimit_sql_per_day >= 1
    error_message = "ratelimit_sql_per_day must be at least 1."
  }
}

variable "anthropic_model" {
  description = "Anthropic model ID used by the worker agent loop (ANTHROPIC_MODEL)."
  type        = string
  default     = "claude-sonnet-5"

  validation {
    condition     = trimspace(var.anthropic_model) != ""
    error_message = "anthropic_model must not be empty."
  }
}

variable "anthropic_effort" {
  description = "Effort level for the worker agent loop (ANTHROPIC_EFFORT; applies to whichever provider is selected). xhigh maps to max on models without it."
  type        = string
  default     = "high"

  validation {
    condition     = contains(["low", "medium", "high", "xhigh", "max"], var.anthropic_effort)
    error_message = "anthropic_effort must be one of low, medium, high, xhigh, max."
  }
}

variable "llm_provider" {
  description = "Model provider for the worker agent loop (LLM_PROVIDER): anthropic or openai. openai additionally requires the openai package in the image and an openai_api_key in the app secret."
  type        = string
  default     = "anthropic"

  validation {
    condition     = contains(["anthropic", "openai"], var.llm_provider)
    error_message = "llm_provider must be one of anthropic, openai."
  }
}

variable "llm_model" {
  description = "Provider-neutral model override (LLM_MODEL). Empty uses ANTHROPIC_MODEL / OPENAI_MODEL defaults for the selected provider."
  type        = string
  default     = ""
}

variable "anthropic_max_tokens" {
  description = "Maximum output tokens per Anthropic request (ANTHROPIC_MAX_TOKENS)."
  type        = number
  default     = 24000

  validation {
    condition     = var.anthropic_max_tokens >= 1
    error_message = "anthropic_max_tokens must be at least 1."
  }
}

variable "anthropic_timeout_seconds" {
  description = "Per-request Anthropic client timeout in seconds (ANTHROPIC_TIMEOUT_SECONDS)."
  type        = number
  default     = 180

  validation {
    condition     = var.anthropic_timeout_seconds >= 1 && var.anthropic_timeout_seconds <= 900
    error_message = "anthropic_timeout_seconds must be between 1 and 900."
  }
}

variable "agent_min_seconds_per_iteration" {
  description = "Minimum remaining Lambda time (seconds) required before the worker starts another agent iteration (AGENT_MIN_SECONDS_PER_ITERATION)."
  type        = number
  default     = 90

  validation {
    condition     = var.agent_min_seconds_per_iteration >= 1 && var.agent_min_seconds_per_iteration <= 900
    error_message = "agent_min_seconds_per_iteration must be between 1 and 900."
  }
}

variable "worker_environment_variables" {
  description = "Additional worker environment variables. Required module values take precedence."
  type        = map(string)
  default     = {}
}

variable "tags" {
  description = "Tags applied to every supported resource."
  type        = map(string)
  default     = {}
}
