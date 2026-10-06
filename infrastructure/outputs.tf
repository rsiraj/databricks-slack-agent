output "aws_account_id" {
  description = "AWS account hosting the bot."
  value       = data.aws_caller_identity.current.account_id
}

output "aws_region" {
  description = "AWS region hosting the bot."
  value       = var.aws_region
}

output "aws_profile" {
  description = "Optional local AWS profile used by Terraform."
  value       = var.aws_profile
}

output "app_name" {
  description = "Configured bot resource prefix."
  value       = var.app_name
}

output "bot_github_repository" {
  description = "Repository authorized to deploy the bot."
  value       = var.bot_github_repository
}

output "github_environment" {
  description = "GitHub Environment authorized to deploy."
  value       = var.github_environment
}

output "lambda_image_tag" {
  description = "Initial image tag used during bootstrap."
  value       = var.lambda_image_tag
}

output "app_credentials_secret_arn" {
  description = "Secret ARN to populate with Anthropic, Slack, and GitHub credentials."
  value       = module.data_assistant.app_credentials_secret_arn
}

output "ecr_repository_url" {
  description = "ECR repository URL used by bootstrap and deployment workflows."
  value       = module.data_assistant.ecr_repository_url
}

output "gha_deploy_role_arn" {
  description = "Role ARN to store as the AWS_DEPLOY_ROLE_ARN GitHub repository variable."
  value       = module.data_assistant.gha_deploy_role_arn
}

output "receiver_execution_role_arn" {
  description = "Receiver Lambda execution role ARN."
  value       = module.data_assistant.receiver_execution_role_arn
}

output "worker_execution_role_arn" {
  description = "Worker Lambda execution role ARN."
  value       = module.data_assistant.worker_execution_role_arn
}

output "receiver_function_name" {
  description = "Receiver Lambda function name."
  value       = module.data_assistant.receiver_function_name
}

output "worker_function_name" {
  description = "Worker Lambda function name."
  value       = module.data_assistant.worker_function_name
}

output "feedback_table_name" {
  description = "DynamoDB table containing per-answer Slack feedback."
  value       = module.data_assistant.feedback_table_name
}

output "slack_events_url" {
  description = "Request URL for Slack Event Subscriptions."
  value       = module.data_assistant.slack_events_url
}
