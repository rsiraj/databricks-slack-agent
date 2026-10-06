output "app_credentials_secret_arn" {
  description = "ARN of the app credentials secret to populate outside Terraform."
  value       = aws_secretsmanager_secret.app_credentials.arn
}

output "ecr_repository_url" {
  description = "ECR repository URL used for image pushes."
  value       = aws_ecr_repository.this.repository_url
}

output "gha_deploy_role_arn" {
  description = "GitHub Actions deployment role ARN."
  value       = aws_iam_role.gha_deploy.arn
}

output "receiver_execution_role_arn" {
  description = "Receiver Lambda execution role ARN."
  value       = aws_iam_role.receiver.arn
}

output "worker_execution_role_arn" {
  description = "Worker Lambda execution role ARN."
  value       = aws_iam_role.worker.arn
}

output "receiver_function_name" {
  description = "Receiver Lambda function name."
  value       = aws_lambda_function.receiver.function_name
}

output "worker_function_name" {
  description = "Worker Lambda function name."
  value       = aws_lambda_function.worker.function_name
}

output "sessions_table_name" {
  description = "DynamoDB sessions table name."
  value       = aws_dynamodb_table.sessions.name
}

output "feedback_table_name" {
  description = "DynamoDB answer-feedback table name."
  value       = aws_dynamodb_table.feedback.name
}

output "ratelimit_table_name" {
  description = "DynamoDB rate-limit table name."
  value       = aws_dynamodb_table.ratelimit.name
}

output "api_endpoint" {
  description = "Base URL for the bot HTTP API."
  value       = aws_apigatewayv2_api.this.api_endpoint
}

output "slack_events_url" {
  description = "Request URL to configure in Slack Event Subscriptions."
  value       = "${aws_apigatewayv2_api.this.api_endpoint}/slack/events"
}
