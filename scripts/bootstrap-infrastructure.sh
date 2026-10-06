#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TF_DIR="${REPO_ROOT}/infrastructure"
TFVARS_FILE="${TF_DIR}/terraform.tfvars"

for command_name in terraform aws docker; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 1
  fi
done

if [[ ! -f "${TFVARS_FILE}" ]]; then
  echo "Missing ${TFVARS_FILE}" >&2
  echo "Copy infrastructure/terraform.tfvars.example to infrastructure/terraform.tfvars and fill in every required value." >&2
  exit 1
fi

echo "Initializing Terraform..."
terraform -chdir="${TF_DIR}" init

echo
echo "Step 1/3: create or verify the ECR repository."
echo "Terraform targeting is used only for this bootstrap dependency; the final apply reconciles the complete configuration."
terraform -chdir="${TF_DIR}" apply \
  -target=module.data_assistant.aws_ecr_repository.this \
  "$@"

ECR_URL="$(terraform -chdir="${TF_DIR}" output -raw ecr_repository_url)"
AWS_REGION="$(terraform -chdir="${TF_DIR}" output -raw aws_region)"
IMAGE_TAG="$(terraform -chdir="${TF_DIR}" output -raw lambda_image_tag)"
export AWS_REGION

TF_AWS_PROFILE="$(terraform -chdir="${TF_DIR}" output -json aws_profile 2>/dev/null || true)"
if [[ -z "${AWS_PROFILE:-}" && "${TF_AWS_PROFILE}" != "null" && -n "${TF_AWS_PROFILE}" ]]; then
  export AWS_PROFILE="${TF_AWS_PROFILE//\"/}"
fi

# After a targeted apply, outputs that depend on un-targeted resources are not
# guaranteed to be populated, so resolve the account from the live credentials
# and check it against the account embedded in the ECR registry hostname.
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ECR_URL%%/*}"
REGISTRY_ACCOUNT="${REGISTRY%%.*}"
if [[ "${ACCOUNT_ID}" != "${REGISTRY_ACCOUNT}" ]]; then
  echo "AWS identity mismatch: Terraform created ECR in account ${REGISTRY_ACCOUNT}, but the AWS CLI is authenticated to ${ACCOUNT_ID}." >&2
  echo "Set AWS_PROFILE to the same profile configured in infrastructure/terraform.tfvars, then retry." >&2
  exit 1
fi

IMAGE_URI="${ECR_URL}:${IMAGE_TAG}"

echo
echo "Step 2/3: build and push ${IMAGE_URI}."
aws ecr get-login-password --region "${AWS_REGION}" |
  docker login --username AWS --password-stdin "${REGISTRY}"
# Lambda rejects OCI image indexes; disable BuildKit provenance/SBOM attestations
# so the pushed tag is a single-platform manifest.
docker build --platform linux/amd64 --provenance=false --sbom=false \
  -t "${IMAGE_URI}" "${REPO_ROOT}"
docker push "${IMAGE_URI}"

echo
echo "Step 3/3: apply the complete bot infrastructure."
terraform -chdir="${TF_DIR}" apply "$@"

APP_NAME="$(terraform -chdir="${TF_DIR}" output -raw app_name)"
BOT_REPO="$(terraform -chdir="${TF_DIR}" output -raw bot_github_repository)"
DEPLOY_ROLE_ARN="$(terraform -chdir="${TF_DIR}" output -raw gha_deploy_role_arn)"
DEPLOY_ENVIRONMENT="$(terraform -chdir="${TF_DIR}" output -raw github_environment)"
SLACK_EVENTS_URL="$(terraform -chdir="${TF_DIR}" output -raw slack_events_url)"
APP_SECRET_ARN="$(terraform -chdir="${TF_DIR}" output -raw app_credentials_secret_arn)"

if [[ "${CONFIGURE_GITHUB_VARIABLES:-false}" == "true" ]]; then
  if ! command -v gh >/dev/null 2>&1; then
    echo "CONFIGURE_GITHUB_VARIABLES=true, but gh is not installed." >&2
    exit 1
  fi

  echo
  echo "Configuring GitHub repository variables on ${BOT_REPO}..."
  gh variable set APP_NAME --repo "${BOT_REPO}" --body "${APP_NAME}"
  gh variable set AWS_REGION --repo "${BOT_REPO}" --body "${AWS_REGION}"
  gh variable set AWS_DEPLOY_ROLE_ARN --repo "${BOT_REPO}" --body "${DEPLOY_ROLE_ARN}"
  gh variable set DEPLOY_ENVIRONMENT --repo "${BOT_REPO}" --body "${DEPLOY_ENVIRONMENT}"
fi

echo
echo "Infrastructure is ready."
echo "Slack event URL: ${SLACK_EVENTS_URL}"
echo "App credentials secret: ${APP_SECRET_ARN}"
echo
echo "Next:"
echo "  1. Run scripts/configure-app-secret.sh to populate the app credentials secret."
echo "  2. Configure Slack Event Subscriptions with the URL above."
echo "  3. Subscribe to app_mention, member_joined_channel, and member_left_channel."

if [[ "${CONFIGURE_GITHUB_VARIABLES:-false}" != "true" ]]; then
  echo "  4. Configure deployment variables (or rerun with CONFIGURE_GITHUB_VARIABLES=true):"
  echo "     gh variable set APP_NAME --repo ${BOT_REPO} --body ${APP_NAME}"
  echo "     gh variable set AWS_REGION --repo ${BOT_REPO} --body ${AWS_REGION}"
  echo "     gh variable set AWS_DEPLOY_ROLE_ARN --repo ${BOT_REPO} --body ${DEPLOY_ROLE_ARN}"
  echo "     gh variable set DEPLOY_ENVIRONMENT --repo ${BOT_REPO} --body ${DEPLOY_ENVIRONMENT}"
fi
