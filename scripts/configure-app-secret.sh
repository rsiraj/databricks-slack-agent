#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TF_DIR="${REPO_ROOT}/infrastructure"

for command_name in terraform aws python3; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 1
  fi
done

if [[ ! -d "${TF_DIR}/.terraform" ]]; then
  echo "Terraform is not initialized. Run scripts/bootstrap-infrastructure.sh first." >&2
  exit 1
fi

SECRET_ARN="$(terraform -chdir="${TF_DIR}" output -raw app_credentials_secret_arn 2>/dev/null || true)"
AWS_REGION="$(terraform -chdir="${TF_DIR}" output -raw aws_region 2>/dev/null || true)"
if [[ -z "${SECRET_ARN}" || -z "${AWS_REGION}" ]]; then
  echo "Bot infrastructure outputs are unavailable. Run scripts/bootstrap-infrastructure.sh first." >&2
  exit 1
fi

TF_AWS_PROFILE="$(terraform -chdir="${TF_DIR}" output -json aws_profile 2>/dev/null || true)"
if [[ -z "${AWS_PROFILE:-}" && "${TF_AWS_PROFILE}" != "null" && -n "${TF_AWS_PROFILE}" ]]; then
  export AWS_PROFILE="${TF_AWS_PROFILE//\"/}"
fi

PAYLOAD_FILE="$(mktemp)"
trap 'rm -f "${PAYLOAD_FILE}"' EXIT
chmod 600 "${PAYLOAD_FILE}"

python3 - "${PAYLOAD_FILE}" <<'PY'
import getpass
import json
import sys

output_path = sys.argv[1]
prompts = {
    "anthropic_api_key": "Anthropic API key",
    "slack_bot_token": "Slack bot token",
    "slack_signing_secret": "Slack signing secret",
    "github_token": "GitHub token",
}

# Expected prefixes. A mismatch is a warning, not a failure: providers change
# token formats, and the operator may be pasting a legitimately different kind
# of credential (e.g. a classic GitHub PAT vs. a fine-grained one).
expected_prefixes = {
    "anthropic_api_key": ("sk-ant-",),
    "slack_bot_token": ("xoxb-",),
    "github_token": ("ghp_", "github_pat_"),
}

payload = {}
for key, label in prompts.items():
    value = getpass.getpass(f"{label}: ").strip()
    if not value:
        raise SystemExit(f"{label} must not be empty.")
    prefixes = expected_prefixes.get(key)
    if prefixes and not value.startswith(prefixes):
        print(
            f"WARNING: {label} does not start with {' or '.join(prefixes)}; "
            "double-check you pasted the right credential.",
            file=sys.stderr,
        )
    payload[key] = value

with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle)
PY

aws secretsmanager put-secret-value \
  --secret-id "${SECRET_ARN}" \
  --secret-string "file://${PAYLOAD_FILE}" \
  --region "${AWS_REGION}" \
  >/dev/null

echo "Updated ${SECRET_ARN}."
echo "Lambda reads the new values on its next cold start (or after the warm-container cache expires)."
