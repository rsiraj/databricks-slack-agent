"""Consolidated app secrets, fetched from a single AWS Secrets Manager secret.

Expected secret JSON shape:

    {
      "anthropic_api_key": "sk-ant-...",
      "slack_bot_token": "xoxb-...",
      "slack_signing_secret": "...",
      "github_token": "github_pat_..."
    }

Set `APP_SECRET_ARN` on both Lambdas (receiver + worker) to the secret's ARN.
The Lambda execution role needs `secretsmanager:GetSecretValue` on that ARN.

Receiver uses: `slack_bot_token`, `slack_signing_secret`.
Worker uses: all four.

The secret is fetched on first access and cached for ~5 minutes inside a warm
container. After a manual rotation, expect up to 5 minutes of stale reads
before the cache refreshes.
"""

import json
import logging
import os
import time
from typing import Any

import boto3

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 300  # 5 minutes

_secrets_client = boto3.client("secretsmanager")
_cache: dict[str, Any] = {"data": None, "expires_at": 0.0}


def _load_app_secret(force_refresh: bool = False) -> dict:
    now = time.time()
    if not force_refresh and _cache["data"] is not None and _cache["expires_at"] > now:
        return _cache["data"]

    resp = _secrets_client.get_secret_value(SecretId=os.environ["APP_SECRET_ARN"])
    blob = json.loads(resp["SecretString"])

    _cache["data"] = blob
    _cache["expires_at"] = now + CACHE_TTL_SECONDS
    return blob


def get_secret(key: str) -> str:
    """Fetch a single key from the consolidated app secret. Raises KeyError if missing."""
    blob = _load_app_secret()
    if key not in blob:
        raise KeyError(f"App secret missing required key: {key!r}")
    return blob[key]
