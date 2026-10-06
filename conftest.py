"""Pytest bootstrap — must run before any project module is imported.

Several modules touch AWS at *import* time:

  - `utils.secrets` / `tools.databricks_tool` construct `boto3.client(...)` at the
    top level (needs a region to be configured).
  - `utils.sessions` / `utils.ratelimit` construct `boto3.resource("dynamodb")`.
  - `utils.slack_helpers` does `from utils.clients import slack_client`, which the
    lazy `__getattr__` in `utils.clients` would resolve by *constructing* a real
    Slack client (and fetching the bot token from Secrets Manager).

We neutralise all of that here so the pure functions under test can be imported
without any network access or real credentials:

  - set dummy AWS env vars + a region (boto3 client construction is offline);
  - pre-seed `utils.clients._clients` with placeholders so the lazy accessor
    returns them instead of building real clients.
"""

import os
import sys
from pathlib import Path

# Dummy environment — boto3 client *construction* is offline; these just satisfy
# region/credential lookups so no client raises at import time.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ.setdefault("APP_SECRET_ARN", "arn:aws:secretsmanager:us-east-1:000000000000:secret:test")
os.environ.setdefault("DATABRICKS_SECRET_ARN", "arn:aws:secretsmanager:us-east-1:000000000000:secret:test-dbx")
os.environ.setdefault("DATABRICKS_WAREHOUSE_ID", "test-warehouse-id")
# The production default (claude-sonnet-5 / high -> max_tokens 24000) is above the
# auto-streaming threshold, so `ANTHROPIC_STREAM=auto` would call
# `messages.stream(...)`. The fake SDK clients in the worker / eval-runner tests
# implement `messages.create` only; the streaming transport has its own tests in
# tests/test_llm_anthropic_provider.py (which delenv/setenv this explicitly).
os.environ.setdefault("ANTHROPIC_STREAM", "never")

# Make the repo root importable (`from tools...`, `from utils...`, `import worker`).
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Pre-seed the lazy client registry so `from utils.clients import slack_client`
# (and friends) returns a placeholder instead of building a real client.
import utils.clients as _clients  # noqa: E402

_clients._clients.setdefault("slack_client", object())
_clients._clients.setdefault("anthropic_client", object())
_clients._clients.setdefault("lambda_client", object())
