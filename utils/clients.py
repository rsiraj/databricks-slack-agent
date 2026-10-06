"""Client initializations for external services.

Clients are lazy — they're only constructed (and their env vars touched) when first
accessed. This lets the receiver Lambda import `lambda_client` without needing
`ANTHROPIC_API_KEY` to be set, and vice versa.
"""

import logging
import os

from pythonjsonlogger.json import JsonFormatter

logger = logging.getLogger()
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

WORKER_FUNCTION_NAME = os.environ.get("WORKER_FUNCTION_NAME", "databricks-slack-agent-worker")
ANTHROPIC_MAX_RETRIES = max(0, int(os.environ.get("ANTHROPIC_MAX_RETRIES", "5")))
# Per-request wall-clock budget for one messages.create call. Applied on every
# call by the worker (see worker._call_model) rather than baked into the client
# so the agent loop can shrink it as the Lambda deadline approaches.
ANTHROPIC_TIMEOUT_SECONDS = max(1.0, float(os.environ.get("ANTHROPIC_TIMEOUT_SECONDS", "180")))
ANTHROPIC_CONNECT_TIMEOUT_SECONDS = 10.0

_clients: dict[str, object] = {}
# Only clients constructed here carry credential metadata. Explicitly injected
# clients (for example the offline evaluator's SDK fake) remain caller-owned.
_client_credentials: dict[str, tuple[object, str]] = {}


def anthropic_request_timeout(seconds: float):
    """Build a per-request timeout for ``messages.create``.

    Uses the SDK's ``Timeout`` type when available so the connect phase gets
    its own short budget and only read/write/pool share ``seconds``; falls back
    to a plain float (total budget) on SDKs without it.
    """
    seconds = max(1.0, float(seconds))
    try:
        import anthropic
        timeout_type = getattr(anthropic, "Timeout", None)
    except Exception:  # pragma: no cover - import failure handled by caller
        timeout_type = None
    if timeout_type is None:
        return seconds
    return timeout_type(
        connect=min(ANTHROPIC_CONNECT_TIMEOUT_SECONDS, seconds),
        read=seconds,
        write=seconds,
        pool=seconds,
    )


def __getattr__(name: str):
    """Lazy attribute access — `from utils.clients import lambda_client` still works,
    but the import doesn't construct any client we didn't ask for."""
    credential_key = {
        "anthropic_client": "anthropic_api_key",
        "slack_client": "slack_bot_token",
    }.get(name)
    cached = _clients.get(name)
    credential_state = _client_credentials.get(name)
    if cached is not None and (
        credential_key is None
        or credential_state is None
        or credential_state[0] is not cached
    ):
        return cached

    if credential_key is not None:
        from utils.secrets import get_secret

        # get_secret owns the five-minute Secrets Manager TTL. Check it even
        # for warm clients so rotating/revoking a key does not require a cold
        # start, while unchanged credentials retain their connection pools.
        credential = get_secret(credential_key)
        if cached is not None and credential_state[1] == credential:
            return cached

    if name == "anthropic_client":
        import anthropic
        _clients[name] = anthropic.Anthropic(
            api_key=credential,
            max_retries=ANTHROPIC_MAX_RETRIES,
        )
        _client_credentials[name] = (_clients[name], credential)
        return _clients[name]

    if name == "slack_client":
        from slack_sdk import WebClient
        _clients[name] = WebClient(token=credential)
        _client_credentials[name] = (_clients[name], credential)
        return _clients[name]

    if name == "lambda_client":
        import boto3
        _clients[name] = boto3.client("lambda")
        return _clients[name]

    raise AttributeError(f"module 'utils.clients' has no attribute {name!r}")
