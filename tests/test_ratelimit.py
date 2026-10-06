"""DynamoDB rate limiter and one-shot dedupe claim — utils/ratelimit.py.

Every check is a single conditional UpdateItem. These tests stub the table so
no network is touched and assert the fail-open / fail-closed contract for each
error class botocore can raise.
"""

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from utils import ratelimit


class _FakeTable:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[dict] = []

    def update_item(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {}


def _client_error(code: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": "simulated"}},
        "UpdateItem",
    )


def _endpoint_error() -> EndpointConnectionError:
    return EndpointConnectionError(endpoint_url="https://dynamodb.us-east-1.amazonaws.com")


@pytest.fixture
def table(monkeypatch):
    def _install(error=None):
        fake = _FakeTable(error)
        monkeypatch.setattr(ratelimit, "_table", fake)
        return fake

    return _install


class TestCheckAndIncrement:
    def test_allows_under_limit(self, table):
        fake = table()

        assert ratelimit._check_and_increment("user", "U1", "hour", 60) is True

        call = fake.calls[0]
        assert call["Key"]["key"].startswith("user#U1#hour#")
        assert call["ConditionExpression"] == "attribute_not_exists(#c) OR #c < :limit"
        assert call["ExpressionAttributeValues"][":limit"] == 60
        assert call["ExpressionAttributeValues"][":one"] == 1

    def test_blocks_at_limit(self, table):
        table(_client_error("ConditionalCheckFailedException"))

        assert ratelimit._check_and_increment("user", "U1", "hour", 60) is False

    def test_fails_open_on_other_client_error(self, table, caplog):
        table(_client_error("ProvisionedThroughputExceededException"))
        caplog.set_level("ERROR")

        assert ratelimit._check_and_increment("user", "U1", "hour", 60) is True
        assert "failing open" in caplog.text

    def test_fails_open_on_endpoint_connection_error(self, table, caplog):
        table(_endpoint_error())
        caplog.set_level("ERROR")

        assert ratelimit._check_and_increment("channel", "C1", "hour", 200) is True
        assert "transport error" in caplog.text

    def test_check_user_returns_reason_when_blocked(self, table):
        table(_client_error("ConditionalCheckFailedException"))

        allowed, reason = ratelimit.check_user("U1")

        assert allowed is False
        assert reason.startswith("per_user_hour")

    def test_check_user_without_id_fails_open_without_ddb(self, table):
        fake = table()

        assert ratelimit.check_user(None) == (True, None)
        assert fake.calls == []


class TestClaimOnce:
    def test_first_claim_succeeds_with_conditional_update_and_ttl(self, table, monkeypatch):
        fake = table()
        monkeypatch.setattr(ratelimit.time, "time", lambda: 1_800_000_000)

        assert ratelimit.claim_once("dedupe#event#Ev1", ttl_seconds=600) is True

        call = fake.calls[0]
        assert call["Key"] == {"key": "dedupe#event#Ev1"}
        assert call["ConditionExpression"] == "attribute_not_exists(#k) OR #rel = :yes"
        assert call["ExpressionAttributeNames"] == {"#k": "key", "#ttl": "ttl", "#rel": "released"}
        assert call["ExpressionAttributeValues"] == {":ttl": 1_800_000_600, ":no": False, ":yes": True}
        # UpdateItem (not PutItem) — the execution role only grants UpdateItem.
        assert "Item" not in call

    def test_second_claim_is_rejected(self, table):
        table(_client_error("ConditionalCheckFailedException"))

        assert ratelimit.claim_once("dedupe#event#Ev1") is False

    def test_other_client_error_returns_none(self, table, caplog):
        table(_client_error("InternalServerError"))
        caplog.set_level("ERROR")

        assert ratelimit.claim_once("dedupe#event#Ev1") is None
        assert "Dedupe claim DDB error" in caplog.text

    def test_endpoint_error_returns_none(self, table, caplog):
        table(_endpoint_error())
        caplog.set_level("ERROR")

        assert ratelimit.claim_once("dedupe#event#Ev1") is None
        assert "transport error" in caplog.text


class TestReleaseClaim:
    def test_claim_condition_allows_released_keys(self, monkeypatch):
        from utils import ratelimit

        calls = []

        class Table:
            def update_item(self, **kwargs):
                calls.append(kwargs)

        monkeypatch.setattr(ratelimit, "_table", Table())
        assert ratelimit.claim_once("dedupe#x") is True
        cond = calls[0]["ConditionExpression"]
        assert "attribute_not_exists(#k)" in cond and "#rel = :yes" in cond
        assert calls[0]["ExpressionAttributeValues"][":no"] is False

    def test_release_sets_released_flag(self, monkeypatch):
        from utils import ratelimit

        calls = []

        class Table:
            def update_item(self, **kwargs):
                calls.append(kwargs)

        monkeypatch.setattr(ratelimit, "_table", Table())
        assert ratelimit.release_claim("dedupe#x") is True
        assert calls[0]["UpdateExpression"] == "SET #rel = :yes"
        assert calls[0]["Key"] == {"key": "dedupe#x"}

    def test_release_swallows_ddb_errors(self, monkeypatch):
        from botocore.exceptions import EndpointConnectionError

        from utils import ratelimit

        class Table:
            def update_item(self, **kwargs):
                raise EndpointConnectionError(endpoint_url="https://dynamodb")

        monkeypatch.setattr(ratelimit, "_table", Table())
        assert ratelimit.release_claim("dedupe#x") is False
