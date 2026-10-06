"""Receiver retry handling — dedupe by Slack event identity instead of dropping retries.

`claim_once` and `_invoke_worker` are stubbed; the tests assert which deliveries
reach the worker under fresh/duplicate/DDB-error conditions.
"""

import json
from urllib.parse import urlencode

import pytest

import receiver


@pytest.fixture
def harness(monkeypatch):
    state = {"invoked": [], "claims": [], "claim_result": True}

    monkeypatch.setattr(receiver, "verify_slack_signature", lambda headers, body: True)
    monkeypatch.setattr(
        receiver,
        "_invoke_worker",
        lambda payload: state["invoked"].append(payload) or True,
    )

    def fake_claim(key, ttl_seconds=600):
        state["claims"].append((key, ttl_seconds))
        return state["claim_result"]

    monkeypatch.setattr(receiver, "claim_once", fake_claim)
    return state


def _mention_payload(**overrides) -> dict:
    payload = {
        "type": "event_callback",
        "event_id": "Ev123",
        "event": {
            "type": "app_mention",
            "user": "U1",
            "channel": "C1",
            "ts": "1700000000.000100",
            "event_ts": "1700000000.000100",
            "text": "<@BOT> hi",
        },
    }
    payload.update(overrides)
    return payload


def _request(payload: dict, retry_num: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if retry_num is not None:
        headers["X-Slack-Retry-Num"] = retry_num
        headers["X-Slack-Retry-Reason"] = "http_timeout"
    return {"headers": headers, "body": json.dumps(payload)}


def _interaction_request(payload: dict, retry_num: str | None = None) -> dict:
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if retry_num is not None:
        headers["X-Slack-Retry-Num"] = retry_num
    return {"headers": headers, "body": urlencode({"payload": json.dumps(payload)})}


class TestEventDedupe:
    def test_first_delivery_is_claimed_and_dispatched(self, harness):
        response = receiver.lambda_handler(_request(_mention_payload()), None)

        assert response["statusCode"] == 200
        assert harness["claims"] == [("dedupe#event#Ev123", receiver.DEDUPE_TTL_SECONDS)]
        assert len(harness["invoked"]) == 1

    def test_retry_is_dispatched_when_original_was_never_claimed(self, harness):
        """A retry whose original was lost (e.g. cold-start timeout before the
        claim) must still be processed — this is the behavior change."""
        response = receiver.lambda_handler(_request(_mention_payload(), retry_num="1"), None)

        assert response["statusCode"] == 200
        assert len(harness["invoked"]) == 1

    def test_duplicate_delivery_is_acked_without_dispatch(self, harness):
        harness["claim_result"] = False

        response = receiver.lambda_handler(_request(_mention_payload(), retry_num="1"), None)

        assert response["statusCode"] == 200
        assert harness["invoked"] == []

    def test_duplicate_on_retry_zero_is_also_skipped(self, harness):
        harness["claim_result"] = False

        receiver.lambda_handler(_request(_mention_payload()), None)

        assert harness["invoked"] == []

    def test_ddb_error_falls_back_to_dispatching_only_original(self, harness):
        harness["claim_result"] = None

        receiver.lambda_handler(_request(_mention_payload()), None)
        receiver.lambda_handler(_request(_mention_payload(), retry_num="1"), None)
        receiver.lambda_handler(_request(_mention_payload(), retry_num="0"), None)

        # original + explicit retry 0 dispatch; retry 1 does not.
        assert len(harness["invoked"]) == 2

    def test_non_triggering_event_types_do_not_touch_ddb(self, harness):
        payload = _mention_payload(event={"type": "reaction_added", "channel": "C1"})

        receiver.lambda_handler(_request(payload), None)

        assert harness["claims"] == []
        assert harness["invoked"] == []

    def test_bot_sourced_mentions_do_not_touch_ddb(self, harness):
        payload = _mention_payload()
        payload["event"]["bot_id"] = "B1"

        receiver.lambda_handler(_request(payload), None)

        assert harness["claims"] == []
        assert harness["invoked"] == []

    def test_url_verification_is_not_claimed(self, harness):
        response = receiver.lambda_handler(
            _request({"type": "url_verification", "challenge": "abc"}), None
        )

        assert json.loads(response["body"]) == {"challenge": "abc"}
        assert harness["claims"] == []

    def test_dedupe_key_falls_back_to_channel_and_event_ts(self):
        payload = _mention_payload()
        del payload["event_id"]

        assert receiver._event_dedupe_key(payload) == "dedupe#event#C1#1700000000.000100"

    def test_dedupe_key_is_none_without_identity(self):
        assert receiver._event_dedupe_key({"type": "event_callback", "event": {}}) is None

    def test_missing_key_uses_legacy_retry_rule(self, harness):
        assert receiver._should_dispatch(None, None) is True
        assert receiver._should_dispatch(None, "0") is True
        assert receiver._should_dispatch(None, "1") is False
        assert harness["claims"] == []


class TestInteractionDedupe:
    def _block_action(self) -> dict:
        return {
            "type": "block_actions",
            "trigger_id": "trigger-1",
            "user": {"id": "U1"},
            "channel": {"id": "C1"},
            "container": {"channel_id": "C1", "thread_ts": "thread-1"},
            "actions": [
                {
                    "action_id": "feedback_helpful",
                    "action_ts": "1700000000.1",
                    "value": json.dumps({"rating": "helpful", "answer_ts": "answer-1"}),
                }
            ],
        }

    def test_block_action_is_claimed_by_trigger_id(self, harness):
        response = receiver.lambda_handler(_interaction_request(self._block_action()), None)

        assert response["statusCode"] == 200
        assert harness["claims"][0][0] == "dedupe#interaction#trigger-1"
        assert len(harness["invoked"]) == 1

    def test_duplicate_interaction_is_acked_without_handling(self, harness, monkeypatch):
        harness["claim_result"] = False
        handled = []
        monkeypatch.setattr(
            receiver, "_handle_interaction", lambda payload: handled.append(payload) or receiver._ok()
        )

        response = receiver.lambda_handler(_interaction_request(self._block_action()), None)

        assert response["statusCode"] == 200
        assert handled == []

    def test_unknown_interaction_type_does_not_touch_ddb(self, harness):
        receiver.lambda_handler(_interaction_request({"type": "shortcut"}), None)

        assert harness["claims"] == []

    def test_interaction_key_falls_back_to_action_ts(self):
        payload = self._block_action()
        del payload["trigger_id"]

        assert receiver._interaction_dedupe_key(payload) == "dedupe#interaction#1700000000.1"


class TestClaimReleaseOnDispatchFailure:
    def test_failed_worker_invoke_releases_event_claim(self, harness, monkeypatch):
        released = []
        monkeypatch.setattr(receiver, "release_claim", lambda key: released.append(key) or True)
        monkeypatch.setattr(receiver, "_invoke_worker", lambda payload: False)

        resp = receiver.lambda_handler(_request(_mention_payload()), None)

        assert resp["statusCode"] == 500
        assert harness["claims"] and harness["claims"][0][0] == "dedupe#event#Ev123"
        assert released == ["dedupe#event#Ev123"]

    def test_successful_dispatch_does_not_release(self, harness, monkeypatch):
        released = []
        monkeypatch.setattr(receiver, "release_claim", lambda key: released.append(key) or True)

        resp = receiver.lambda_handler(_request(_mention_payload()), None)

        assert resp["statusCode"] == 200
        assert released == []

    def test_failed_interaction_referral_releases_claim(self, harness, monkeypatch):
        released = []
        monkeypatch.setattr(receiver, "release_claim", lambda key: released.append(key) or True)
        monkeypatch.setattr(
            receiver,
            "_handle_interaction",
            lambda payload: {"statusCode": 500, "body": "invoke failed"},
        )
        payload = {"type": "block_actions", "trigger_id": "T1", "actions": [{"action_ts": "1.2"}]}

        resp = receiver.lambda_handler(_interaction_request(payload), None)

        assert resp["statusCode"] == 500
        assert released == ["dedupe#interaction#T1"]
