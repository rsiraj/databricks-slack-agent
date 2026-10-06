import time

import pytest

from utils import deadline


@pytest.fixture(autouse=True)
def _clear_deadline():
    deadline.clear()
    yield
    deadline.clear()


def test_no_deadline_by_default():
    assert deadline.deadline() is None
    assert deadline.remaining_seconds() is None
    assert deadline.expired() is False
    assert deadline.bounded_timeout(120.0) == 120.0


def test_set_from_remaining_applies_safety_margin():
    deadline.set_deadline_from_remaining(100.0, safety_margin=40.0)
    rem = deadline.remaining_seconds()
    assert 59.0 < rem <= 60.0
    assert deadline.expired() is False


def test_safety_margin_larger_than_remaining_clamps_to_zero():
    deadline.set_deadline_from_remaining(10.0, safety_margin=30.0)
    assert deadline.remaining_seconds() == 0.0
    assert deadline.expired() is True


def test_bounded_timeout_clamps_to_remaining_and_floor():
    deadline.set_deadline(time.time() + 30.0)
    assert 29.0 < deadline.bounded_timeout(120.0) <= 30.0
    assert deadline.bounded_timeout(10.0) == 10.0

    deadline.set_deadline(time.time() + 0.1)
    assert deadline.bounded_timeout(120.0, floor=5.0) == 5.0


def test_remaining_never_negative_after_expiry():
    deadline.set_deadline(time.time() - 5.0)
    assert deadline.remaining_seconds() == 0.0
    assert deadline.expired() is True


def test_clear_resets():
    deadline.set_deadline(time.time() + 100.0)
    deadline.clear()
    assert deadline.remaining_seconds() is None


def test_total_deadline_interrupts_exception_retry_loop_and_restores_alarm():
    import signal

    old_handler = signal.getsignal(signal.SIGALRM)
    deadline.set_deadline_from_remaining(0.03)
    started = time.monotonic()
    with pytest.raises(deadline.DeadlineExceeded):
        with deadline.enforce_remaining():
            # An SDK may swallow/retry ordinary timeout exceptions. The alarm
            # must escape that loop and still become a normal application error.
            while True:
                try:
                    time.sleep(0.5)
                except Exception:
                    pass
    assert time.monotonic() - started < 0.5
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    assert signal.getsignal(signal.SIGALRM) == old_handler


def test_total_deadline_clears_alarm_on_success_and_error():
    import signal

    deadline.set_deadline_from_remaining(60)
    with deadline.enforce_remaining():
        pass
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    with pytest.raises(ValueError):
        with deadline.enforce_remaining():
            raise ValueError("provider failed")
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_total_deadline_interrupts_actual_sdk_retries(monkeypatch):
    import anthropic
    import httpx2
    from utils.llm.anthropic_provider import AnthropicProvider
    from utils.llm.types import ModelRequest

    attempts = []

    def timeout(request):
        attempts.append(request)
        raise httpx2.ReadTimeout("synthetic timeout", request=request)

    client = anthropic.Anthropic(
        api_key="test", max_retries=5,
        http_client=httpx2.Client(transport=httpx2.MockTransport(timeout)),
    )
    provider = AnthropicProvider("claude-sonnet-5", client=client, stream="auto")
    deadline.set_deadline_from_remaining(0.04)
    try:
        with pytest.raises(deadline.DeadlineExceeded):
            with deadline.enforce_remaining():
                provider.create(ModelRequest(
                    system="Test", messages=[{"role": "user", "content": "hi"}],
                    tools=[], effort="high", max_tokens=24000, timeout_s=180,
                ))
        assert len(attempts) == 1
    finally:
        client.close()
