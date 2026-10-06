"""Invocation-wide wall-clock deadline shared by the agent loop and tools.

The worker sets a deadline from ``context.get_remaining_time_in_millis()`` at
the top of each Lambda invocation. Long-running tool calls (Databricks polling,
retry backoff) consult ``remaining_seconds()`` so they stop before Lambda hard
kills the container. Stored in a ``ContextVar`` so it is invocation-scoped.
"""

from __future__ import annotations

import time
import signal
import threading
from contextlib import contextmanager
from contextvars import ContextVar

_deadline_epoch: ContextVar[float | None] = ContextVar("deadline_epoch", default=None)


class DeadlineExceeded(TimeoutError):
    """The invocation's reserved completion deadline was reached."""


class _AlarmExpired(BaseException):
    """Escape SDK retry loops, which catch Exception (including TimeoutError)."""


@contextmanager
def enforce_remaining(max_seconds: float | None = None):
    """Interrupt a synchronous model operation at the invocation deadline.

    Lambda's Python handler executes on the main thread on Linux. A real-time
    alarm bounds SDK retries, retry sleeps, and an otherwise healthy stream
    that keeps emitting tokens forever. Socket read timeouts alone do not.
    Offline callers without an invocation deadline require no signal support.
    """
    remaining = remaining_seconds()
    if max_seconds is not None:
        remaining = max_seconds if remaining is None else min(remaining, max_seconds)
    if remaining is None:
        yield
        return
    if remaining <= 0:
        raise DeadlineExceeded("Invocation deadline reached")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Invocation deadline enforcement requires the main thread")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    # Do not replace an independent alarm whose handler we do not own.
    if previous_timer[0] > 0:
        raise RuntimeError("Cannot enforce invocation deadline over an existing alarm")

    def expired_alarm(signum, frame):
        raise _AlarmExpired()

    signal.signal(signal.SIGALRM, expired_alarm)
    try:
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            yield
        except _AlarmExpired:
            raise DeadlineExceeded("Invocation deadline reached") from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def set_deadline(epoch_seconds: float | None) -> None:
    """Set the absolute wall-clock deadline (``time.time()`` scale) or clear it."""
    _deadline_epoch.set(epoch_seconds)


def set_deadline_from_remaining(remaining_seconds: float, *, safety_margin: float = 0.0) -> None:
    """Convenience: set the deadline ``remaining_seconds - safety_margin`` from now."""
    set_deadline(time.time() + max(0.0, remaining_seconds - safety_margin))


def clear() -> None:
    _deadline_epoch.set(None)


def deadline() -> float | None:
    return _deadline_epoch.get()


def remaining_seconds() -> float | None:
    """Seconds until the deadline, ``None`` if no deadline is set. Never negative."""
    dl = _deadline_epoch.get()
    if dl is None:
        return None
    return max(0.0, dl - time.time())


def expired() -> bool:
    rem = remaining_seconds()
    return rem is not None and rem <= 0.0


def bounded_timeout(requested: float, *, floor: float = 1.0) -> float:
    """Clamp a per-request timeout so it cannot outlive the invocation deadline."""
    rem = remaining_seconds()
    if rem is None:
        return requested
    return max(floor, min(requested, rem))
