"""Small, testable job coordinators for desktop background work."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar


T = TypeVar("T")
_EMPTY = object()


@dataclass(frozen=True)
class JobSubmission:
    started_immediately: bool
    replaced_pending: bool = False


class LatestJobRunner(Generic[T]):
    """Run one job at a time and keep only the latest pending request."""

    def __init__(
        self,
        handler: Callable[[T, threading.Event], None],
        *,
        name: str = "pascribe-latest-job",
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._handler = handler
        self._name = name
        self._on_error = on_error
        self._lock = threading.Lock()
        self._pending: object = _EMPTY
        self._current_cancel: threading.Event | None = None
        self._active = False
        self._idle = threading.Event()
        self._idle.set()

    def submit(self, payload: T) -> JobSubmission:
        with self._lock:
            if self._active:
                if self._current_cancel is not None:
                    self._current_cancel.set()
                replaced = self._pending is not _EMPTY
                self._pending = payload
                return JobSubmission(False, replaced)

            self._active = True
            self._pending = payload
            self._idle.clear()
            worker = threading.Thread(target=self._run, name=self._name, daemon=True)
            worker.start()
            return JobSubmission(True)

    def _run(self) -> None:
        while True:
            with self._lock:
                payload = self._pending
                self._pending = _EMPTY
                if payload is _EMPTY:
                    self._active = False
                    self._current_cancel = None
                    self._idle.set()
                    return
                cancel_event = threading.Event()
                self._current_cancel = cancel_event

            try:
                self._handler(payload, cancel_event)  # type: ignore[arg-type]
            except Exception as exc:
                if self._on_error is not None:
                    try:
                        self._on_error(exc)
                    except Exception:
                        pass

            with self._lock:
                self._current_cancel = None
                if self._pending is _EMPTY:
                    self._active = False
                    self._idle.set()
                    return

    def cancel(self, *, clear_pending: bool = True) -> bool:
        with self._lock:
            if not self._active:
                return False
            if self._current_cancel is not None:
                self._current_cancel.set()
            if clear_pending:
                self._pending = _EMPTY
            return True

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def wait(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)


class ExclusiveJobRunner:
    """Start at most one cancellable instance of a background job."""

    def __init__(self, *, name: str = "pascribe-exclusive-job") -> None:
        self._name = name
        self._lock = threading.Lock()
        self._cancel_event: threading.Event | None = None
        self._active = False
        self._idle = threading.Event()
        self._idle.set()

    def start(
        self,
        handler: Callable[[threading.Event], None],
        *,
        on_error: Callable[[Exception], None] | None = None,
    ) -> bool:
        with self._lock:
            if self._active:
                return False
            self._active = True
            self._idle.clear()
            cancel_event = threading.Event()
            self._cancel_event = cancel_event

        def run() -> None:
            try:
                handler(cancel_event)
            except Exception as exc:
                if on_error is not None:
                    try:
                        on_error(exc)
                    except Exception:
                        pass
            finally:
                with self._lock:
                    self._active = False
                    self._cancel_event = None
                    self._idle.set()

        threading.Thread(target=run, name=self._name, daemon=True).start()
        return True

    def cancel(self) -> bool:
        with self._lock:
            if self._cancel_event is None:
                return False
            self._cancel_event.set()
            return True

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def wait(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)
