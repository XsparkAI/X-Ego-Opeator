from __future__ import annotations

import contextlib
import threading

_limit_lock = threading.Lock()
_limit_value = 1
_limit_semaphore = threading.BoundedSemaphore(_limit_value)
_cpu_limit_lock = threading.Lock()
_cpu_limit_value = 1
_cpu_limit_semaphore = threading.BoundedSemaphore(_cpu_limit_value)


def set_vlm_global_limit(limit: int) -> None:
    global _limit_value, _limit_semaphore
    limit = max(1, int(limit))
    with _limit_lock:
        _limit_value = limit
        _limit_semaphore = threading.BoundedSemaphore(limit)


def get_vlm_global_limit() -> int:
    with _limit_lock:
        return _limit_value


@contextlib.contextmanager
def vlm_api_slot():
    with _limit_lock:
        sem = _limit_semaphore
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


def set_cpu_global_limit(limit: int) -> None:
    global _cpu_limit_value, _cpu_limit_semaphore
    limit = max(1, int(limit))
    with _cpu_limit_lock:
        _cpu_limit_value = limit
        _cpu_limit_semaphore = threading.BoundedSemaphore(limit)


def get_cpu_global_limit() -> int:
    with _cpu_limit_lock:
        return _cpu_limit_value


@contextlib.contextmanager
def cpu_task_slot():
    with _cpu_limit_lock:
        sem = _cpu_limit_semaphore
    sem.acquire()
    try:
        yield
    finally:
        sem.release()
