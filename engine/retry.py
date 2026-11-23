"""Tiny retry-with-exponential-backoff helper (no extra dependency)."""
from __future__ import annotations

import functools
import logging
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)


def retry(
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] | None = None,
):
    """Retry the wrapped call up to ``attempts`` times.

    Delay before retry ``n`` (1-based) is ``base_delay * 2**(n-1)`` plus up
    to 10% jitter, capped at ``max_delay``. The last exception is re-raised.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:  # noqa: PERF203
                    if attempt == attempts:
                        raise
                    delay = min(max_delay, base_delay * 2 ** (attempt - 1))
                    delay += random.uniform(0, delay * 0.1)
                    log.warning("%s failed (%s); retry %d/%d in %.1fs",
                                fn.__name__, exc, attempt, attempts - 1, delay)
                    (sleep or time.sleep)(delay)
            raise AssertionError("unreachable")

        return wrapper

    return decorator
