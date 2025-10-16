import asyncio
import logging
from typing import Callable, Iterable, Tuple, Type, Any

logger = logging.getLogger(__name__)

def is_transient_pg_error(sqlstate: str | None) -> bool:
    return sqlstate in {"40001", "40P01"}  # serialization_failure, deadlock_detected

async def retry_async(
    func: Callable[..., Any],
    *args,
    retries: int = 5,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    retry_exceptions: Iterable[Type[BaseException]] = (),
    retry_on_predicate: Callable[[BaseException], bool] | None = None,
    **kwargs,
):
    attempt = 0
    delay = initial_delay
    while True:
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            attempt += 1
            should_retry = False
            if any(isinstance(exc, t) for t in retry_exceptions):
                should_retry = True
            if retry_on_predicate and retry_on_predicate(exc):
                should_retry = True
            if not should_retry or attempt > retries:
                logger.exception("Operation failed and will not retry")
                raise
            logger.warning("Transient error detected, retrying attempt %d/%d after %.2fs: %s", attempt, retries, delay, exc)
            await asyncio.sleep(delay)
            delay *= backoff_factor
