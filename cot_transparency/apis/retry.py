from logging import Logger, getLogger
import random
import time

from functools import partial
from typing import Callable, Optional, TypeVar

from decorator import decorator

_R = TypeVar("_R")

logging_logger: Logger = getLogger(__name__)


def __retry_internal(
    f: Callable[..., _R],
    exceptions=Exception,
    tries: int = -1,
    delay: float = 0,
    max_delay: Optional[float] = None,
    backoff: float=1,
    jitter: tuple[float, float] | float = 0,
    logger: Optional[Logger]=logging_logger,
    on_retry: Optional[Callable[[], None]] = None,
) -> Optional[_R]:
    """
    Executes a function and retries it if it failed.

    :param f: the function to execute.
    :param exceptions: an exception or a tuple of exceptions to catch. default: Exception.
    :param tries: the maximum number of attempts. default: -1 (infinite).
    :param delay: initial delay between attempts. default: 0.
    :param max_delay: the maximum value of delay. default: None (no limit).
    :param backoff: multiplier applied to delay between attempts. default: 1 (no backoff).
    :param jitter: extra seconds added to delay between attempts. default: 0.
                   fixed if a number, random if a range tuple (min, max)
    :param logger: logger.warning(fmt, error, delay) will be called on failed attempts.
                   default: retry.logging_logger. if None, logging is disabled.
    :returns: the result of the f function.
    :param on_retry: callable that is triggered on every retry event.
                     default: None. if None, no callable is triggered.
    """
    _tries, _delay = tries, delay
    while _tries:
        try:
            return f()
        except exceptions as e:
            _tries -= 1
            if not _tries:
                raise

            print("retrying")
            if logger is not None:
                logger.warning("%s, retrying in %s seconds...", e, _delay)

            if on_retry is not None:
                on_retry()

            time.sleep(_delay)
            _delay *= backoff

            if isinstance(jitter, tuple):
                _delay += random.uniform(*jitter)
            else:
                _delay += jitter

            if max_delay is not None:
                _delay = min(_delay, max_delay)


def retry(
    exceptions=Exception,
    tries: int = -1,
    delay: float = 0,
    max_delay: Optional[float] = None,
    backoff: float=1,
    jitter: tuple[float, float] | float = 0,
    logger: Optional[Logger]=logging_logger,
    on_retry: Optional[Callable[[], None]] = None,
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    """Returns a retry decorator.

    :param exceptions: an exception or a tuple of exceptions to catch. default: Exception.
    :param tries: the maximum number of attempts. default: -1 (infinite).
    :param delay: initial delay between attempts. default: 0.
    :param max_delay: the maximum value of delay. default: None (no limit).
    :param backoff: multiplier applied to delay between attempts. default: 1 (no backoff).
    :param jitter: extra seconds added to delay between attempts. default: 0.
                   fixed if a number, random if a range tuple (min, max)
    :param logger: logger.warning(fmt, error, delay) will be called on failed attempts.
                   default: retry.logging_logger. if None, logging is disabled.
    :param on_retry: callable that is triggered on every retry event.
                     default: None. if None, no callable is triggered.

    :returns: a retry decorator.
    """

    @decorator
    def retry_decorator(f: Callable[..., _R], *fargs, **fkwargs) -> Optional[_R]:
        args = fargs if fargs else list()
        kwargs = fkwargs if fkwargs else dict()
        return __retry_internal(
            partial(f, *args, **kwargs),
            exceptions,
            tries,
            delay,
            max_delay,
            backoff,
            jitter,
            logger,
            on_retry,
        )

    return retry_decorator


def retry_call(
    f: Callable[..., _R],
    fargs=None,
    fkwargs=None,
    exceptions=Exception,
    tries: int = -1,
    delay: float = 0,
    max_delay: Optional[float] = None,
    backoff: float=1,
    jitter: tuple[float, float] | float = 0,
    logger: Optional[Logger]=logging_logger,
    on_retry: Optional[Callable[[], None]] = None,
) -> Optional[_R]:
    """
    Calls a function and re-executes it if it failed.

    :param f: the function to execute.
    :param fargs: the positional arguments of the function to execute.
    :param fkwargs: the named arguments of the function to execute.
    :param exceptions: an exception or a tuple of exceptions to catch. default: Exception.
    :param tries: the maximum number of attempts. default: -1 (infinite).
    :param delay: initial delay between attempts. default: 0.
    :param max_delay: the maximum value of delay. default: None (no limit).
    :param backoff: multiplier applied to delay between attempts. default: 1 (no backoff).
    :param jitter: extra seconds added to delay between attempts. default: 0.
                   fixed if a number, random if a range tuple (min, max)
    :param logger: logger.warning(fmt, error, delay) will be called on failed attempts.
                   default: retry.logging_logger. if None, logging is disabled.
    :returns: the result of the f function.
    :param on_retry: callable that is triggered on every retry event.
                     default: None. if None, no callable is triggered.
    """
    args = fargs if fargs else list()
    kwargs = fkwargs if fkwargs else dict()
    return __retry_internal(
        partial(f, *args, **kwargs),
        exceptions,
        tries,
        delay,
        max_delay,
        backoff,
        jitter,
        logger,
        on_retry,
    )