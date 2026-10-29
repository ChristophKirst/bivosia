"""Spawn-based block processing for recording analysis."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from collections.abc import Callable, Iterable, Iterator
from typing import Any, TypeVar

import multiprocessing as mp
import os


T = TypeVar("T")
R = TypeVar("R")


def worker_count(requested: int | None, task_count: int) -> int:
    """Resolve a safe worker count without creating idle processes."""
    if requested is not None and requested < 1:
        raise ValueError("workers must be positive")
    return max(1, min(requested or os.cpu_count() or 1, max(1, task_count)))


def unordered_blocks(
    function: Callable[[T], R],
    tasks: Iterable[T],
    *,
    workers: int,
    initializer: Callable[..., Any] | None = None,
    initargs: tuple[Any, ...] = (),
) -> Iterator[R]:
    """Yield block results from a spawn-based process pool."""
    if workers == 1:
        if initializer is not None:
            initializer(*initargs)
        yield from map(function, tasks)
        return
    with mp.get_context("spawn").Pool(
        workers, initializer=initializer, initargs=initargs
    ) as pool:
        yield from pool.imap_unordered(function, tasks, chunksize=1)
