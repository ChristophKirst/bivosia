"""Shared logging setup and recording progress.

Examples
--------
>>> from bivosia.processing.log import Progress
>>> progress = Progress("Example", 3)
>>> progress.update(2)
>>> progress.update(1)
>>> progress.done
3
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from collections.abc import Callable

import logging
import time


def configure_logging(level=logging.INFO):
    """Configure terminal output once at the application entry point."""
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


class Progress:
    """Report completed work to logging and an optional UI callback."""

    def __init__(self, stage, total, *, logger=None, callback=None, cancelled=None):
        self.stage = stage
        self.total = int(total)
        self.done = 0
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.callback: Callable[[int, int], None] | None = callback
        self.cancelled = cancelled
        self.last_report = time.monotonic()
        self.check_cancelled()
        self._record("Started")

    def check_cancelled(self):
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError(f"{self.stage} cancelled")

    def _record(self, message, **context):
        self.logger.info("%s: %s (%s/%s)", self.stage, message, self.done, self.total,
                         extra={"stage": self.stage, "completed": self.done,
                                "total": self.total, **context})

    def update(self, amount):
        self.check_cancelled()
        self.done += int(amount)
        now = time.monotonic()
        if self.done == self.total or now - self.last_report >= 1.0:
            self._record("Progress")
            if self.callback is not None:
                self.callback(self.done, self.total)
            self.last_report = now
        self.check_cancelled()

    def finish(self, result_path=None):
        """Report success after writing, with an optional dataset-relative result."""
        self.check_cancelled()
        if self.done != self.total:
            raise ValueError("Cannot finish incomplete work")
        context = {} if result_path is None else {"result_path": str(result_path)}
        self._record("Saved" if result_path is not None else "Complete", **context)
