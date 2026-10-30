"""Execute a callable off the GUI thread and deliver its result through signals."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import logging
from threading import get_ident

from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal, Slot


class Task(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, function, owner):
        # Workers outlive their windows so file operations can finish safely.
        application = QCoreApplication.instance()
        super().__init__(application)
        self.function = function
        self._owner_alive = True
        self._ready = self._failed = self._finished = None
        self._owner_connection = owner.destroyed.connect(self._owner_destroyed)
        application.aboutToQuit.connect(self.wait)
        self.finished.connect(self.deleteLater)

    def connect_callbacks(self, *, ready=None, failed=None, finished=None):
        """Deliver callbacks on the GUI thread while the requesting owner exists."""
        self._ready, self._failed, self._finished = ready, failed, finished
        self.ready.connect(self._deliver_ready)
        self.failed.connect(self._deliver_failed)
        self.finished.connect(self._deliver_finished)

    @Slot()
    def _owner_destroyed(self):
        self._owner_alive = False
        self.requestInterruption()

    @Slot(object)
    def _deliver_ready(self, result):
        if self._owner_alive and not self.isInterruptionRequested() and self._ready is not None:
            self._ready(result)

    @Slot(str)
    def _deliver_failed(self, message):
        if self._owner_alive and not self.isInterruptionRequested() and self._failed is not None:
            self._failed(message)

    @Slot()
    def _deliver_finished(self):
        try:
            if self._owner_alive and self._finished is not None:
                self._finished()
        finally:
            # Releasing callbacks can destroy their window. Disconnect it while
            # this worker's Python wrapper is still alive, before deleteLater.
            QObject.disconnect(self._owner_connection)
            self._ready = self._failed = self._finished = self.function = None

    def run(self):
        try:
            result = self.function()
            if not self.isInterruptionRequested():
                self.ready.emit(result)
        except Exception as error:
            if self.isInterruptionRequested():
                return
            logging.getLogger(__name__).exception("Background operation failed")
            self.failed.emit(str(error))


class LoggedTask(Task):
    """Deliver logging records emitted by this worker thread to its monitor."""

    logged = Signal(object)

    def run(self):
        thread = get_ident()
        handler = logging.Handler()
        handler.addFilter(lambda record: record.thread == thread)
        handler.emit = self.logged.emit
        logger = logging.getLogger()
        logger.addHandler(handler)
        try:
            super().run()
        finally:
            logger.removeHandler(handler)
            handler.close()
