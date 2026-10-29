"""Execute a callable off the GUI thread and deliver its result through signals."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import logging

from PySide6.QtCore import QThread, Signal


class Task(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

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
