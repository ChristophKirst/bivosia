"""Check worker delivery, window destruction and serialized continuations."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from threading import Event
import unittest

from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QThread, QTimer
from PySide6.QtWidgets import QApplication, QDialog, QWidget
from shiboken6 import delete, isValid

from bivosia.ui.task import LoggedTask, Task
from bivosia.ui.task_monitor import TaskMonitor


class TaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def finish(self, task):
        loop = QEventLoop()
        complete = []
        task.finished.connect(lambda: (complete.append(True), loop.quit()))
        QTimer.singleShot(3000, loop.quit)
        loop.exec()
        self.assertTrue(complete, 'Worker did not finish')
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertFalse(isValid(task))

    def test_success_failure_and_monitor_finish_before_continuation(self):
        owner = QWidget()
        monitor = TaskMonitor(owner)
        received = []
        task = LoggedTask(lambda: 42, owner)
        monitor.follow(task)
        task.connect_callbacks(ready=lambda result: received.append((result, QThread.currentThread())),
                               finished=lambda: received.append(monitor.job_status.text()))
        task.start()
        self.finish(task)
        self.assertEqual(received, [(42, self.app.thread()), 'Complete'])
        def fail():
            raise ValueError('expected failure')
        task = Task(fail, owner)
        task.connect_callbacks(failed=received.append, finished=lambda: received.append('finished'))
        with self.assertLogs('bivosia.ui.task', level='ERROR'):
            task.start()
            self.finish(task)
        self.assertEqual(received[-2:], ['expected failure', 'finished'])
        delete(owner)

    def test_destroyed_owner_suppresses_callbacks_without_destroying_worker(self):
        owner = QWidget()
        release = Event()
        received = []
        task = Task(lambda: release.wait(2), owner)
        task.connect_callbacks(ready=received.append, failed=received.append,
                               finished=lambda: received.append('finished'))
        task.start()
        delete(owner)
        self.assertTrue(isValid(task))
        self.assertTrue(task.isInterruptionRequested())
        release.set()
        self.finish(task)
        self.assertEqual(received, [])

    def test_modal_owner_can_be_released_before_worker_deletion(self):
        destroyed = []
        class Dialog(QDialog):
            def complete(self):
                self.accept()
        def run_dialog():
            dialog = Dialog()
            dialog.destroyed.connect(lambda: destroyed.append('dialog'))
            task = Task(lambda: None, dialog)
            task.destroyed.connect(lambda: destroyed.append('task'))
            task.connect_callbacks(finished=dialog.complete)
            task.start()
            return dialog.exec()
        self.assertEqual(run_dialog(), QDialog.DialogCode.Accepted)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertCountEqual(destroyed, ['dialog', 'task'])
