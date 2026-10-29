"""Interactive point prompts and manual correction of a single key-frame mask."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from copy import deepcopy

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from bivosia.analysis.key_frames import accept_masks, combine_masks, edited_predictions
from bivosia.analysis.mask_cleanup import clean_label_mask
from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.dataset.key_frames import MaskDraft
from bivosia.dataset.mask_cleanup import save_cleaned_key_frame
from bivosia.processing.key_frames import predict_key_frames
from bivosia.ui.colors import bgr_colors, identity_colors
from bivosia.ui.mask_canvas import MaskCanvas
from bivosia.ui.mask_tools import MaskTools
from bivosia.ui.task import Task
from bivosia.ui.video_panels import MaskDisplayControl, VideoViewGroup, image_from_bgr


class MaskEditor(QWidget):
    saved = Signal(int)
    navigate = Signal(int)
    busyChanged = Signal(bool)

    def __init__(self, predictor):
        super().__init__(None, Qt.WindowType.Window)
        self.predictor = predictor
        self.session = self.draft = self.image = None
        self.frame = None
        self.worker = None
        self.predictions = {}
        self.review = None
        self.cleanup_draft = None
        self.cleanup_parameters = None
        self.pending_model = None
        self.unavailable_reason = ""
        self.resize(1100, 760)
        layout = QVBoxLayout(self)
        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        controls_layout.addLayout(row)
        row.addWidget(QLabel("Animal"))
        self.animal = QComboBox()
        self.animal.setToolTip("Left/Right: previous/next animal, wrapping from first to last")
        self.animal.currentIndexChanged.connect(self.render)
        row.addWidget(self.animal)
        self.tools = MaskTools()
        row.addWidget(self.tools)
        row.addWidget(QLabel("Radius"))
        self.radius = QSpinBox()
        self.radius.setRange(1, 100)
        self.radius.setValue(3)
        self.radius.setToolTip("Brush radius in image pixels")
        row.addWidget(self.radius)
        self.clear_points_button = self._button("Clear points", self.clear_points, row)
        self.undo_button = self._button("Undo", lambda: self.history(False), row)
        self.undo_button.setShortcut("Ctrl+Z")
        self.redo_button = self._button("Redo", lambda: self.history(True), row)
        self.redo_button.setShortcut("Ctrl+Shift+Z")
        self.save_button = self._button("Save", self.save, row)
        self.save_button.setShortcut("Ctrl+S")
        row.addStretch()
        row = QHBoxLayout()
        controls_layout.addLayout(row)
        self.predict_button = self._button(predictor.label, self.predict, row)
        self.predict_one_button = self._button("Predict selected", self.predict_selected, row)
        self.accept_button = self._button("Accept selected", self.accept_selected, row)
        self.reject_button = self._button("Reject selected", self.reject_selected, row)
        self.approve_button = self._button("Approve all", self.approve, row)
        self.approve_button.setToolTip("Apply all remaining proposals as one undoable edit")
        self.cleanup_button = self._button("Clean up", self.clean_up, row)
        self.cleanup_button.setToolTip("Preview contour cleanup in full-resolution pixels; save a separate copy")
        self.discard_button = self._button("Discard", self.discard_preview, row)
        self.status = QLabel()
        row.addWidget(self.status, 1)
        layout.addWidget(self.controls)
        self.canvas = MaskCanvas()
        layout.addWidget(self.canvas, 1)
        self.navigation = VideoViewGroup([self.canvas])
        self.footer = QWidget()
        row = QHBoxLayout(self.footer)
        row.setContentsMargins(0, 0, 0, 0)
        previous = self._button("Previous frame", lambda: self.navigate.emit(-1), row)
        previous.setToolTip("Previous selected key frame")
        next_frame = self._button("Next frame", lambda: self.navigate.emit(1), row)
        next_frame.setToolTip("Next selected key frame")
        row.addStretch()
        self.mask_display = MaskDisplayControl()
        self.mask_display.valueChanged.connect(self.render)
        row.addWidget(self.mask_display)
        self._button("Reset zoom", lambda: self.navigation.reset(), row)
        layout.addWidget(self.footer)
        self.busyChanged.connect(self.footer.setDisabled)
        self.tools.changed.connect(self.canvas.set_tool)
        self.radius.valueChanged.connect(self.canvas.set_radius)
        self.canvas.maskClicked.connect(self.select_mask)
        self.canvas.pointAdded.connect(self.add_point)
        self.canvas.strokeStarted.connect(self.begin_stroke)
        self.canvas.strokeMoved.connect(self.paint)
        for key, step in ((Qt.Key.Key_Left, -1), (Qt.Key.Key_Right, 1)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda step=step: self.cycle_animal(step))

    @staticmethod
    def _button(text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    @property
    def busy(self):
        return self.worker is not None

    @property
    def preview_mask(self):
        return self.review.mask if self.review is not None else None

    @property
    def active_draft(self):
        return self.cleanup_draft if self.cleanup_draft is not None else (
            self.review if self.review is not None else self.draft)

    def set_predictor(self, predictor):
        """Switch models without changing the current mask or its prompts."""
        if self.busy:
            raise RuntimeError("Wait for the current prediction to finish")
        self.predictor.close()
        self.predictor = predictor
        self.predict_button.setText(predictor.label)
        self.status.clear()
        self._clear_preview()

    def set_availability(self, reason):
        self.unavailable_reason = reason
        self.predict_button.setEnabled(not reason)
        self.predict_button.setToolTip(reason or f"Predict all prompted animals · {self.predictor.device.upper()}")
        self.predict_one_button.setToolTip(reason or "Predict only the selected animal; keep other proposals")
        self.render()

    def set_frame(self, session, frame, image):
        if self.session is session and self.frame == frame:
            return True
        if not self.can_leave():
            return False
        self.session, self.frame, self.image = session, frame, image
        self.draft = session.draft(frame)
        self.pending_model = None
        self.colors = identity_colors(session.animals, .27)
        self.canvas.colors = self.colors
        self.animal.blockSignals(True)
        self.animal.clear()
        for animal in range(1, session.animals + 1):
            self.animal.addItem(str(animal), animal)
            self.animal.setItemData(animal - 1, self.colors[animal - 1], Qt.ItemDataRole.ForegroundRole)
        self.animal.addItem("Background", 0)
        self.animal.blockSignals(False)
        self._clear_preview()
        self.tools.set_mode("Prompt")
        self.navigation.reset()
        self.status.setText("")
        self.render()
        return True

    def _clear_preview(self):
        if self.preview_mask is not None:
            self.status.clear()
        self.predictions.clear()
        self.review = None
        self.render()

    def discard_preview(self):
        if self.cleanup_draft is None:
            self._clear_preview()
            return
        self.cleanup_draft = self.cleanup_parameters = None
        self.status.setText(f"{len(self.predictions)} masks to review" if self.predictions else "")
        self.render()

    def clean_up(self):
        """Create an independently editable cleanup preview without changing original masks."""
        if self.busy or self.draft is None or self.cleanup_draft is not None:
            return
        parameters = dict(ANALYSIS_DEFAULTS['mask_cleanup'])
        try:
            cleaned = clean_label_mask(self.active_draft.mask, **parameters)
        except (ValueError, TypeError) as error:
            QMessageBox.warning(self, "Clean up", str(error))
            return
        self.cleanup_draft = MaskDraft(self.active_draft.mask)
        self.cleanup_draft.replace_mask(cleaned)
        self.cleanup_parameters = parameters
        self.tools.set_mode("Select")
        self.status.setText("Cleaned preview · Save writes a separate copy")
        self.render()

    def _update_preview(self):
        self.review = MaskDraft(combine_masks(self.draft.mask, self.predictions)) if self.predictions else None
        self.status.setText(f"{len(self.predictions)} masks to review" if self.predictions else "")
        self.render()

    def select_mask(self, x, y):
        mask = self.active_draft.mask
        animal = int(mask[int(y), int(x)])
        if animal:
            self.animal.setCurrentIndex(self.animal.findData(animal))

    def reject_selected(self):
        if self.cleanup_draft is not None:
            return
        self.predictions.pop(self.animal.currentData(), None)
        self._update_preview()

    def cycle_animal(self, step):
        """Cycle through animal IDs, excluding the background entry."""
        if self.session is None or self.busy:
            return
        index = self.animal.currentIndex()
        if index >= self.session.animals:
            index = -1 if step > 0 else 0
        self.animal.setCurrentIndex((index + step) % self.session.animals)

    def add_point(self, x, y):
        if self.cleanup_draft is not None:
            return
        self.draft.add_prompt(x, y, self.animal.currentData())
        self.reject_selected()

    def clear_points(self):
        if self.cleanup_draft is not None:
            return
        self.draft.clear_prompts()
        self._clear_preview()

    def begin_stroke(self, x, y):
        self.active_draft.checkpoint()
        self.last_point = (x, y)
        self.paint(x, y)

    def paint(self, x, y):
        target = self.active_draft
        before = target.mask
        animal = 0 if self.tools.mode == "Erase" else self.animal.currentData()
        target.paint(self.last_point, (x, y), self.radius.value(), animal)
        self.last_point = (x, y)
        self._edited(before)

    def history(self, redo):
        target = self.active_draft
        before = target.mask
        target.redo() if redo else target.undo()
        self._edited(before)

    def _edited(self, before):
        if self.cleanup_draft is not None:
            self.status.setText("Cleaned preview · Save writes a separate copy")
        elif self.review is not None:
            self.predictions = edited_predictions(self.predictions, before, self.review.mask)
            self.status.setText(f"{len(self.predictions)} masks to review")
        self.render()

    def predict_selected(self):
        self.predict(animal=self.animal.currentData())

    def predict(self, *, animal=None):
        if self.busy or self.draft is None or self.unavailable_reason or self.cleanup_draft is not None:
            return
        if animal is None:
            self._clear_preview()
        else:
            self.predictions.pop(animal, None)
            self._update_preview()
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        prompts = deepcopy(self.draft.prompts)
        image_key = (str(self.session.dataset.video), self.frame)
        self.worker = Task(lambda: predict_key_frames(self.predictor, rgb, prompts, animal=animal,
                                               image_key=image_key), self)
        self.worker.ready.connect(self._prediction_ready)
        self.worker.failed.connect(self._error)
        self.worker.finished.connect(self._prediction_finished)
        self.controls.setEnabled(False)
        self.canvas.setEnabled(False)
        self.busyChanged.emit(True)
        self.status.setText(self.predictor.label + "…")
        self.worker.start()

    def _prediction_ready(self, result):
        self.predictions.update(result)
        self._update_preview()
        self.tools.set_mode("Select")

    def _prediction_finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.controls.setEnabled(True)
        self.canvas.setEnabled(True)
        self.busyChanged.emit(False)
        self.render()

    def _error(self, message):
        self.status.setText("Prediction failed")
        QMessageBox.warning(self, self.predictor.label, message)

    def accept_selected(self):
        self.approve(animal=self.animal.currentData())

    def approve(self, *, animal=None):
        if self.preview_mask is None or self.busy or self.cleanup_draft is not None:
            return
        animals = list(self.predictions) if animal is None else [animal]
        if any(label not in self.predictions for label in animals):
            return
        self.draft.replace_mask(accept_masks(self.draft.mask, self.preview_mask, animals))
        for label in animals:
            self.predictions.pop(label)
        self.pending_model = self.predictor.metadata
        self._update_preview()
        if not self.predictions:
            self.status.setText("Approved · Save to write")

    def render(self):
        if self.draft is None:
            return
        reviewing = self.preview_mask is not None
        cleaning = self.cleanup_draft is not None
        mask = self.active_draft.mask
        self.canvas.prompts = self.draft.prompts
        self.canvas.set_selection(mask, self.animal.currentData())
        self.canvas.set_image(image_from_bgr(self.mask_display.composite(
            self.image, mask, None, bgr_colors(self.colors),
        )))
        self.approve_button.setEnabled(reviewing and not cleaning)
        self.accept_button.setEnabled(not cleaning and self.animal.currentData() in self.predictions)
        self.reject_button.setEnabled(not cleaning and self.animal.currentData() in self.predictions)
        self.predict_button.setEnabled(not cleaning and not self.unavailable_reason)
        self.predict_one_button.setEnabled(not cleaning and not self.unavailable_reason and self.animal.currentData() > 0)
        self.clear_points_button.setEnabled(not cleaning)
        self.tools.modes['Prompt'].setEnabled(not cleaning)
        self.cleanup_button.setEnabled(not cleaning)
        self.discard_button.setEnabled(cleaning or reviewing)
        self.save_button.setEnabled(cleaning or not reviewing)
        self.save_button.setText("Save cleaned" if cleaning else "Save")
        self.undo_button.setEnabled(bool(self.active_draft.undo_stack))
        self.redo_button.setEnabled(bool(self.active_draft.redo_stack))
        changed = self.cleanup_draft.dirty if cleaning else self.draft.dirty
        self.setWindowTitle(("Cleaned mask" if cleaning else "Mask") + f" · {self.frame:,}" + (" *" if changed else ""))

    def save(self):
        if self.draft is None or self.busy:
            return False
        if self.cleanup_draft is not None:
            try:
                path = save_cleaned_key_frame(self.session.dataset, self.frame, self.cleanup_draft.mask,
                                             frame_count=self.session.frame_count, animals=self.session.animals,
                                             parameters=self.cleanup_parameters)
            except Exception as error:
                QMessageBox.warning(self, "Save cleaned", str(error))
                return False
            self.cleanup_draft.mark_saved()
            self.status.setText(f"Saved · {path.name}")
            self.render()
            return True
        if self.preview_mask is not None:
            self.status.setText("Approve or discard the preview before saving")
            return False
        try:
            self.session.save_frame(self.frame, self.draft, self.pending_model)
        except Exception as error:
            QMessageBox.warning(self, "Save", str(error))
            return False
        self.draft.mark_saved()
        self.status.setText("Saved")
        self.render()
        self.saved.emit(self.frame)
        return True

    def can_leave(self):
        if self.busy:
            QMessageBox.information(self, "Prediction", "Wait for the current prediction to finish")
            return False
        if self.cleanup_draft is not None:
            if self.cleanup_draft.dirty:
                answer = QMessageBox.question(self, "Save cleaned", "Save the cleaned masks as a separate copy?",
                                              QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                              QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
                if answer == QMessageBox.StandardButton.Cancel:
                    return False
                if answer == QMessageBox.StandardButton.Save and not self.save():
                    return False
            self.discard_preview()
        if self.preview_mask is not None:
            answer = QMessageBox.question(self, "Review", "Discard the unapproved mask preview?",
                                          QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                                          QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Discard:
                return False
            self._clear_preview()
        if self.draft is None or not self.draft.dirty:
            return True
        answer = QMessageBox.question(self, "Save", "Save changes to this frame?",
                                      QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                      QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        if answer == QMessageBox.StandardButton.Discard:
            self.draft = self.session.draft(self.frame)
            self.pending_model = None
            self._clear_preview()
            return True
        return False

    def closeEvent(self, event):
        if self.can_leave():
            event.accept()
        else:
            event.ignore()
