"""Shared controls for named boolean and integer model parameters."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout, QSpinBox, QWidget


class ParameterForm(QWidget):
    changed = Signal()

    def __init__(self, options, defaults, help_text=None):
        super().__init__()
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.controls = {}
        for name, label, minimum, maximum in options:
            value = defaults[name]
            if isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
                widget.toggled.connect(self.changed)
            else:
                widget = QDoubleSpinBox() if isinstance(value, float) else QSpinBox()
                widget.setRange(minimum, maximum)
                if isinstance(value, float):
                    widget.setDecimals(2)
                    widget.setSingleStep(.01)
                widget.setValue(value)
                widget.valueChanged.connect(self.changed)
            widget.setToolTip((help_text or {}).get(name, ''))
            self.controls[name] = widget
            form.addRow(label, widget)

    @property
    def values(self):
        return {name: widget.isChecked() if isinstance(widget, QCheckBox) else widget.value()
                for name, widget in self.controls.items()}

    def set_values(self, values):
        for name, widget in self.controls.items():
            if name not in values:
                continue
            widget.blockSignals(True)
            widget.setChecked(values[name]) if isinstance(widget, QCheckBox) else widget.setValue(values[name])
            widget.blockSignals(False)
