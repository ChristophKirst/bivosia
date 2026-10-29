"""Shared controls for named boolean and integer model parameters."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtWidgets import QCheckBox, QFormLayout, QSpinBox, QWidget


class ParameterForm(QWidget):
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
            else:
                widget = QSpinBox()
                widget.setRange(minimum, maximum)
                widget.setValue(value)
            widget.setToolTip((help_text or {}).get(name, ''))
            self.controls[name] = widget
            form.addRow(label, widget)

    @property
    def values(self):
        return {name: widget.isChecked() if isinstance(widget, QCheckBox) else widget.value()
                for name, widget in self.controls.items()}
