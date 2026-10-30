"""Choose file versions before consolidating a dataset into one location."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QHeaderView,
                               QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout)


class ConsolidationDialog(QDialog):
    def __init__(self, plan, parent=None, *, action=None, scope=None):
        super().__init__(parent)
        self.setWindowTitle(f'{action} folder' if action else
                            'Consolidate dataset' if plan.complete else 'Sync processing inputs')
        self.resize(850, 400)
        self.selectors = {}
        layout = QVBoxLayout(self)
        target = (plan.location['host'] + ':' if plan.location['host'] else '') + plan.location['path']
        scope = scope or ('all dataset files' if plan.complete else 'required processing inputs')
        retention = ('Move removes source copies identical to the chosen version; other versions remain.'
                     if action == 'Move' else 'Source copies remain in place.')
        label = QLabel(f'{action or "Copy"} {scope} to {target}\nChoose which copy to keep for each conflict.\n'
                       + ('The destination description will use local, relative paths. ' if plan.complete else '')
                       + retention)
        label.setWordWrap(True)
        layout.addWidget(label)
        table = QTableWidget(len(plan.conflicts), 2)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setHorizontalHeaderLabels(('Conflicting file', 'Use this copy'))
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for row, name in enumerate(plan.conflicts):
            table.setItem(row, 0, QTableWidgetItem(name))
            selector = QComboBox()
            for entry in plan.files[name]:
                label = (entry.host + ':' if entry.host else '') + str(entry.path)
                selector.addItem(label)
            selector.setCurrentIndex(plan.defaults[name])
            table.setCellWidget(row, 1, selector)
            self.selectors[name] = selector
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(action or
                                                                  ('Consolidate' if plan.complete else 'Sync and continue'))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def choices(self):
        return {name: selector.currentIndex() for name, selector in self.selectors.items()}
