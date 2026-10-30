"""Reusable analysis parameter panels for editors and workflow settings."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from bivosia.config import AUTO_DIRECTION_METHODS, ANALYSIS_DEFAULTS, CONSENSUS_MASK_SOURCES
from bivosia.ui.parameters import ParameterForm
from bivosia.ui.step_settings import step_store

SHAPE_OPTIONS = (
    ('start_frame', 'Start frame', 0, 2147483647),
    ('end_frame', 'End frame', -1, 2147483647),
    ('frame_step', 'Every Nth frame', 1, 2147483647),
    ('downsample_factor', 'Downsample factor', 1, 2147483647),
    ('preprocess_masks', 'Preprocess masks', 0, 1),
    ('filter_by_area', 'Filter by area', 0, 1),
    ('main_body_area', 'Main body area (pixels²)', 0, 2147483647),
    ('filter_by_distance', 'Filter by distance', 0, 1),
    ('fragment_distance', 'Fragment distance (pixels)', 0, 2147483647),
    ('apply_opening', 'Morphological opening', 0, 1),
    ('opening_radius', 'Opening radius (pixels)', 0, 2147483647),
    ('degree', 'Zernike degree', 0, 2147483647),
    ('radius', 'Zernike radius (pixels)', 0, 2147483647),
    ('n_components', 'UMAP dimensions', 2, 3),
    ('z_score', 'Z-score features before UMAP', 0, 1),
)

SHAPE_HELP = {
    'start_frame': 'First frame included in feature extraction (zero-based). Invalid masks are excluded.',
    'end_frame': 'Last included frame. Last uses the end of the recording.',
    'frame_step': 'Select frames from Start frame at this interval before feature extraction.',
    'downsample_factor': '1 keeps the source resolution; 2 halves width and height. Nearest-neighbor resizing runs before preprocessing. All pixel-based settings use this downsampled resolution.',
    'preprocess_masks': 'Select body components and remove thin protrusions before shape analysis. Originals are unchanged.',
    'filter_by_area': 'Keep only the largest component if its area exceeds Main body area; otherwise leave components unchanged.',
    'main_body_area': 'Keep only the largest component when its area in downsampled pixels exceeds this threshold.',
    'filter_by_distance': 'Keep at most the two largest remaining components when nearby; otherwise keep only the largest.',
    'fragment_distance': 'Keep the two largest remaining components when their centroids are within this distance in downsampled pixels.',
    'apply_opening': 'Remove thin protrusions using erosion followed by dilation with a disk.',
    'opening_radius': 'Disk radius for morphological opening in downsampled pixels. Zero skips opening.',
    'degree': 'Maximum degree of the absolute Zernike moments. Mahotas defaults to 8.',
    'radius': 'Fixed disk radius around each mask centroid, in downsampled pixels. It must contain every foreground pixel.',
    'n_components': 'Calculate a separate 2D or 3D UMAP using the same Zernike features.',
    'z_score': 'Center and scale each feature column across all samples before standard UMAP.',
}


class AnalysisParameters(QWidget):
    """Edit the same saved step settings in an editor or a standalone panel."""

    changed = Signal()

    def __init__(self, dataset, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.store = step_store(dataset)
        self._updating = False
        self.editable = True
        values = self.store.get(operation)
        options = {
            'quantify': (('n_animals', 'Animals', 2, 2147483647), ('workers', 'Workers', 0, 2147483647)),
            'align': (('n_animals', 'Animals', 1, 2147483647), ('workers', 'Workers', 0, 2147483647),
                      ('fallback_previous', 'Reuse previous assignments for ties', 0, 1)),
            'detect_link': (('theta_quality', 'Seed', -1., 1.), ('theta_recovery', 'Recovery', -1., 1.),
                            ('theta_gap', 'Gap', 0, 2147483647)),
            'shape_features': SHAPE_OPTIONS,
            'export': (),
        }[operation]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.form = ParameterForm(options, values, SHAPE_HELP if operation == 'shape_features' else {
            'workers': 'Automatic uses available CPUs, limited by the number of blocks.'})
        self.controls = self.form.controls
        if 'workers' in self.controls:
            self.controls['workers'].setSpecialValueText('Automatic')
        if 'radius' in self.controls:
            self.controls['radius'].setSpecialValueText('Select')
        if 'end_frame' in self.controls:
            self.controls['end_frame'].setSpecialValueText('Last')
        layout.addWidget(self.form)
        self.direction_method = self.destination = self.nodes = self.mask_source = None
        if operation == 'shape_features':
            self.mask_source = QComboBox()
            for key, definition in CONSENSUS_MASK_SOURCES.items():
                self.mask_source.addItem(definition['label'], key)
            self.mask_source.setToolTip('Automatic uses a matching export when available, otherwise masks plus annotation.')
            self.form.layout().insertRow(0, 'Mask source', self.mask_source)
            self.mask_source.currentIndexChanged.connect(self._edited)
        if operation == 'detect_link':
            self.direction_method = QComboBox()
            for key, label in AUTO_DIRECTION_METHODS.items():
                self.direction_method.addItem(label, key)
            self.form.layout().addRow('Direction method', self.direction_method)
            self.direction_method.currentIndexChanged.connect(self._edited)
        if operation == 'export':
            form = QFormLayout()
            self.destination = QLineEdit()
            self.destination.setPlaceholderText('Use the selected annotation’s linked-mask filename')
            form.addRow('Output file', self.destination)
            layout.addLayout(form)
            self.destination.editingFinished.connect(self._edited)
        if operation == 'align':
            layout.addWidget(QLabel('Keypoints'))
            self.nodes = QListWidget()
            layout.addWidget(self.nodes)
            self.nodes.itemChanged.connect(self._edited)
        self.set_values(values)
        self.form.changed.connect(self._edited)
        self.store.changed.connect(self._shared_changed)

    @property
    def values(self):
        values = self.form.values
        if self.mask_source is not None:
            values['mask_source'] = self.mask_source.currentData()
        if self.direction_method is not None:
            values['direction_method'] = self.direction_method.currentData()
        if self.destination is not None:
            values['destination'] = self.destination.text().strip()
        if self.nodes is not None:
            values['exclude_keypoints'] = ([self.nodes.item(i).text() for i in range(self.nodes.count())
                                            if self.nodes.item(i).checkState() != Qt.CheckState.Checked]
                                           if self.nodes.count() else self.store.get(self.operation)['exclude_keypoints'])
        return values

    def set_values(self, values):
        self._updating = True
        self.form.set_values(values)
        if self.mask_source is not None:
            self.mask_source.setCurrentIndex(self.mask_source.findData(values['mask_source']))
        if self.direction_method is not None:
            self.direction_method.setCurrentIndex(self.direction_method.findData(values['direction_method']))
        if self.destination is not None:
            self.destination.setText(values.get('destination', ''))
        if self.nodes is not None:
            for i in range(self.nodes.count()):
                item = self.nodes.item(i)
                item.setCheckState(Qt.CheckState.Unchecked if item.text() in values['exclude_keypoints'] else Qt.CheckState.Checked)
        self._updating = False
        self._enabled()

    def set_nodes(self, names):
        self._updating = True
        self.nodes.clear()
        for name in names:
            item = QListWidgetItem(name, self.nodes)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        self._updating = False
        self.set_values(self.store.get(self.operation))

    def _enabled(self):
        if self.operation != 'shape_features':
            return
        enabled = self.controls['preprocess_masks'].isChecked()
        for name in ANALYSIS_DEFAULTS['masks']['preprocessing']:
            self.controls[name].setEnabled(enabled)
        for step, parameter in (('filter_by_area', 'main_body_area'), ('filter_by_distance', 'fragment_distance'),
                                ('apply_opening', 'opening_radius')):
            self.controls[parameter].setEnabled(enabled and self.controls[step].isChecked())

    def _edited(self, *args):
        if not self._updating:
            self._enabled()
            self.store.set(self.operation, self.values)
            self.changed.emit()

    def _shared_changed(self, operation, values):
        if self.editable and operation == self.operation and values != self.values:
            self.set_values(values)
            self.changed.emit()

    def set_dataset(self, dataset):
        self.store.changed.disconnect(self._shared_changed)
        self.store = step_store(dataset)
        self.set_values(self.store.get(self.operation))
        self.store.changed.connect(self._shared_changed)

    def set_editable(self, enabled):
        self.editable = enabled
        self.setEnabled(enabled)
        if enabled:
            self.set_values(self.store.get(self.operation))
