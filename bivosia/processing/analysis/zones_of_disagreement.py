"""Detect ZoDs and save automatic direction choices and identity links."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import closing
from dataclasses import replace
import uuid

from bivosia.config import ANALYSIS_DEFAULTS, AUTOMATIC_ANNOTATION_PREFIX
from bivosia.analysis.mask_linking import automatic_link
from bivosia.analysis.zones_of_disagreement import detect_zods, detection_parameters, mask_count_seed_signal
from bivosia.dataset.annotations.masks import AnnotationInputs, RevisionSession
from bivosia.dataset.masks import mask_pair
from bivosia.processing.log import Progress


def detect_link_job(dataset, *, theta_quality=ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_quality'],
                    theta_recovery=ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_recovery'],
                    theta_gap=ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_gap'],
                    direction_method=ANALYSIS_DEFAULTS['zones_of_disagreement']['direction_method'],
                    progress=None, cancelled=None):
    with closing(AnnotationInputs(dataset)) as data, mask_pair(
            dataset.mask('forward'), dataset.mask('reverse')) as (forward, reverse):
        sources = {'forward': forward, 'reverse': reverse}
        if forward.shape[0] != data.count:
            raise ValueError('Masks and analysis frame counts differ')
        session = create_automatic_annotation(
            dataset, data, lambda source, frame: sources[source][frame],
            theta_quality=theta_quality, theta_recovery=theta_recovery, theta_gap=theta_gap,
            direction_method=direction_method, progress=progress, cancelled=cancelled)
        return [str(session.path)]


def create_automatic_annotation(dataset, data, mask, **settings):
    """Generate and save a new annotation without overwriting existing files."""
    session = prepare_automatic_annotation(dataset, data, mask, **settings)
    session.save()
    dataset.select_annotation(session.path)
    return session


def prepare_automatic_annotation(dataset, data, mask, *, path=None,
                                theta_quality=ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_quality'],
                                theta_recovery=ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_recovery'],
                                theta_gap=ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_gap'],
                                direction_method=ANALYSIS_DEFAULTS['zones_of_disagreement']['direction_method'],
                                progress=None, cancelled=None):
    """Detect zones and propose directions and links for a new annotation.

    ``data`` supplies loaded mask-analysis evidence and missing-mask counts.
    ``mask(direction, frame)`` reads the corresponding local or remote mask.
    The returned session is unsaved; cancellation leaves existing files intact.
    """
    callback = None if progress is None else lambda done, total: progress({'completed': done, 'total': total})
    path = (dataset.new_annotation(theta_quality, theta_recovery,
                prefix=AUTOMATIC_ANNOTATION_PREFIX.format(run_id=uuid.uuid4().hex[:12]))
            if path is None else dataset.annotation_path(path, write=True))
    if path.exists():
        raise FileExistsError(f'Annotation already exists: {path}')
    report = Progress('ZoD detection', 1, callback=callback, cancelled=cancelled)
    missing = mask_count_seed_signal(data.evidence, data.missing_forward, data.missing_reverse)
    detection = detect_zods(data.evidence, theta_quality, theta_recovery=theta_recovery,
                            theta_gap=theta_gap, missing_signal=missing)
    header = detection_parameters(theta_quality, theta_recovery, theta_gap, direction_method=direction_method)
    header['mode'] = 'automatic'
    session = RevisionSession.create(header, detection.final_zods, data.missing_forward,
                                      data.missing_reverse, data.n, temporal_quality=data.temporal_quality)
    session.zones = [replace(zone, status='automatic') for zone in session.zones]
    report.update(1)
    report.finish()
    session.links.update(calculate_links(session.jobs, mask, data.n, progress=progress, cancelled=cancelled))
    session.path = path
    return session


def calculate_links(jobs, mask, n, *, progress=None, cancelled=None):
    """Compute supplied boundary links with progress and cancellation."""
    callback = None if progress is None else lambda done, total: progress({'completed': done, 'total': total})
    report = Progress('Automatic linking', len(jobs), callback=callback, cancelled=cancelled)
    records = {}
    for boundary in jobs:
        report.check_cancelled()
        records[boundary.id] = automatic_link(boundary, mask, n)
        report.update(1)
    report.finish()
    return records
