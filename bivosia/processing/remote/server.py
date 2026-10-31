"""Dispatch remote requests to dataset, processing and segmentation services."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from importlib import import_module
from pathlib import Path
import json
import sys

from bivosia.config import KEY_FRAME_BACKENDS, REMOTE_PROTOCOL_VERSION, VOS_BACKENDS
from bivosia.dataset.layout import Dataset
from bivosia.dataset.storage.locations import remote_path
from bivosia.dataset.workflow.jobs import Job, list_jobs
from bivosia.processing.jobs import inspect, submit


def installation_info():
    """Report the interpreter, code and model paths after checking core imports."""
    if sys.version_info < (3, 10):
        raise RuntimeError('BiVOSIA requires Python 3.10 or newer')
    for name in ('numpy', 'scipy', 'h5py', 'cv2', 'bivosia.processing.segmentation.model_install',
                 'bivosia.processing.segmentation.vos'):
        import_module(name)
    return {'protocol': REMOTE_PROTOCOL_VERSION, 'python': sys.executable,
            'code': str(Path(__file__).resolve().parents[2]),
            'checkpoints': {name: str(spec['checkpoint'])
                            for name, spec in (KEY_FRAME_BACKENDS | VOS_BACKENDS).items()}}


def dispatch(request):
    action = request['action']
    if action == 'installation':
        return installation_info()
    if action == 'check_model':
        from bivosia.segmentation.video import model_availability
        settings = request['settings']
        selected, reason = model_availability(settings.get('model') or next(iter(VOS_BACKENDS)),
                                             settings.get('checkpoint'), settings.get('device', 'auto'),
                                             settings.get('parameters'))
        if reason:
            raise RuntimeError(reason)
        return {'device': selected}
    dataset = Dataset(remote_path(request['dataset']))
    if action == 'preflight':
        from bivosia.processing.jobs import check_inputs
        if request.get('annotation'):
            dataset.settings['annotation'] = request['annotation']
        return check_inputs(dataset, request['operation'], request['settings'])
    if action == 'read_json':
        from bivosia.dataset.storage.file_operations import checked_file
        from bivosia.dataset.io.json import read_json
        path, state = checked_file(dataset, request['path'], request['expected'])
        value = read_json(path)
        checked_file(dataset, path, state)
        return {'value': value}
    if action == 'location':
        return {'path': str(dataset.directory)}
    if action == 'file_signatures':
        from bivosia.dataset.storage.consolidation import file_signatures
        from bivosia.processing.jobs import require_idle
        require_idle(dataset)
        return file_signatures(dataset, request['files'])
    if action == 'analysis_inputs':
        if not dataset.description.is_file():
            raise FileNotFoundError('Dataset description is missing')
        if request.get('annotation'):
            dataset.settings['annotation'] = request['annotation']
        from bivosia.dataset.workflow.processing import processing_inputs
        from bivosia.dataset.io.transport import pack_arrays
        result = processing_inputs(dataset, request['operation'], request['settings'])
        if result.get('coordinates') is not None:
            result['coordinates'] = pack_arrays(coordinates=result['coordinates'],
                                                frames=result.pop('frames'), identities=result.pop('identities'))
        return result
    if action == 'keypoint_names':
        from bivosia.dataset.keypoints import open_keypoints
        return {'names': list(open_keypoints(dataset).node_names)}
    if action == 'outputs':
        from bivosia.dataset.workflow.status import inspect_outputs
        result = inspect_outputs(dataset)
        for group in result.values():
            for entry in group.values():
                entry['path'] = str(entry['path']) if entry['path'] else None
                entry['metadata'] = None
        return result
    if action == 'workflow':
        from bivosia.dataset.workflow.status import workflow_snapshot
        return workflow_snapshot(dataset, annotation_record=request.get('annotation_record'),
                                  feature_settings=request.get('feature_settings'))
    if action == 'console':
        names = list_jobs(dataset, operation=request.get('operation'))
        selected = request.get('job_id')
        job_id = selected if selected in names else next(iter(names), None)
        return {'jobs': [(name, Job(dataset, name).settings['operation']) for name in names],
                'result': inspect(Job(dataset, job_id), request.get('offset', 0) if selected == job_id else 0)
                if job_id else None}
    if action == 'inventory':
        from bivosia.dataset.storage.inventory import inventory_records
        return {'files': inventory_records(dataset)}
    if action in ('check_contents', 'delete_contents', 'create_folders', 'completed_job_folders'):
        from bivosia.processing.storage.folder_operations import contents_on_host
        return contents_on_host(dataset, action, **{key: request[key] for key in
                                ('files', 'folders', 'root', 'relatives') if key in request})
    if action == 'delete_file':
        from bivosia.dataset.storage.file_operations import delete_file
        from bivosia.processing.jobs import require_idle
        require_idle(dataset)
        delete_file(dataset, request['path'], request['expected'])
        return {'deleted': request['path']}
    if action in ('move_file', 'copy_file'):
        from bivosia.processing.storage.file_operations import transfer_on_host
        transfer_on_host(dataset, request['path'], request['expected'], request['root'], request['relative'],
                         move=action == 'move_file', replace_expected=request.get('replace_expected'))
        return {'moved' if action == 'move_file' else 'copied': request['relative']}
    if action == 'check':
        from bivosia.processing.segmentation.vos import check
        return check(dataset, **request['settings'])
    if action == 'submit':
        # A repeated submission ID reconnects; it never launches a duplicate worker.
        job = Job(dataset, request['job_id'])
        if job.directory.exists():
            return inspect(job)
        from bivosia.processing.jobs import require_idle
        require_idle(dataset)
        if request.get('annotation') is not None:
            dataset.settings['annotation'] = dataset.relative(dataset.annotation_path(request['annotation']))
        if request['operation'] != 'vos':
            from bivosia.dataset.workflow.processing import processing_inputs
            processing_inputs(dataset, request['operation'], request['settings'])
        if request.get('annotation') is not None:
            dataset.select_annotation(request['annotation'])
        return inspect(submit(dataset, request['operation'], request['settings'],
                              job_id=job.id, persistent=True))
    if action == 'list':
        return {'jobs': list_jobs(dataset, operation=request.get('operation'))}
    job = Job(dataset, request['job_id'])
    if action == 'inspect':
        return inspect(job, request.get('offset', 0))
    if action == 'cancel':
        job.cancel()
        return inspect(job)
    raise ValueError(f'Unknown remote action: {action}')


def serve(request):
    """Handle one JSON request or delegate to a streaming service."""
    try:
        if request['action'] in ('read_file', 'write_file'):
            from bivosia.processing.storage.file_operations import serve_file
            serve_file(request)
            return
        if request['action'] == 'data_session':
            from bivosia.dataset.storage.remote_reader import serve_data
            serve_data(request)
            return
        if request['action'] == 'image_session':
            from bivosia.processing.segmentation.keyframes import serve_images
            serve_images(request)
            return
        if request['action'] == 'install_model':
            from bivosia.run.run_install import execute
            execute(request['model'], json_output=True)
            return
        print(json.dumps(dispatch(request)))
    except Exception as error:
        print(json.dumps({'error': str(error), 'error_type': type(error).__name__}), flush=True)
