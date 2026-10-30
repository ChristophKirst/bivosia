"""Launch, monitor and cancel independent processing jobs.

Examples
--------
With an existing dataset, saved key frames and a local Cutie checkpoint:

>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.processing.jobs import submit
>>> dataset = Dataset("/path/to/dataset")
>>> settings = {'model': 'cutie', 'checkpoint': '/path/to/cutie-base-mega.pth'}
>>> job = submit(dataset, 'vos', settings)  # doctest: +SKIP
>>> job.status  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import logging
import os
import shutil
import subprocess
import sys
import threading
import time

from bivosia.config import JOB_TERMINAL_STATES, PROCESSING_DEFAULTS
from bivosia.dataset.jobs import Job, list_jobs
from bivosia.processing.log import configure_logging


def operation(name):
    """Resolve supported processing operations without importing model packages."""
    if name == 'vos':
        from bivosia.processing.segmentation import segment
        return segment
    if name == 'quantify':
        from bivosia.processing.quantify import quantify_job
        return quantify_job
    if name == 'shape_features':
        from bivosia.processing.shape_features import shape_features_job
        return shape_features_job
    if name == 'align':
        from bivosia.processing.alignment import alignment_job
        return alignment_job
    if name == 'detect_link':
        from bivosia.processing.zod_detection import detect_link_job
        return detect_link_job
    if name == 'export':
        from bivosia.processing.export import export_job
        return export_job
    if name == 'workflow':
        from bivosia.processing.workflow import workflow_job
        return workflow_job
    raise ValueError(f'Unknown processing operation: {name}')


def require_idle(dataset):
    """Require existing dataset jobs to end before starting another calculation."""
    if any(Job(dataset, name).status['state'] not in JOB_TERMINAL_STATES
           for name in list_jobs(dataset)):
        raise ValueError('Finish or cancel the active dataset job before starting another calculation.')


def _launch_worker(job, command, persistent):
    """Keep SSH workers outside login-session cleanup on systemd hosts."""
    launcher = shutil.which('systemd-run') if persistent and sys.platform == 'linux' else None
    if launcher and Path('/run/systemd/system').is_dir():
        unit = f'bivosia-{job.id}.service'
        options = [launcher, '--user', '--no-ask-password', '--collect', '--quiet',
                   f'--unit={unit}', '--property=Type=exec', '--same-dir',
                   f'--property=StandardOutput=append:{job.path("log")}',
                   '--property=StandardError=inherit']
        # Copy the submitting environment without exposing its values in process arguments.
        managed = {'INVOCATION_ID', 'JOURNAL_STREAM', 'NOTIFY_SOCKET', 'SYSTEMD_EXEC_PID',
                   'LISTEN_PID', 'LISTEN_FDS', 'LISTEN_FDNAMES'}
        options.extend(f'--setenv={name}' for name in os.environ if name not in managed)
        result = subprocess.run([*options, '--', *command], capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError('Could not start the remote worker as a systemd user service: ' +
                               (result.stderr.strip() or result.stdout.strip()) +
                               '. Check the user service manager; see Help > Remote setup.')
        return f'user service {unit}'
    options = ({'creationflags': subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
               if os.name == 'nt' else {'start_new_session': True})
    with job.path('log').open('ab', buffering=0) as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   close_fds=True, **options)
    return f'PID {process.pid}'


def submit(dataset, name, settings, *, background=True, job_id=None, persistent=False):
    """Start a saved job; persistent SSH launches use user services on systemd hosts."""
    operation(name)
    from bivosia.dataset.layout import Dataset

    if dataset.settings != Dataset(dataset.directory).settings:
        raise ValueError('Save the dataset description before starting a background job')
    job = Job.create(dataset, name, settings, job_id)
    if background:
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'run/run_job.py'),
                   str(dataset.directory), job.id]
        try:
            worker = _launch_worker(job, command, persistent)
            # The worker owns status updates; the launcher never overwrites its state.
            logging.getLogger(__name__).info('Started job %s (%s)', job.id, worker)
        except Exception as error:
            job.update(state='failed', error=str(error))
            raise
    return job


def run(job, *, console=True):
    """Execute once, logging progress, errors and completion status."""
    try:
        with job.path('claim').open('x', encoding='utf-8') as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError('This job has already been started') from None
    if job.status['state'] in JOB_TERMINAL_STATES:
        raise RuntimeError('This job has already ended')
    configure_logging()
    handler = logging.FileHandler(job.path('log'), encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    logger = logging.getLogger()
    previous = list(logger.handlers) if not console else []
    for stream_handler in previous:
        logger.removeHandler(stream_handler)
    logger.addHandler(handler)
    started = time.time()
    lock, done = threading.Lock(), threading.Event()

    def update(**values):
        with lock:
            return job.update(**values)

    def heartbeat():
        while not done.wait(PROCESSING_DEFAULTS['jobs']['heartbeat_seconds']):
            update(elapsed=time.time() - started)

    update(state='running', pid=os.getpid(), started=started)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    result, state, error = [], 'complete', None
    try:
        if job.cancelled:
            raise InterruptedError('Cancelled before processing')
        description = job.settings
        result = operation(description['operation'])(job.dataset, **description['settings'],
                    progress=lambda values: update(**values), cancelled=lambda: job.cancelled)
    except (InterruptedError, KeyboardInterrupt) as exception:
        state, error = 'cancelled', str(exception)
        logging.getLogger(__name__).info('Job cancelled: %s', error)
    except BaseException as exception:
        state, error = 'failed', str(exception)
        logging.getLogger(__name__).exception('Job failed')
    finally:
        done.set()
        thread.join()
        update(state=state, error=error, results=result, elapsed=time.time() - started)
        logger.removeHandler(handler)
        handler.close()
        for stream_handler in previous:
            logger.addHandler(stream_handler)
    return 0 if state == 'complete' else 1


def inspect(job, offset=0):
    """Read status and incremental logs; stale workers are not reported as healthy."""
    status = job.status
    if status['state'] not in JOB_TERMINAL_STATES and time.time() - status['updated'] > PROCESSING_DEFAULTS['jobs']['stale_seconds']:
        status = {**status, 'state': 'unresponsive'}
    log, offset = job.read_log(offset)
    return {'id': job.id, 'status': status, 'log': log, 'offset': offset}
