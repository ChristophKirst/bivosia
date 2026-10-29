"""Submit background jobs and read their results through OpenSSH.

Examples
--------
>>> from bivosia.processing.remote import Remote
>>> server = Remote('workstation', 'python', '/software/bivosia', '/data/recording')
>>> server.call('list')  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys

from bivosia.config import KEY_FRAME_BACKENDS, MODEL_INSTALLATIONS, REMOTE_PROTOCOL_VERSION, VOS_BACKENDS
from bivosia.dataset.jobs import Job, list_jobs
from bivosia.dataset.layout import Dataset
from bivosia.dataset.locations import remote_path
from bivosia.dataset.paths import auxiliary_path
from bivosia.processing.jobs import inspect, submit


# Only fixed code enters the remote shell. Paths/settings travel as JSON on stdin.
BOOTSTRAP = ("import json,sys; from pathlib import Path; r=json.loads(sys.stdin.buffer.readline()); "
             "p=Path(r['code']).expanduser().resolve(); "
             "sys.exit('BiVOSIA was not found in the selected remote Code folder. "
             "Install the bivosia project on the server and set Code to its inner bivosia package folder "
             "(containing config.py and processing/). Set Python to the server environment "
             "with the required dependencies installed. See Help > Remote setup in the local application.') "
             "if not (p/'config.py').is_file() or not (p/'processing/remote.py').is_file() else None; "
             "sys.path.insert(0,str(p.parent)); "
             "from bivosia.processing.remote import serve; serve(r)")


@dataclass(frozen=True)
class Remote:
    host: str
    python: str
    code: str
    dataset: str = ''

    def command(self):
        if not self.code.strip():
            raise ValueError('Specify the remote bivosia Code folder containing config.py and processing/')
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@:-]*', self.host):
            raise ValueError('Use an SSH host alias or user@host; configure ports/keys in SSH config')
        if not re.fullmatch(r'[A-Za-z0-9_/\\ .:-]+', self.python) or not self.python.strip():
            raise ValueError('Python executable path contains unsupported shell characters')
        return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3',
                '-o', 'StrictHostKeyChecking=yes', self.host,
                f'"{self.python}" -c "{BOOTSTRAP}"']

    def request(self, action, **values):
        return {'action': action, 'code': self.code, 'dataset': self.dataset, **values}

    def sync(self, dataset, progress=None, cancelled=None, *, dry_run=False):
        """Synchronize this dataset's required VOS inputs without replacing remote files."""
        from bivosia.processing.remote_sync import synchronize
        return synchronize(self, dataset, progress, cancelled, dry_run=dry_run)

    def call(self, action, *, timeout=60, **values):
        payload = json.dumps(self.request(action, **values)) + '\n'
        result = subprocess.run(self.command(), input=payload, text=True, encoding='utf-8',
                                capture_output=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f'SSH exited with code {result.returncode}')
        try:
            response = json.loads(result.stdout)
        except ValueError:
            raise RuntimeError('Remote Python did not return a job response: ' + result.stdout[-1000:]) from None
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response

    def install_model(self, model, progress=None):
        """Install model dependencies on this host without requiring a dataset."""
        from bivosia.processing.model_install import read_install_process

        if model not in MODEL_INSTALLATIONS:
            raise ValueError(f'Automatic installation is not supported for {model}')
        return read_install_process(self.command(), progress,
                                    request=self.request('install_model', model=model))

    def check_installation(self):
        """Check an installation without requiring a dataset or model weights."""
        try:
            response = self.call('installation')
        except RuntimeError as error:
            if 'Unknown remote action: installation' in str(error):
                raise RuntimeError('Update the remote bivosia folder to the current code before registering it.') from None
            raise
        if not isinstance(response, dict):
            raise ValueError('Remote BiVOSIA returned invalid installation details')
        if response.get('protocol') != REMOTE_PROTOCOL_VERSION:
            raise ValueError('Remote BiVOSIA uses an incompatible protocol. Update its code before saving it.')
        if not all(isinstance(response.get(key), str) and response[key]
                   for key in ('python', 'code')) or not isinstance(response.get('checkpoints'), dict):
            raise ValueError('Remote BiVOSIA returned incomplete installation details. Update its code.')
        if not all(isinstance(value, str) and value for value in response['checkpoints'].values()):
            raise ValueError('Remote BiVOSIA returned invalid checkpoint paths.')
        if not set(VOS_BACKENDS).issubset(response['checkpoints']):
            raise ValueError('Remote BiVOSIA does not support the configured VOS models.')
        return response

    def fetch(self, job_id, dataset, cancelled=None, progress=None):
        """Explicitly fetch a completed job's outputs without overwriting local data."""
        response = self.call('inspect', job_id=job_id)
        if response['status']['state'] != 'complete':
            raise ValueError('Only completed results can be fetched')
        results = response['status']['results']
        for relative in results:
            target = dataset.path(relative, write=True)
            partial = auxiliary_path(target, 'download')
            if target.exists() or partial.exists():
                raise FileExistsError(f'Existing local result or download: {target}')
        for index, relative in enumerate(results):
            target = dataset.path(relative, write=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = auxiliary_path(target, 'download')
            request = self.request('fetch', job_id=job_id, relative=relative)
            with subprocess.Popen(self.command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL) as process:
                process.stdin.write((json.dumps(request) + '\n').encode('utf-8'))
                process.stdin.close()
                header = json.loads(process.stdout.readline())
                if 'error' in header:
                    raise RuntimeError(header['error'])
                digest, remaining = hashlib.sha256(), header['size']
                try:
                    with partial.open('xb') as stream:
                        while remaining:
                            if cancelled and cancelled():
                                raise InterruptedError('Download cancelled')
                            chunk = process.stdout.read(min(1024**2, remaining))
                            if not chunk:
                                raise IOError('Remote download ended early')
                            stream.write(chunk)
                            digest.update(chunk)
                            remaining -= len(chunk)
                            if progress:
                                progress(index, len(results), header['size'] - remaining, header['size'])
                    trailer = json.loads(process.stdout.readline())
                    if process.wait() or trailer.get('sha256') != digest.hexdigest():
                        raise IOError('Downloaded result checksum differs')
                    with target.open('xb'):
                        pass
                    os.replace(partial, target)
                except BaseException:
                    process.kill()
                    raise
        return results


def installation_info():
    """Report the interpreter, code and model paths after checking core imports."""
    if sys.version_info < (3, 10):
        raise RuntimeError('BiVOSIA requires Python 3.10 or newer')
    for name in ('numpy', 'scipy', 'h5py', 'cv2', 'bivosia.processing.model_install',
                 'bivosia.processing.segmentation'):
        import_module(name)
    return {'protocol': REMOTE_PROTOCOL_VERSION, 'python': sys.executable,
            'code': str(Path(__file__).resolve().parents[1]),
            'checkpoints': {name: str(spec['checkpoint'])
                            for name, spec in (KEY_FRAME_BACKENDS | VOS_BACKENDS).items()}}


def dispatch(request):
    action = request['action']
    if action == 'installation':
        return installation_info()
    if action == 'check_model':
        from bivosia.segmentation.vos import model_availability
        settings = request['settings']
        selected, reason = model_availability(settings.get('model') or next(iter(VOS_BACKENDS)),
                                             settings.get('checkpoint'), settings.get('device', 'auto'),
                                             settings.get('parameters'))
        if reason:
            raise RuntimeError(reason)
        return {'device': selected}
    dataset = Dataset(remote_path(request['dataset']))
    if action == 'inventory':
        from bivosia.dataset.inventory import inventory_records
        return {'files': inventory_records(dataset)}
    if action == 'delete_file':
        from bivosia.dataset.file_operations import delete_file
        from bivosia.processing.jobs import require_idle
        require_idle(dataset)
        delete_file(dataset, request['path'], request['expected'])
        return {'deleted': request['path']}
    if action in ('move_file', 'copy_file'):
        from bivosia.processing.file_operations import transfer_on_host
        transfer_on_host(dataset, request['path'], request['expected'], request['root'], request['relative'],
                         move=action == 'move_file')
        return {'moved' if action == 'move_file' else 'copied': request['relative']}
    if action == 'check':
        from bivosia.processing.segmentation import check
        return check(dataset, **request['settings'])
    if action == 'submit':
        # A repeated submission ID reconnects; it never launches a duplicate worker.
        job = Job(dataset, request['job_id'])
        if job.directory.exists():
            return inspect(job)
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
    """Handle one JSON request, optionally streaming a completed result."""
    try:
        if request['action'] in ('read_file', 'write_file'):
            from bivosia.processing.file_operations import serve_file
            serve_file(request)
            return
        if request['action'] == 'image_session':
            from bivosia.processing.remote_images import serve_images
            serve_images(request)
            return
        if request['action'] == 'install_model':
            from bivosia.run.run_install import execute
            execute(request['model'], json_output=True)
            return
        if request['action'] == 'sync_inputs':
            from bivosia.processing.remote_sync import serve_upload
            serve_upload(request)
            return
        if request['action'] != 'fetch':
            print(json.dumps(dispatch(request)))
            return
        dataset = Dataset(remote_path(request['dataset']))
        job = Job(dataset, request['job_id'])
        status = job.status
        relative = request['relative']
        if status['state'] != 'complete' or relative not in status['results']:
            raise ValueError('File is not a completed result of this job')
        path = dataset.path(relative)
        with path.open('rb') as stream:
            print(json.dumps({'size': path.stat().st_size}), flush=True)
            digest = hashlib.sha256()
            while chunk := stream.read(1024**2):
                sys.stdout.buffer.write(chunk)
                digest.update(chunk)
            sys.stdout.buffer.flush()
            print(json.dumps({'sha256': digest.hexdigest()}), flush=True)
    except Exception as error:
        print(json.dumps({'error': str(error)}), flush=True)
