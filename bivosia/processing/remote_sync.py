"""Transfer verified VOS inputs through one SSH connection before remote execution."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import closing
import json
import subprocess
import sys
import tempfile

from bivosia.dataset.layout import Dataset
from bivosia.dataset.locations import remote_path
from bivosia.dataset.transfer import InputUpload, file_digest, vos_inputs
from bivosia.processing.jobs import require_idle
from bivosia.processing.log import Progress


def synchronize(remote, dataset, progress=None, cancelled=None, *, dry_run=False):
    """Send missing inputs, skip identical files and report checksum/transfer progress."""
    remote_path(remote.dataset)

    def tracker(stage, total):
        if progress:
            progress({'stage': stage, 'completed': 0, 'total': total})
        return Progress(stage, total, cancelled=cancelled, callback=(
            lambda done, count: progress({'stage': stage, 'completed': done, 'total': count})) if progress else None)

    installation = tracker('Checking remote installation', 0)
    remote.check_installation()
    installation.check_cancelled()
    with vos_inputs(dataset) as sources:
        sizes = {name: path.stat().st_size for name, path in sources.items()}
        hashing = tracker('Checking local inputs', sum(sizes.values()))
        manifest = {name: {'size': sizes[name], 'sha256': file_digest(path, hashing.update)}
                    for name, path in sources.items()}
        hashing.finish()
        checking = tracker('Checking remote inputs', 0)
        with tempfile.TemporaryFile() as errors, subprocess.Popen(
                remote.command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors) as process:
            def response():
                line = process.stdout.readline()
                if not line:
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode('utf-8', errors='replace').strip() or
                                       'Remote input synchronization ended unexpectedly')
                result = json.loads(line)
                if 'error' in result:
                    raise RuntimeError(result['error'])
                return result

            try:
                request = remote.request('sync_inputs', manifest=manifest, dry_run=dry_run)
                process.stdin.write((json.dumps(request) + '\n').encode('utf-8'))
                process.stdin.flush()
                missing = response()['missing']
                if len(set(missing)) != len(missing) or any(name not in manifest for name in missing):
                    raise ValueError('Remote upload plan contains unexpected files')
                checking.check_cancelled()
                if dry_run:
                    process.stdin.close()
                    if process.wait():
                        raise RuntimeError('Remote upload check failed')
                    return {'missing': missing}
                uploading = tracker('Uploading inputs', sum(sizes[name] for name in missing))
                for name in missing:
                    remaining = sizes[name]
                    with sources[name].open('rb') as source:
                        while remaining:
                            uploading.check_cancelled()
                            chunk = source.read(min(1024**2, remaining))
                            if not chunk:
                                raise IOError(f'Local input changed during transfer: {name}')
                            process.stdin.write(chunk)
                            remaining -= len(chunk)
                            uploading.update(len(chunk))
                        if source.read(1):
                            raise IOError(f'Local input changed during transfer: {name}')
                process.stdin.close()
                result = response()
                if process.wait():
                    raise RuntimeError('Remote upload process failed')
                uploading.finish()
                if progress:
                    progress({'stage': 'Inputs synchronized', 'completed': 1, 'total': 1})
                return result
            except BaseException:
                process.kill()
                raise


def serve_upload(request):
    """Receive the input set and publish it only after checksum verification."""
    dataset = Dataset(remote_path(request['dataset']))
    require_idle(dataset)
    with closing(InputUpload(dataset, request['manifest'])) as upload:
        missing = upload.plan()
        print(json.dumps({'missing': missing}), flush=True)
        if request.get('dry_run', False):
            return
        upload.receive(sys.stdin.buffer)
        upload.publish()
        print(json.dumps({'uploaded': missing,
                          'unchanged': [name for name in request['manifest'] if name not in missing]}), flush=True)
