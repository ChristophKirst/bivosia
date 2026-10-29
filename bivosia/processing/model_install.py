"""Install selected segmentation packages and their model weights.

Examples
--------
>>> from bivosia.processing.model_install import installation
>>> installation('sam2')['source'].name
'sam2'
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
import importlib
import json
import logging
import os
import subprocess
import sys

from bivosia.config import KEY_FRAME_BACKENDS, MODEL_INSTALLATIONS, MODEL_INSTALL_LOCK, VOS_BACKENDS
from bivosia.segmentation.model_downloads import download_weight


def installation(model):
    """Resolve installation settings and check the running Python version."""
    if model not in MODEL_INSTALLATIONS:
        raise ValueError(f'Unknown model: {model}')
    spec = MODEL_INSTALLATIONS[model]
    if sys.version_info[:2] < spec['python']:
        required = '.'.join(map(str, spec['python']))
        raise RuntimeError(f'{model.upper()} requires Python {required} or newer. '
                           f'Run BiVOSIA or this command in a compatible environment; current Python: {sys.executable}')
    return spec


def _command(arguments, report, *, env=None, cwd=None):
    """Stream a child command's output while preserving its failure status."""
    lines = []
    with subprocess.Popen(arguments, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
        try:
            for line in process.stdout:
                line = line.rstrip()
                lines.append(line)
                lines = lines[-30:]
                if line:
                    report(line)
            if process.wait():
                raise RuntimeError('\n'.join(lines) or f'Command failed: {arguments[0]}')
        except BaseException:
            if process.poll() is None:
                process.terminate()
            raise


def _source(spec, report):
    """Reuse the pinned clean checkout, or clone it into a new source directory."""
    destination = spec['source']
    if destination.exists():
        for args, expected in ((['rev-parse', 'HEAD'], spec['revision']),
                               (['status', '--porcelain', '--untracked-files=no'], '')):
            result = subprocess.run(['git', '-C', str(destination), *args], capture_output=True, text=True)
            if result.returncode or result.stdout.strip() != expected:
                raise RuntimeError(f'Existing source differs from the supported revision: {destination}. '
                                   'Preserve or move it before installing; it will not be replaced.')
        report(f'Using {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='install-', dir=destination.parent) as folder:
        checkout = Path(folder) / destination.name
        _command(['git', 'clone', '--no-checkout', spec['repository'], str(checkout)], report)
        _command(['git', '-C', str(checkout), 'checkout', '--detach', spec['revision']], report)
        checkout.rename(destination)


def install_model(model, progress=None):
    """Install into this interpreter and the configured external model folders."""
    spec = installation(model)
    MODEL_INSTALL_LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock = MODEL_INSTALL_LOCK.open('x', encoding='utf-8')
    except FileExistsError:
        raise RuntimeError(f'Model installation is already locked: {MODEL_INSTALL_LOCK}. '
                           'If an earlier installer was interrupted, verify that it has stopped before removing this file.') from None
    try:
        with lock:
            lock.write(f'{os.getpid()}\n{sys.executable}\n')
        return _install(model, spec, progress)
    finally:
        MODEL_INSTALL_LOCK.unlink()


def _install(model, spec, progress):
    backend = (KEY_FRAME_BACKENDS | VOS_BACKENDS)[model]
    logger = logging.getLogger(__name__)

    def report(message):
        logger.info('%s', message)
        if progress:
            progress(message)

    report(f'Installing {backend["label"]} with {sys.executable}')
    env = {**os.environ, 'PIP_DISABLE_PIP_VERSION_CHECK': '1', 'PIP_NO_INPUT': '1',
           'GIT_TERMINAL_PROMPT': '0', 'SAM2_BUILD_CUDA': '0'}
    with TemporaryDirectory(prefix='bivosia-install-') as folder:
        constraints = Path(folder) / 'constraints.txt'
        protected = []
        for name in ('numpy', 'scipy', 'h5py', 'PySide6', 'torch', 'torchvision', 'setuptools'):
            try:
                protected.append(f'{name}=={version(name)}')
            except PackageNotFoundError:
                pass
        constraints.write_text('\n'.join(protected) + '\n', encoding='utf-8')
        pip = [sys.executable, '-m', 'pip', 'install']
        requirements = ['-c', str(constraints), '-r', str(spec['requirements']), *spec['build']]
        report('Checking dependencies; installed core and PyTorch versions are retained')
        _command([*pip, '--dry-run', *requirements], report, env=env)
        _source(spec, report)
        _command([*pip, *requirements], report, env=env)
        _command([*pip, '--no-deps', '--no-build-isolation', '-e', str(spec['source'])], report, env=env)
    importlib.invalidate_caches()
    checkpoint = backend['checkpoint']
    for filename, url, algorithm, checksum in spec['weights']:
        download_weight(checkpoint.parent / filename, url, algorithm, checksum, report)
    check = ('from bivosia.config import KEY_FRAME_BACKENDS,VOS_BACKENDS; '
             'from bivosia.segmentation.models import create_segmenter; '
             'import sys; m=create_segmenter(KEY_FRAME_BACKENDS|VOS_BACKENDS,sys.argv[1]); '
             'print(m.check()); m.close()')
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[2]), env.get('PYTHONPATH')]))
    _command([sys.executable, '-c', check, model], report, env=env)
    report(f'Installed {backend["label"]}')
    return {'model': model, 'checkpoint': str(checkpoint), 'python': sys.executable}


def read_install_process(command, progress=None, *, request=None):
    """Read progress and the completion response from a local or SSH installer."""
    result, error, tail = None, None, []
    with subprocess.Popen(command, stdin=subprocess.PIPE if request else subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace') as process:
        try:
            if request:
                process.stdin.write(json.dumps(request) + '\n')
                process.stdin.close()
            for line in process.stdout:
                text = line.rstrip()
                try:
                    response = json.loads(text)
                except ValueError:
                    response = {'message': text}
                if not isinstance(response, dict):
                    response = {'message': text}
                if 'error' in response:
                    error = response['error']
                if 'result' in response:
                    result = response['result']
                message = response.get('message')
                if message:
                    tail.append(message)
                    tail = tail[-30:]
                    if progress:
                        progress(message)
            if process.wait() or error or result is None:
                raise RuntimeError(error or '\n'.join(tail) or 'Model installation did not complete')
        except BaseException:
            if process.poll() is None:
                process.terminate()
            raise
    return result


def install_local(model, progress=None):
    """Run installation in a fresh process using the application's interpreter."""
    script = Path(__file__).resolve().parents[1] / 'run/run_install.py'
    return read_install_process([sys.executable, str(script), model, '--json'], progress)
