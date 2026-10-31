"""Send requests to a BiVOSIA installation through OpenSSH.

Examples
--------
>>> from bivosia.processing.remote.client import Remote
>>> server = Remote('workstation', 'python', '/software/bivosia', '/data/recording')
>>> server.call('list')  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from dataclasses import dataclass
import json
import re
import subprocess

from bivosia.config import MODEL_INSTALLATIONS, REMOTE_PROTOCOL_VERSION, VOS_BACKENDS


# Only fixed code enters the remote shell. Paths/settings travel as JSON on stdin.
BOOTSTRAP = ("import json,sys; from pathlib import Path; r=json.loads(sys.stdin.buffer.readline()); "
             "p=Path(r['code']).expanduser().resolve(); "
             "sys.exit('BiVOSIA was not found in the selected remote Code folder. "
             "Install the bivosia project on the server and set Code to its inner bivosia package folder "
             "(containing config.py and processing/). Set Python to the server environment "
             "with the required dependencies installed. See Help > Remote setup in the local application.') "
             "if not (p/'config.py').is_file() or not (p/'processing/remote/server.py').is_file() else None; "
             "sys.path.insert(0,str(p.parent)); "
             "from bivosia.processing.remote.server import serve; serve(r)")


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
            error_type = FileNotFoundError if response.get('error_type') == 'FileNotFoundError' else RuntimeError
            raise error_type(response['error'])
        return response

    def install_model(self, model, progress=None):
        """Install model dependencies on this host without requiring a dataset."""
        from bivosia.processing.segmentation.model_install import read_install_process

        if model not in MODEL_INSTALLATIONS:
            raise ValueError(f'Automatic installation is not supported for {model}')
        return read_install_process(self.command(), progress,
                                    request=self.request('install_model', model=model))

    def check_installation(self):
        """Check an installation without requiring a dataset or model weights."""
        response = self.call('installation')
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
