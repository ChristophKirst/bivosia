"""Exchange JSON requests with a persistent Python process over SSH."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import suppress
import json
import subprocess
import tempfile
import threading

from bivosia.config import PROCESSING_DEFAULTS, REMOTE_PROTOCOL_VERSION


class RemoteSession:
    """Keep one SSH connection open, with bounded requests and explicit cleanup."""

    def __init__(self, remote, action, *, timeout=None):
        self.remote, self.action = remote, action
        self.timeout = timeout if timeout is not None else PROCESSING_DEFAULTS['remote_images']['timeout_seconds']
        self.process = None
        self.closed = False
        self.error_output = ''
        self._lock = threading.Lock()
        self._requests = threading.Lock()
        self._stderr = None

    def request(self, operation, **values):
        """Send one request; closing the session interrupts blocked pipe operations."""
        if not self._requests.acquire(blocking=False):
            raise RuntimeError('A remote request is already running')
        expired = threading.Event()
        timer = None
        try:
            with self._lock:
                if self.closed:
                    raise RuntimeError('Remote session is closed')
                first = self.process is None
                if first:
                    self._stderr = tempfile.TemporaryFile()
                    self.process = subprocess.Popen(self.remote.command(), stdin=subprocess.PIPE,
                                                    stdout=subprocess.PIPE, stderr=self._stderr)
                process = self.process

            def expire():
                expired.set()
                self.close()

            timer = threading.Timer(self.timeout, expire)
            timer.daemon = True
            timer.start()
            if first:
                initial = self.remote.request(self.action, protocol=REMOTE_PROTOCOL_VERSION)
                process.stdin.write((json.dumps(initial) + '\n').encode('utf-8'))
            process.stdin.write((json.dumps({'operation': operation, **values}) + '\n').encode('utf-8'))
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                raise RuntimeError('Remote session ended before replying')
            response = json.loads(line)
            if not isinstance(response, dict):
                raise ValueError('Invalid remote session response')
            if response.get('protocol') != REMOTE_PROTOCOL_VERSION:
                raise RuntimeError('Update the remote BiVOSIA code: incompatible image-session protocol')
            if 'error' in response:
                raise RuntimeError(response['error'])
            return response
        except Exception as error:
            self.close()
            if expired.is_set():
                raise TimeoutError(f'Remote request timed out after {self.timeout:g} seconds') from error
            detail = self.error_output.strip()
            raise RuntimeError(str(error) + ('\n' + detail if detail else '')) from error
        finally:
            if timer is not None:
                timer.cancel()
            self._requests.release()

    def close(self):
        """Release the SSH process and interrupt any pending request."""
        with self._lock:
            self.closed = True
            process, self.process = self.process, None
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait()
                for stream in (process.stdin, process.stdout):
                    with suppress(OSError):
                        stream.close()
            if self._stderr is not None:
                self._stderr.seek(0, 2)
                self._stderr.seek(max(0, self._stderr.tell() - 4000))
                self.error_output = self._stderr.read().decode('utf-8', errors='replace')
                self._stderr.close()
                self._stderr = None
