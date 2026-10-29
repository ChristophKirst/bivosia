"""Install a selected Cutie or SAM package and its weights on this machine.

Examples
--------
>>> from bivosia.processing.model_install import installation
>>> installation('cutie')['source'].name
'Cutie'
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import argparse
import json
import logging
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bivosia.config import MODEL_INSTALLATIONS
from bivosia.processing.log import configure_logging
from bivosia.processing.model_install import install_model


def execute(model, *, json_output=False):
    """Report installation progress and errors in terminal or structured form."""
    configure_logging(logging.WARNING if json_output else logging.INFO)

    def emit(message):
        print(json.dumps(message), flush=True)

    try:
        result = install_model(model, (lambda text: emit({'message': text})) if json_output else None)
        if json_output:
            emit({'result': result})
        return 0
    except Exception as error:
        if json_output:
            emit({'error': str(error)})
        else:
            logging.getLogger(__name__).error('%s', error)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('model', choices=MODEL_INSTALLATIONS)
    parser.add_argument('--json', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    return execute(args.model, json_output=args.json)


if __name__ == '__main__':
    raise SystemExit(main())
