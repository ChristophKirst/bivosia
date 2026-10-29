"""Transport image prompts and model responses without changing their arrays.

Examples
--------
>>> import numpy as np
>>> from bivosia.processing.remote_images import pack_arrays, unpack_arrays
>>> source = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
>>> np.array_equal(unpack_arrays(pack_arrays(rgb=source))['rgb'], source)
True
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import redirect_stdout
from io import BytesIO
import base64
import json
import sys

import numpy as np

from bivosia.config import REMOTE_PROTOCOL_VERSION
from bivosia.processing.log import configure_logging
from bivosia.segmentation.image import create_segmenter, model_availability


def pack_arrays(**arrays):
    """Encode numeric arrays losslessly in a compressed NumPy archive."""
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    return base64.b64encode(stream.getvalue()).decode('ascii')


def unpack_arrays(payload):
    """Decode only NumPy arrays that do not require pickle."""
    with np.load(BytesIO(base64.b64decode(payload, validate=True)), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def check_models(session, checkpoints, device, parameters):
    """Check model dependencies on the selected host without loading weights."""
    try:
        return session.request('availability', checkpoints={key: str(path) for key, path in checkpoints.items()},
                               device=device, parameters=parameters)['availability']
    finally:
        session.close()


def serve_images(request):
    """Keep the current model and image alive until the SSH input closes."""
    output = sys.stdout
    predictor = rgb = None
    image_key = 0
    configure_logging()
    with redirect_stdout(sys.stderr):
        try:
            if request.get('protocol') != REMOTE_PROTOCOL_VERSION:
                raise ValueError('Update BiVOSIA on both machines: incompatible image-session protocol')
            for line in sys.stdin.buffer:
                values = json.loads(line)
                operation = values['operation']
                if operation == 'availability':
                    result = {'availability': {
                        name: (model_availability(name, path, values['device'], values['parameters'].get(name))
                               if path else (None, 'Specify the remote checkpoint path'))
                        for name, path in values['checkpoints'].items()}}
                elif operation == 'configure':
                    if predictor is not None:
                        predictor.close()
                    predictor = create_segmenter(**values['settings'])
                    predictor.device = predictor.check()
                    rgb = None
                    result = {'device': predictor.device, 'metadata': predictor.metadata}
                elif operation == 'predict':
                    if predictor is None:
                        raise ValueError('Configure an image model before prediction')
                    arrays = unpack_arrays(values['arrays'])
                    if 'rgb' in arrays:
                        rgb = arrays['rgb']
                        image_key += 1
                    if rgb is None:
                        raise ValueError('Send the image before predicting its prompts')
                    masks, scores = predictor.predict(rgb, arrays['points'], arrays['labels'], image_key=image_key)
                    result = {'arrays': pack_arrays(masks=masks, scores=scores), 'metadata': predictor.metadata}
                else:
                    raise ValueError(f'Unknown image-session operation: {operation}')
                print(json.dumps({'protocol': REMOTE_PROTOCOL_VERSION, **result}), file=output, flush=True)
        except Exception as error:
            print(json.dumps({'protocol': REMOTE_PROTOCOL_VERSION, 'error': str(error)}), file=output, flush=True)
        finally:
            if predictor is not None:
                predictor.close()
