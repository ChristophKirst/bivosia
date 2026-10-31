"""Run prompted key-frame prediction locally or through a persistent remote service.

Examples
--------
With an RGB image and an installed SAM backend with its checkpoint:

>>> import cv2
>>> from bivosia.processing.segmentation.keyframes import predict_key_frames
>>> from bivosia.segmentation.image import create_segmenter
>>> predictor = create_segmenter('sam3', checkpoint='/path/to/sam3.pt')
>>> rgb_image = cv2.cvtColor(cv2.imread('/path/to/frame.png'), cv2.COLOR_BGR2RGB)  # doctest: +SKIP
>>> try:  # doctest: +SKIP
...     predictions = predict_key_frames(predictor, rgb_image, [[20, 30, 1]], image_key=120)
... finally:
...     predictor.close()
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import redirect_stdout
import json
import sys

from bivosia.config import REMOTE_PROTOCOL_VERSION
from bivosia.analysis.keyframes import best_mask, prompted_animals, point_prompts
from bivosia.dataset.io.transport import pack_arrays, unpack_arrays
from bivosia.processing.log import Progress, configure_logging
from bivosia.segmentation.image import create_segmenter, model_availability


def predict_key_frames(predictor, image, prompts, *, image_key, animal=None, cancelled=None):
    """Return the best mask and score for all prompted animals, or just one."""
    animals = prompted_animals(prompts, animal)
    progress = Progress(predictor.label, len(animals), cancelled=cancelled)
    predictions = {}
    for animal in animals:
        points, labels = point_prompts(prompts, animal)
        predictions[animal] = best_mask(*predictor.predict(image, points, labels, image_key=image_key))
        progress.update(1)
    progress.finish()
    return predictions


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
