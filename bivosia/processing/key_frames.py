"""Run prompted key-frame mask prediction with shared progress reporting.

Examples
--------
With an RGB image and an installed SAM backend with its checkpoint:

>>> import cv2
>>> from bivosia.processing.key_frames import predict_key_frames
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

from bivosia.analysis.key_frames import best_mask, prompted_animals, point_prompts
from bivosia.processing.log import Progress


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
