"""Focused checks for optional mask features and scale-independent roughness."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import math
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from bivosia.analysis.features import (
    FFT_FEATURE_NAMES, HU_FEATURE_NAMES, _spectral_boundary_features, feature_names, frame_features,
)
from bivosia.analysis.measures import mask_measures


def _radial_contour(radii: np.ndarray, center=(100.0, 100.0)) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, len(radii), endpoint=False)
    points = np.column_stack(
        (center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles))
    )
    return points.astype(np.float32).reshape(-1, 1, 2)


class BoundaryFeatureTests(unittest.TestCase):
    def test_disabling_hu_skips_calculations_and_preserves_other_features(self) -> None:
        labels = np.zeros((3, 24, 24), dtype=np.uint8)
        labels[0, 3:10, 3:8] = 1
        labels[1, 3:9, 4:11] = 1
        labels[:, 15:20, 15:21] = 2
        for convex_hull in (False, True):
            with self.subTest(convex_hull=convex_hull):
                options = {'include_convex_hull': convex_hull, 'include_fft_features': True}
                enabled_names = feature_names(**options, include_hu_features=True)
                disabled_names = feature_names(**options)
                self.assertFalse(set(HU_FEATURE_NAMES) & set(disabled_names))
                enabled, disabled = [], []
                previous = None
                for frame in labels:
                    values, previous, _, _ = frame_features(
                        frame, 2, previous, **options, include_hu_features=True)
                    enabled.append(values)
                previous = None
                with patch('bivosia.analysis.features.cv2.HuMoments', side_effect=AssertionError('Hu calculation')), \
                     patch('bivosia.analysis.features.cv2.matchShapes', side_effect=AssertionError('Hu shape comparison')):
                    for frame in labels:
                        values, previous, _, _ = frame_features(
                            frame, 2, previous, **options)
                        disabled.append(values)
                enabled, disabled = np.asarray(enabled), np.asarray(disabled)
                retained = [enabled_names.index(name) for name in disabled_names]
                np.testing.assert_allclose(disabled, enabled[:, :, retained], equal_nan=True)
                full_measures, full_names = mask_measures(np.stack((enabled, enabled)), enabled_names)
                kept_measures, kept_names = mask_measures(np.stack((disabled, disabled)), disabled_names)
                self.assertIn('max_shape_change_i1', full_names)
                self.assertNotIn('max_shape_change_i1', kept_names)
                retained = [full_names.index(name) for name in kept_names]
                np.testing.assert_allclose(kept_measures, full_measures[:, :, retained], equal_nan=True)

    def test_convex_hull_and_fft_are_independent(self) -> None:
        labels = np.zeros((24, 24), dtype=np.uint8)
        labels[3:16, 3:7] = labels[12:16, 3:15] = 1
        enabled, _, _, _ = frame_features(labels, 1, None, include_convex_hull=True, include_fft_features=True)
        enabled_names = feature_names(include_convex_hull=True, include_fft_features=True)
        full, full_names = mask_measures(np.stack((enabled[None], enabled[None])), enabled_names)
        self.assertEqual(feature_names(), feature_names(include_convex_hull=True, include_fft_features=False))
        for convex_hull in (False, True):
            for fft in (False, True):
                with self.subTest(convex_hull=convex_hull, fft=fft):
                    options = {'include_convex_hull': convex_hull, 'include_fft_features': fft}
                    names = feature_names(**options)
                    self.assertEqual('convex_hull_perimeter_excess' in names, convex_hull)
                    self.assertEqual(set(FFT_FEATURE_NAMES) & set(names), set(FFT_FEATURE_NAMES) if fft else set())
                    with patch('bivosia.analysis.features.cv2.convexHull', wraps=cv2.convexHull) as hull_call, \
                         patch('bivosia.analysis.features._spectral_boundary_features', wraps=_spectral_boundary_features) as fft_call:
                        values, _, _, _ = frame_features(labels, 1, None, **options)
                    self.assertEqual(hull_call.called, convex_hull)
                    self.assertEqual(fft_call.called, fft)
                    retained = [enabled_names.index(name) for name in names]
                    np.testing.assert_allclose(values, enabled[:, retained], equal_nan=True)
                    kept, kept_names = mask_measures(np.stack((values[None], values[None])), names)
                    self.assertEqual('max_convex_hull_perimeter_excess' in kept_names, convex_hull)
                    for name in FFT_FEATURE_NAMES:
                        self.assertEqual('max_' + name in kept_names, fft)
                    retained = [full_names.index(name) for name in kept_names]
                    np.testing.assert_allclose(kept, full[:, :, retained], equal_nan=True)

    def test_jagged_contour_is_rougher_than_circle(self) -> None:
        circle = _radial_contour(np.full(128, 50.0))
        jagged = _radial_contour(np.tile((50.0, 38.0), 64))
        smooth_values = _spectral_boundary_features(circle)
        jagged_values = _spectral_boundary_features(jagged)
        self.assertTrue(
            all(bad > smooth for smooth, bad in zip(smooth_values, jagged_values))
        )

    def test_spectral_features_ignore_translation_and_scale(self) -> None:
        radii = 50.0 + 4.0 * np.sin(
            np.linspace(0, 12 * math.pi, 128, endpoint=False)
        )
        contour = _radial_contour(radii)
        transformed = contour * 3.5 + np.array([[[240.0, -90.0]]], dtype=np.float32)
        np.testing.assert_allclose(
            _spectral_boundary_features(contour),
            _spectral_boundary_features(transformed),
            rtol=2e-5,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
