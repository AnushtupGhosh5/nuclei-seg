"""Mask-supported PDE marker watershed; no new watershed mechanism."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


def mask_pde_watershed(
    nucleus_probability: np.ndarray,
    field: np.ndarray,
    *,
    nucleus_threshold: float = 0.5,
    marker_threshold: float = 0.22,
    min_distance: int = 9,
    min_size: int = 10,
    smoothing_sigma: float = 1.5,
) -> np.ndarray:
    """Use mask probability for extent and PDE extrema for instance markers."""
    foreground = np.asarray(
        nucleus_probability >= nucleus_threshold, dtype=bool
    )
    foreground = remove_small_objects(foreground, min_size=min_size)
    if not foreground.any():
        return np.zeros(foreground.shape, dtype=np.int32)
    potential = np.clip(np.asarray(field, np.float32), 0, 1)
    if smoothing_sigma > 0:
        potential = cv2.GaussianBlur(
            potential, (0, 0), sigmaX=smoothing_sigma,
            sigmaY=smoothing_sigma,
        )
    coordinates = peak_local_max(
        potential,
        labels=foreground.astype(np.uint8),
        min_distance=min_distance,
        threshold_abs=marker_threshold,
        exclude_border=False,
    )
    marker_mask = np.zeros_like(foreground)
    if len(coordinates):
        marker_mask[coordinates[:, 0], coordinates[:, 1]] = True
    markers = ndi.label(marker_mask)[0]
    if markers.max() == 0:
        markers = ndi.label(foreground)[0]
    instances = watershed(-potential, markers=markers, mask=foreground)
    return remove_small_objects(
        instances, min_size=min_size
    ).astype(np.int32)

