from __future__ import annotations
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


def make_poisson_field(instance_map: np.ndarray, iterations: int = 64) -> np.ndarray:
    """Approximate Delta u=-1 per instance with u=0 on the instance boundary."""
    labels = np.asarray(instance_map, dtype=np.int32)
    field = np.zeros(labels.shape, dtype=np.float32)
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
    for instance_id in np.unique(labels):
        if instance_id == 0:
            continue
        mask = labels == instance_id
        ys, xs = np.where(mask)
        if not len(ys):
            continue
        y0, y1 = max(0, ys.min()-1), min(labels.shape[0], ys.max()+2)
        x0, x1 = max(0, xs.min()-1), min(labels.shape[1], xs.max()+2)
        local_mask = mask[y0:y1, x0:x1]
        u = np.zeros(local_mask.shape, dtype=np.float32)
        for _ in range(iterations):
            neighbours = cv2.filter2D(
                u, -1, kernel, borderType=cv2.BORDER_CONSTANT
            )
            u = ((neighbours + 1.0) * 0.25) * local_mask
        maximum = float(u.max())
        if maximum > 0:
            u /= maximum
        target = field[y0:y1, x0:x1]
        target[local_mask] = u[local_mask]
    return field


def poisson_watershed(
    field: np.ndarray,
    foreground_threshold: float = 0.12,
    marker_threshold: float = 0.22,
    min_distance: int = 9,
    min_size: int = 10,
    smoothing_sigma: float = 1.5,
) -> np.ndarray:
    """Convert a predicted scalar potential into instances.

    The learned field is Gaussian-smoothed before thresholding. Without this
    step, isolated high-confidence pixels are all discarded by the minimum-size
    filter before watershed. The defaults were selected on the five-image
    validation split only (seed 42), never on the held-out test set.
    """
    potential = np.clip(np.asarray(field, dtype=np.float32), 0.0, 1.0)
    if smoothing_sigma > 0:
        potential = cv2.GaussianBlur(
            potential, (0, 0), sigmaX=smoothing_sigma,
            sigmaY=smoothing_sigma,
        )
    foreground = potential >= foreground_threshold
    foreground = remove_small_objects(foreground, min_size=min_size)
    if not foreground.any():
        return np.zeros(potential.shape, dtype=np.int32)
    coords = peak_local_max(
        potential,
        labels=foreground.astype(np.uint8),
        min_distance=min_distance,
        threshold_abs=marker_threshold,
        exclude_border=False,
    )
    marker_mask = np.zeros_like(foreground, dtype=bool)
    if len(coords):
        marker_mask[coords[:, 0], coords[:, 1]] = True
    markers = ndi.label(marker_mask)[0]
    if markers.max() == 0:
        markers = ndi.label(foreground)[0]
    instances = watershed(-potential, markers=markers, mask=foreground)
    return remove_small_objects(
        instances, min_size=min_size
    ).astype(np.int32)
