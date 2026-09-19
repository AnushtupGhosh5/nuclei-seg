from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


def post_process_instances(
    nucleus_probability: np.ndarray,
    hv_map: np.ndarray,
    type_probability: np.ndarray | None = None,
    nucleus_threshold: float = 0.5,
    boundary_threshold: float = 0.4,
    min_size: int = 10,
    type_boundary_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split touching nuclei with HoVer gradients and marker watershed.

    If ``type_boundary_weight`` is positive, changes in the type probability
    map add a weak boundary cue. Zero reproduces the standard HoVer principle.
    """
    nucleus = nucleus_probability >= nucleus_threshold
    nucleus = remove_small_objects(nucleus, min_size=min_size)
    if not nucleus.any():
        return np.zeros(nucleus.shape, np.int32), np.zeros(1, np.int16)

    # This follows the reference HoVer-Net marker-controlled watershed. The
    # wide Sobel operator is intentional: it turns HV discontinuities between
    # touching instances into robust separation ridges.
    horizontal = cv2.normalize(hv_map[..., 0].astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    vertical = cv2.normalize(hv_map[..., 1].astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    horizontal = cv2.Sobel(horizontal, cv2.CV_64F, 1, 0, ksize=21)
    vertical = cv2.Sobel(vertical, cv2.CV_64F, 0, 1, ksize=21)
    horizontal = 1.0 - cv2.normalize(horizontal, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    vertical = 1.0 - cv2.normalize(vertical, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    boundary = np.maximum(horizontal, vertical)
    boundary = boundary - (~nucleus).astype(np.float32)
    boundary[boundary < 0] = 0

    if type_probability is not None and type_boundary_weight > 0:
        foreground_types = type_probability[..., 1:]
        type_dx = np.max(np.abs(np.diff(foreground_types, axis=1, prepend=foreground_types[:, :1])), axis=-1)
        type_dy = np.max(np.abs(np.diff(foreground_types, axis=0, prepend=foreground_types[:1])), axis=-1)
        boundary = np.clip(boundary + type_boundary_weight * np.maximum(type_dx, type_dy), 0, 1)

    distance = -cv2.GaussianBlur((1.0 - boundary) * nucleus, (3, 3), 0)
    ridges = boundary >= boundary_threshold
    markers = nucleus.astype(np.int32) - ridges.astype(np.int32)
    markers[markers < 0] = 0
    markers = ndi.binary_fill_holes(markers).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    markers = cv2.morphologyEx(markers, cv2.MORPH_OPEN, kernel)
    marker_map, _ = ndi.label(markers)
    marker_map = remove_small_objects(marker_map, min_size=min_size)
    if marker_map.max() == 0:
        marker_map, _ = ndi.label(nucleus)
    instance_map = watershed(distance, markers=marker_map, mask=nucleus).astype(np.int32)
    instance_map = _remove_small_instances(instance_map, min_size)
    instance_types = assign_instance_types(instance_map, type_probability)
    return instance_map, instance_types


def _remove_small_instances(instance_map: np.ndarray, min_size: int) -> np.ndarray:
    output = np.zeros_like(instance_map, dtype=np.int32)
    next_id = 1
    for instance_id in np.unique(instance_map)[1:]:
        mask = instance_map == instance_id
        if mask.sum() >= min_size:
            output[mask] = next_id
            next_id += 1
    return output


def assign_instance_types(instance_map: np.ndarray, type_probability: np.ndarray | None) -> np.ndarray:
    types = np.zeros(int(instance_map.max()) + 1, dtype=np.int16)
    if type_probability is None:
        return types
    pixel_types = type_probability.argmax(axis=-1)
    for instance_id in range(1, len(types)):
        pixels = pixel_types[instance_map == instance_id]
        if len(pixels):
            labels, counts = np.unique(pixels, return_counts=True)
            order = np.argsort(-counts)
            selected = int(labels[order[0]])
            if selected == 0 and len(order) > 1:
                selected = int(labels[order[1]])
            types[instance_id] = selected
    return types
