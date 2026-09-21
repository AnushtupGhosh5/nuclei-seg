from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


def smile_watershed(
    nucleus_probability: np.ndarray,
    hv_map: np.ndarray,
    nucleus_threshold: float = 0.5,
    boundary_threshold: float = 0.4,
    min_size: int = 10,
) -> np.ndarray:
    """SMILE/HoVer marker watershed used by the executed reference notebook."""
    nucleus = np.asarray(nucleus_probability >= nucleus_threshold, dtype=np.int32)
    nucleus = ndi.label(nucleus)[0]
    nucleus = remove_small_objects(nucleus, min_size=min_size)
    nucleus[nucleus > 0] = 1

    horizontal = cv2.normalize(
        hv_map[..., 0].astype(np.float32), None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    vertical = cv2.normalize(
        hv_map[..., 1].astype(np.float32), None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    horizontal = cv2.Sobel(horizontal, cv2.CV_64F, 1, 0, ksize=21)
    vertical = cv2.Sobel(vertical, cv2.CV_64F, 0, 1, ksize=21)
    horizontal = 1 - cv2.normalize(
        horizontal, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    vertical = 1 - cv2.normalize(vertical, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    boundary = np.maximum(horizontal, vertical)
    boundary = boundary - (1 - nucleus)
    boundary[boundary < 0] = 0
    distance = -cv2.GaussianBlur((1.0 - boundary) * nucleus, (3, 3), 0)

    boundary = np.asarray(boundary >= boundary_threshold, dtype=np.int32)
    marker = nucleus - boundary
    marker[marker < 0] = 0
    marker = ndi.binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = ndi.label(marker)[0]
    marker = remove_small_objects(marker, min_size=min_size)
    return watershed(distance, markers=marker, mask=nucleus).astype(np.int32)


def instance_type_vector(instance_map: np.ndarray, type_map: np.ndarray) -> np.ndarray:
    output = np.zeros(int(instance_map.max()) + 1, dtype=np.uint8)
    for instance_id in range(1, len(output)):
        values = type_map[instance_map == instance_id]
        values = values[values > 0]
        if len(values):
            output[instance_id] = np.bincount(values, minlength=4).argmax()
    return output


def instance_type_map(instance_map: np.ndarray, pixel_types: np.ndarray) -> np.ndarray:
    output = np.zeros_like(pixel_types, dtype=np.uint8)
    vector = instance_type_vector(instance_map, pixel_types)
    for instance_id in range(1, len(vector)):
        output[instance_map == instance_id] = vector[instance_id]
    return output

