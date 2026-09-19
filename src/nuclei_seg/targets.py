from __future__ import annotations

import numpy as np


def remap_instances(instance_map: np.ndarray) -> np.ndarray:
    """Make foreground instance IDs contiguous while preserving background 0."""
    output = np.zeros(instance_map.shape, dtype=np.int32)
    for new_id, old_id in enumerate(np.unique(instance_map)[1:], start=1):
        output[instance_map == old_id] = new_id
    return output


def generate_hv_map(instance_map: np.ndarray) -> np.ndarray:
    """Generate normalized horizontal/vertical centroid offsets in [-1, 1]."""
    instance_map = np.asarray(instance_map)
    hv = np.zeros((*instance_map.shape, 2), dtype=np.float32)
    for instance_id in np.unique(instance_map):
        if instance_id == 0:
            continue
        ys, xs = np.nonzero(instance_map == instance_id)
        if not len(xs):
            continue
        dx = xs.astype(np.float32) - xs.mean(dtype=np.float64)
        dy = ys.astype(np.float32) - ys.mean(dtype=np.float64)
        neg_x = abs(float(dx.min()))
        pos_x = abs(float(dx.max()))
        neg_y = abs(float(dy.min()))
        pos_y = abs(float(dy.max()))
        dx[dx < 0] /= max(neg_x, 1.0)
        dx[dx > 0] /= max(pos_x, 1.0)
        dy[dy < 0] /= max(neg_y, 1.0)
        dy[dy > 0] /= max(pos_y, 1.0)
        hv[ys, xs, 0] = dx
        hv[ys, xs, 1] = dy
    return hv
