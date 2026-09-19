from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment


def instance_types_from_map(instance_map: np.ndarray, type_map: np.ndarray) -> np.ndarray:
    output = np.zeros(int(instance_map.max()) + 1, dtype=np.int16)
    for instance_id in range(1, len(output)):
        values = type_map[instance_map == instance_id]
        values = values[values > 0]
        if len(values):
            labels, counts = np.unique(values, return_counts=True)
            output[instance_id] = labels[np.argmax(counts)]
        else:
            output[instance_id] = -1
    return output


def binary_metrics(true_map: np.ndarray, pred_map: np.ndarray) -> dict[str, float]:
    true = true_map > 0
    pred = pred_map > 0
    intersection = np.logical_and(true, pred).sum()
    return {
        "dice": float((2 * intersection + 1.0e-6) / (true.sum() + pred.sum() + 1.0e-6)),
        "iou": float((intersection + 1.0e-6) / (np.logical_or(true, pred).sum() + 1.0e-6)),
    }


def pair_instances(
    true_map: np.ndarray, pred_map: np.ndarray, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true_ids = np.unique(true_map)
    pred_ids = np.unique(pred_map)
    true_ids, pred_ids = true_ids[true_ids > 0], pred_ids[pred_ids > 0]
    iou = np.zeros((len(true_ids), len(pred_ids)), dtype=np.float64)
    pred_area = {pid: int((pred_map == pid).sum()) for pid in pred_ids}
    for true_index, true_id in enumerate(true_ids):
        true_mask = true_map == true_id
        overlap_ids, intersections = np.unique(pred_map[true_mask], return_counts=True)
        true_area = int(true_mask.sum())
        for pred_id, intersection in zip(overlap_ids, intersections):
            if pred_id == 0:
                continue
            pred_index = np.searchsorted(pred_ids, pred_id)
            union = true_area + pred_area[pred_id] - int(intersection)
            iou[true_index, pred_index] = intersection / max(union, 1)
    if iou.size:
        rows, cols = linear_sum_assignment(-iou)
        keep = iou[rows, cols] > threshold
        rows, cols = rows[keep], cols[keep]
    else:
        rows = cols = np.empty(0, dtype=np.int64)
    pairs = np.column_stack((true_ids[rows], pred_ids[cols])).astype(np.int64)
    paired_iou = iou[rows, cols]
    unmatched_true = np.setdiff1d(true_ids, pairs[:, 0] if len(pairs) else [])
    unmatched_pred = np.setdiff1d(pred_ids, pairs[:, 1] if len(pairs) else [])
    return pairs, paired_iou, unmatched_true, unmatched_pred


def panoptic_quality(true_map: np.ndarray, pred_map: np.ndarray) -> dict[str, float]:
    pairs, paired_iou, unmatched_true, unmatched_pred = pair_instances(true_map, pred_map)
    tp, fp, fn = len(pairs), len(unmatched_pred), len(unmatched_true)
    denominator = tp + 0.5 * fp + 0.5 * fn
    dq = tp / denominator if denominator else 1.0
    sq = float(paired_iou.mean()) if tp else 0.0
    return {"dq": float(dq), "sq": sq, "pq": float(dq * sq)}


def aggregated_jaccard_plus(true_map: np.ndarray, pred_map: np.ndarray) -> float:
    pairs, _, unmatched_true, unmatched_pred = pair_instances(true_map, pred_map, threshold=0.0)
    intersection, union = 0, 0
    for true_id, pred_id in pairs:
        true_mask, pred_mask = true_map == true_id, pred_map == pred_id
        intersection += int(np.logical_and(true_mask, pred_mask).sum())
        union += int(np.logical_or(true_mask, pred_mask).sum())
    union += sum(int((true_map == item).sum()) for item in unmatched_true)
    union += sum(int((pred_map == item).sum()) for item in unmatched_pred)
    return float(intersection / union) if union else 1.0


def aggregated_jaccard(true_map: np.ndarray, pred_map: np.ndarray) -> float:
    """Original AJI with best-overlap pairing (predictions may be reused)."""
    true_ids = np.unique(true_map)[1:]
    pred_ids = np.unique(pred_map)[1:]
    pred_areas = {int(item): int((pred_map == item).sum()) for item in pred_ids}
    paired_pred: set[int] = set()
    intersection = union = 0
    for true_id in true_ids:
        true_mask = true_map == true_id
        candidates = np.unique(pred_map[true_mask])
        best = None
        for pred_id in candidates[candidates > 0]:
            inter = int(np.logical_and(true_mask, pred_map == pred_id).sum())
            candidate_union = int(true_mask.sum()) + pred_areas[int(pred_id)] - inter
            iou = inter / max(candidate_union, 1)
            if best is None or iou > best[0]:
                best = (iou, int(pred_id), inter, candidate_union)
        if best is None:
            union += int(true_mask.sum())
            continue
        paired_pred.add(best[1])
        intersection += best[2]
        union += best[3]
    union += sum(pred_areas[item] for item in pred_ids if int(item) not in paired_pred)
    return float(intersection / union) if union else 1.0


def pair_instances_by_centroid(
    true_map: np.ndarray, pred_map: np.ndarray, radius: float = 12.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_ids = np.unique(true_map)[1:]
    pred_ids = np.unique(pred_map)[1:]
    true_centers = (
        np.asarray(ndi.center_of_mass(true_map > 0, true_map, true_ids))[:, ::-1]
        if len(true_ids)
        else np.empty((0, 2))
    )
    pred_centers = (
        np.asarray(ndi.center_of_mass(pred_map > 0, pred_map, pred_ids))[:, ::-1]
        if len(pred_ids)
        else np.empty((0, 2))
    )
    if len(true_ids) and len(pred_ids):
        distances = np.linalg.norm(true_centers[:, None] - pred_centers[None], axis=-1)
        rows, cols = linear_sum_assignment(distances)
        keep = distances[rows, cols] < radius
        rows, cols = rows[keep], cols[keep]
    else:
        rows = cols = np.empty(0, dtype=np.int64)
    pairs = np.column_stack((true_ids[rows], pred_ids[cols])).astype(np.int64)
    return (
        pairs,
        np.setdiff1d(true_ids, pairs[:, 0] if len(pairs) else []),
        np.setdiff1d(pred_ids, pairs[:, 1] if len(pairs) else []),
    )


def evaluate_instances(
    true_map: np.ndarray,
    pred_map: np.ndarray,
    true_types: np.ndarray,
    pred_types: np.ndarray,
    type_names: list[str],
) -> dict[str, float]:
    metrics = binary_metrics(true_map, pred_map)
    metrics.update(panoptic_quality(true_map, pred_map))
    metrics["aji"] = aggregated_jaccard(true_map, pred_map)
    metrics["aji_plus"] = aggregated_jaccard_plus(true_map, pred_map)
    pairs, _, unmatched_true, unmatched_pred = pair_instances(true_map, pred_map)
    metrics["detection_f1"] = float(
        2 * len(pairs) / max(2 * len(pairs) + len(unmatched_true) + len(unmatched_pred), 1)
    )

    paired_true = true_types[pairs[:, 0]] if len(pairs) else np.empty(0)
    paired_pred = pred_types[pairs[:, 1]] if len(pairs) else np.empty(0)
    valid_pairs = paired_true > 0
    ignored_pred_ids = pairs[~valid_pairs, 1] if len(pairs) else np.empty(0, dtype=np.int64)
    class_f1 = []
    for class_id, name in enumerate(type_names[1:], start=1):
        tp = int(np.sum(valid_pairs & (paired_true == class_id) & (paired_pred == class_id)))
        fp = int(np.sum(valid_pairs & (paired_true != class_id) & (paired_pred == class_id)))
        fn = int(np.sum(valid_pairs & (paired_true == class_id) & (paired_pred != class_id)))
        fp += int(np.sum(pred_types[unmatched_pred] == class_id))
        fn += int(np.sum(true_types[unmatched_true] == class_id))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        metrics[f"f1_{name}"] = float(f1)
        if np.any(true_types[1:] == class_id):
            class_f1.append(f1)

        true_class_map = np.where(
            np.isin(true_map, np.flatnonzero(true_types == class_id)), true_map, 0
        )
        pred_ids = np.setdiff1d(np.flatnonzero(pred_types == class_id), ignored_pred_ids)
        pred_class_map = np.where(np.isin(pred_map, pred_ids), pred_map, 0)
        metrics[f"pq_{name}"] = panoptic_quality(true_class_map, pred_class_map)["pq"]
    metrics["type_macro_f1"] = float(np.mean(class_f1)) if class_f1 else 0.0
    return metrics


def type_classification_counts(
    true_map: np.ndarray,
    pred_map: np.ndarray,
    true_types: np.ndarray,
    pred_types: np.ndarray,
    num_types: int,
) -> np.ndarray:
    """Return per-class TP/FP/FN counts, excluding ambiguous GT instances."""
    pairs, _, unmatched_true, unmatched_pred = pair_instances(true_map, pred_map)
    paired_true = true_types[pairs[:, 0]] if len(pairs) else np.empty(0)
    paired_pred = pred_types[pairs[:, 1]] if len(pairs) else np.empty(0)
    valid = paired_true > 0
    counts = np.zeros((num_types, 3), dtype=np.int64)
    for class_id in range(1, num_types):
        counts[class_id, 0] = np.sum(valid & (paired_true == class_id) & (paired_pred == class_id))
        counts[class_id, 1] = np.sum(valid & (paired_true != class_id) & (paired_pred == class_id))
        counts[class_id, 2] = np.sum(valid & (paired_true == class_id) & (paired_pred != class_id))
        counts[class_id, 1] += np.sum(pred_types[unmatched_pred] == class_id)
        counts[class_id, 2] += np.sum(true_types[unmatched_true] == class_id)
    return counts


def hover_type_classification_counts(
    true_map: np.ndarray,
    pred_map: np.ndarray,
    true_types: np.ndarray,
    pred_types: np.ndarray,
    num_types: int,
) -> np.ndarray:
    """TP, paired FP/FN, detection FP/FN using official 12px pairing."""
    pairs, unmatched_true, unmatched_pred = pair_instances_by_centroid(true_map, pred_map)
    paired_true = true_types[pairs[:, 0]] if len(pairs) else np.empty(0)
    paired_pred = pred_types[pairs[:, 1]] if len(pairs) else np.empty(0)
    valid = paired_true > 0
    counts = np.zeros((num_types, 5), dtype=np.int64)
    for class_id in range(1, num_types):
        counts[class_id, 0] = np.sum(valid & (paired_true == class_id) & (paired_pred == class_id))
        counts[class_id, 1] = np.sum(valid & (paired_true != class_id) & (paired_pred == class_id))
        counts[class_id, 2] = np.sum(valid & (paired_true == class_id) & (paired_pred != class_id))
        counts[class_id, 3] = np.sum(pred_types[unmatched_pred] == class_id)
        counts[class_id, 4] = np.sum(true_types[unmatched_true] == class_id)
    return counts
