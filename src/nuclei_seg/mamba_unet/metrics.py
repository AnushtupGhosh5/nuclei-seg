from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .postprocess import instance_type_vector


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / (denominator + 1.0e-12))


def remap_label(values: np.ndarray) -> np.ndarray:
    ids = np.unique(values)
    output = np.zeros(values.shape, np.int32)
    for new, old in enumerate(ids[ids != 0], 1):
        output[values == old] = new
    return output


def _contingency(true: np.ndarray, predicted: np.ndarray):
    true, predicted = remap_label(true), remap_label(predicted)
    number_true, number_predicted = int(true.max()), int(predicted.max())
    table = np.bincount(
        (true.ravel() * (number_predicted + 1) + predicted.ravel()).astype(np.int64),
        minlength=(number_true + 1) * (number_predicted + 1),
    ).reshape(number_true + 1, number_predicted + 1)
    intersection = table[1:, 1:].astype(np.float64)
    true_area = table[1:, :].sum(1).astype(np.float64)
    predicted_area = table[:, 1:].sum(0).astype(np.float64)
    union = true_area[:, None] + predicted_area[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return true, predicted, intersection, union, iou, true_area, predicted_area


def instance_metrics(true: np.ndarray, predicted: np.ndarray, threshold: float = 0.5) -> dict:
    true, predicted, intersection, union, iou, true_area, predicted_area = _contingency(
        true, predicted
    )
    number_true, number_predicted = len(true_area), len(predicted_area)
    if number_true and number_predicted:
        row, column = linear_sum_assignment(-iou)
        valid = iou[row, column] > threshold
        row, column = row[valid], column[valid]
        matched_iou = iou[row, column]
    else:
        row = column = np.array([], dtype=int)
        matched_iou = np.array([])
    tp, fp, fn = len(row), number_predicted - len(row), number_true - len(row)
    dq = safe_ratio(tp, tp + 0.5 * fp + 0.5 * fn)
    sq = safe_ratio(matched_iou.sum(), tp)

    if number_true and number_predicted:
        best = iou.argmax(1)
        valid_true = iou.max(1) > 0
        true_indices, predicted_indices = np.flatnonzero(valid_true), best[valid_true]
        aji_intersection = intersection[true_indices, predicted_indices].sum()
        aji_union = union[true_indices, predicted_indices].sum()
        used = set(predicted_indices.tolist())
        aji_union += predicted_area[[i for i in range(number_predicted) if i not in used]].sum()
        aji_union += true_area[~valid_true].sum()
        aji = safe_ratio(aji_intersection, aji_union)

        plus_row, plus_column = linear_sum_assignment(-iou)
        nonzero = iou[plus_row, plus_column] > 0
        plus_row, plus_column = plus_row[nonzero], plus_column[nonzero]
        plus_intersection = intersection[plus_row, plus_column].sum()
        plus_union = union[plus_row, plus_column].sum()
        plus_union += true_area[
            [i for i in range(number_true) if i not in set(plus_row.tolist())]
        ].sum()
        plus_union += predicted_area[
            [i for i in range(number_predicted) if i not in set(plus_column.tolist())]
        ].sum()
        aji_plus = safe_ratio(plus_intersection, plus_union)
    else:
        aji = aji_plus = float(number_true == 0 and number_predicted == 0)

    return {
        "dq": dq,
        "sq": sq,
        "pq": dq * sq,
        "aji": aji,
        "aji_plus": aji_plus,
        "det_precision": safe_ratio(tp, tp + fp),
        "det_recall": safe_ratio(tp, tp + fn),
        "det_f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "paired_iou_sum": float(matched_iou.sum()),
        "paired_true": row + 1,
        "paired_pred": column + 1,
    }


def _instance_centroids(instance_map: np.ndarray) -> np.ndarray:
    points = []
    for instance_id in range(1, int(instance_map.max()) + 1):
        y, x = np.where(instance_map == instance_id)
        points.append([x.mean(), y.mean()] if len(x) else [np.nan, np.nan])
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def centroid_pairing(true: np.ndarray, predicted: np.ndarray, radius: float):
    true_points, predicted_points = _instance_centroids(true), _instance_centroids(predicted)
    if len(true_points) and len(predicted_points):
        distance = np.sqrt(((true_points[:, None, :] - predicted_points[None, :, :]) ** 2).sum(-1))
        row, column = linear_sum_assignment(distance)
        valid = distance[row, column] <= radius
        row, column = row[valid], column[valid]
    else:
        row = column = np.array([], dtype=int)
    unmatched_true = np.array(
        [i for i in range(len(true_points)) if i not in set(row.tolist())], dtype=int
    )
    unmatched_predicted = np.array(
        [i for i in range(len(predicted_points)) if i not in set(column.tolist())], dtype=int
    )
    return row + 1, column + 1, unmatched_true + 1, unmatched_predicted + 1


def instances_of_type(
    instance_map: np.ndarray, type_map: np.ndarray, class_id: int
) -> np.ndarray:
    base = remap_label(instance_map)
    vector = instance_type_vector(base, type_map)
    keep = np.flatnonzero(vector == class_id)
    return remap_label(np.where(np.isin(base, keep), base, 0))


def panoptic_from_totals(counts: dict) -> dict[str, float]:
    dq = safe_ratio(counts["tp"], counts["tp"] + 0.5 * counts["fp"] + 0.5 * counts["fn"])
    sq = safe_ratio(counts["paired_iou_sum"], counts["tp"])
    return {"dq": dq, "sq": sq, "pq": dq * sq}


class MetricAccumulator:
    """Dataset-level aggregation matching the executed notebook's definitions."""

    default_class_names = {0: "background", 1: "other", 2: "lymphocyte", 3: "epithelial"}

    def __init__(
        self,
        match_iou: float,
        magnification: int,
        class_names: dict[int, str] | None = None,
    ) -> None:
        self.match_iou = match_iou
        self.radius = 12 if magnification == 40 else 6
        self.magnification = magnification
        self.class_names = dict(class_names or self.default_class_names)
        self.num_classes = len(self.class_names)
        self.pixel_confusion = np.zeros((self.num_classes, self.num_classes), np.int64)
        self.instance_counts = {
            class_id: defaultdict(int) for class_id in range(1, self.num_classes)
        }
        self.centroid_totals = defaultdict(int)
        self.binary_totals = defaultdict(float)
        self.multiclass_totals = defaultdict(float)
        self.class_panoptic = {
            class_id: defaultdict(float) for class_id in range(1, self.num_classes)
        }
        self.classification_correct = 0
        self.classification_total = 0
        self.per_image: list[dict] = []

    def update(
        self,
        stem: str,
        true_instance: np.ndarray,
        true_type: np.ndarray,
        predicted_instance: np.ndarray,
        predicted_type: np.ndarray,
        inference_seconds: float,
    ) -> None:
        true_flat, predicted_flat = true_type.ravel(), predicted_type.ravel()
        self.pixel_confusion += np.bincount(
            true_flat * self.num_classes + predicted_flat,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)
        true_binary, predicted_binary = true_instance > 0, predicted_instance > 0
        intersection = np.logical_and(true_binary, predicted_binary).sum()
        dice = safe_ratio(2 * intersection, true_binary.sum() + predicted_binary.sum())
        iou = safe_ratio(intersection, np.logical_or(true_binary, predicted_binary).sum())

        binary = instance_metrics(true_instance, predicted_instance, self.match_iou)
        for key in ("tp", "fp", "fn", "paired_iou_sum"):
            self.binary_totals[key] += binary[key]
        image_multiclass = defaultdict(float)
        for class_id in range(1, self.num_classes):
            result = instance_metrics(
                instances_of_type(true_instance, true_type, class_id),
                instances_of_type(predicted_instance, predicted_type, class_id),
                self.match_iou,
            )
            for key in ("tp", "fp", "fn", "paired_iou_sum"):
                image_multiclass[key] += result[key]
                self.class_panoptic[class_id][key] += result[key]
        multiclass = panoptic_from_totals(image_multiclass)
        for key in ("tp", "fp", "fn", "paired_iou_sum"):
            self.multiclass_totals[key] += image_multiclass[key]

        remapped_true, remapped_predicted = remap_label(true_instance), remap_label(predicted_instance)
        paired_true, paired_predicted, unmatched_true, unmatched_predicted = centroid_pairing(
            remapped_true, remapped_predicted, self.radius
        )
        centroid_tp, centroid_fp, centroid_fn = (
            len(paired_true),
            len(unmatched_predicted),
            len(unmatched_true),
        )
        self.centroid_totals["tp"] += centroid_tp
        self.centroid_totals["fp"] += centroid_fp
        self.centroid_totals["fn"] += centroid_fn
        self.per_image.append(
            {
                "stem": stem,
                "binary_dice": dice,
                "binary_iou": iou,
                **{key: binary[key] for key in ("dq", "sq", "pq", "aji", "aji_plus", "det_precision", "det_recall", "det_f1")},
                "binary_dq": binary["dq"],
                "binary_sq": binary["sq"],
                "binary_pq": binary["pq"],
                "multiclass_dq": multiclass["dq"],
                "multiclass_sq": multiclass["sq"],
                "multiclass_pq": multiclass["pq"],
                "centroid_det_precision": safe_ratio(centroid_tp, centroid_tp + centroid_fp),
                "centroid_det_recall": safe_ratio(centroid_tp, centroid_tp + centroid_fn),
                "centroid_det_f1": safe_ratio(2 * centroid_tp, 2 * centroid_tp + centroid_fp + centroid_fn),
                "inference_seconds_including_postprocess": inference_seconds,
            }
        )

        true_vector = instance_type_vector(remapped_true, true_type)
        predicted_vector = instance_type_vector(remapped_predicted, predicted_type)
        for true_id, predicted_id in zip(paired_true, paired_predicted):
            self.classification_total += 1
            self.classification_correct += int(true_vector[true_id] == predicted_vector[predicted_id])
            for class_id in range(1, self.num_classes):
                if true_vector[true_id] == class_id and predicted_vector[predicted_id] == class_id:
                    self.instance_counts[class_id]["tp"] += 1
                elif true_vector[true_id] == class_id:
                    self.instance_counts[class_id]["fn_classification"] += 1
                elif predicted_vector[predicted_id] == class_id:
                    self.instance_counts[class_id]["fp_classification"] += 1
        for true_id in unmatched_true:
            if int(true_vector[true_id]) in self.instance_counts:
                self.instance_counts[int(true_vector[true_id])]["fn_detection"] += 1
        for predicted_id in unmatched_predicted:
            if int(predicted_vector[predicted_id]) in self.instance_counts:
                self.instance_counts[int(predicted_vector[predicted_id])]["fp_detection"] += 1

    def finalize(self) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        per_image = pd.DataFrame(self.per_image)
        confusion = self.pixel_confusion
        total = confusion.sum()
        pixel_rows = []
        for class_id, name in self.class_names.items():
            tp = confusion[class_id, class_id]
            fp = confusion[:, class_id].sum() - tp
            fn = confusion[class_id, :].sum() - tp
            tn = total - tp - fp - fn
            pixel_rows.append(
                {
                    "class_id": class_id,
                    "class": name,
                    "precision": safe_ratio(tp, tp + fp),
                    "recall": safe_ratio(tp, tp + fn),
                    "f1_dice": safe_ratio(2 * tp, 2 * tp + fp + fn),
                    "iou": safe_ratio(tp, tp + fp + fn),
                    "specificity": safe_ratio(tn, tn + fp),
                    "one_vs_rest_accuracy": safe_ratio(tp + tn, total),
                    "support_pixels": int(confusion[class_id, :].sum()),
                }
            )
        pixel = pd.DataFrame(pixel_rows)

        instance_rows = []
        for class_id in range(1, self.num_classes):
            counts = self.instance_counts[class_id]
            tp = counts["tp"]
            fp = counts["fp_classification"] + counts["fp_detection"]
            fn = counts["fn_classification"] + counts["fn_detection"]
            official_denominator = (
                2 * tp
                + 2 * counts["fp_classification"]
                + 2 * counts["fn_classification"]
                + counts["fp_detection"]
                + counts["fn_detection"]
            )
            instance_rows.append(
                {
                    "class_id": class_id,
                    "class": self.class_names[class_id],
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": safe_ratio(tp, tp + fp),
                    "recall": safe_ratio(tp, tp + fn),
                    "f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
                    "official_weighted_f1": safe_ratio(2 * tp, official_denominator),
                }
            )
        instance = pd.DataFrame(instance_rows)

        panoptic_rows = []
        for class_id in range(1, self.num_classes):
            counts = self.class_panoptic[class_id]
            panoptic_rows.append(
                {
                    "class_id": class_id,
                    "class": self.class_names[class_id],
                    **panoptic_from_totals(counts),
                    **{key: float(counts[key]) for key in ("tp", "fp", "fn", "paired_iou_sum")},
                }
            )
        panoptic = pd.DataFrame(panoptic_rows)

        background_tp = confusion[0, 0]
        foreground_tp = confusion[1:, 1:].sum()
        binary_fp = confusion[0, 1:].sum()
        binary_fn = confusion[1:, 0].sum()
        binary_global = panoptic_from_totals(self.binary_totals)
        multiclass_global = panoptic_from_totals(self.multiclass_totals)
        summary = {
            f"{column}_image_mean": float(per_image[column].mean())
            for column in per_image.columns
            if column != "stem"
        }
        summary.update(
            {
                "binary_pixel_accuracy_global": safe_ratio(background_tp + foreground_tp, total),
                "binary_precision_global": safe_ratio(foreground_tp, foreground_tp + binary_fp),
                "binary_recall_global": safe_ratio(foreground_tp, foreground_tp + binary_fn),
                "binary_f1_dice_global": safe_ratio(
                    2 * foreground_tp, 2 * foreground_tp + binary_fp + binary_fn
                ),
                "binary_iou_global": safe_ratio(foreground_tp, foreground_tp + binary_fp + binary_fn),
                "binary_dq_global": binary_global["dq"],
                "binary_sq_global": binary_global["sq"],
                "binary_pq_global": binary_global["pq"],
                "multiclass_dq_global": multiclass_global["dq"],
                "multiclass_sq_global": multiclass_global["sq"],
                "multiclass_pq_global": multiclass_global["pq"],
                "multiclass_macro_dq_classwise": float(panoptic.dq.mean()),
                "multiclass_macro_sq_classwise": float(panoptic.sq.mean()),
                "multiclass_macro_pq_classwise": float(panoptic.pq.mean()),
                "multiclass_pixel_accuracy_global": safe_ratio(np.trace(confusion), total),
                "pixel_macro_precision_no_background": float(pixel.loc[pixel.class_id > 0, "precision"].mean()),
                "pixel_macro_recall_no_background": float(pixel.loc[pixel.class_id > 0, "recall"].mean()),
                "pixel_macro_f1_no_background": float(pixel.loc[pixel.class_id > 0, "f1_dice"].mean()),
                "pixel_mean_iou_no_background": float(pixel.loc[pixel.class_id > 0, "iou"].mean()),
                "pixel_macro_f1_including_background": float(pixel.f1_dice.mean()),
                "pixel_mean_iou_including_background": float(pixel.iou.mean()),
                "instance_classification_accuracy_matched": safe_ratio(
                    self.classification_correct, self.classification_total
                ),
                "instance_macro_f1_detection_aware": float(instance.f1.mean()),
                "instance_macro_official_weighted_f1": float(instance.official_weighted_f1.mean()),
                "centroid_detection_precision_global": safe_ratio(
                    self.centroid_totals["tp"], self.centroid_totals["tp"] + self.centroid_totals["fp"]
                ),
                "centroid_detection_recall_global": safe_ratio(
                    self.centroid_totals["tp"], self.centroid_totals["tp"] + self.centroid_totals["fn"]
                ),
                "centroid_detection_f1_global": safe_ratio(
                    2 * self.centroid_totals["tp"],
                    2 * self.centroid_totals["tp"]
                    + self.centroid_totals["fp"]
                    + self.centroid_totals["fn"],
                ),
                "inference_seconds_total_including_postprocess": float(
                    per_image.inference_seconds_including_postprocess.sum()
                ),
                "inference_images_per_second_including_postprocess": safe_ratio(
                    len(per_image), per_image.inference_seconds_including_postprocess.sum()
                ),
                "n_images": len(per_image),
                "match_iou": self.match_iou,
                "magnification": self.magnification,
            }
        )
        return summary, per_image, pixel, instance, panoptic
