from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from skimage.segmentation import find_boundaries


TYPE_COLORS = np.asarray(
    [(0, 0, 0), (230, 57, 70), (255, 190, 11), (58, 134, 255), (131, 56, 236), (6, 214, 160)],
    dtype=np.uint8,
)

UNKNOWN_TYPE_COLOR = np.asarray((180, 180, 180), dtype=np.uint8)


def instance_boundary_overlay(
    image: np.ndarray,
    instance_map: np.ndarray,
    boundary_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw instance contours without encoding any nuclear type information."""
    output = image.copy()
    output[find_boundaries(instance_map, mode="inner")] = np.asarray(
        boundary_color, dtype=np.uint8
    )
    return output


def instance_label_image(instance_map: np.ndarray) -> np.ndarray:
    """Render nuclei as white instances on black, with black separating contours."""
    output = np.zeros((*instance_map.shape, 3), dtype=np.uint8)
    output[instance_map > 0] = 255
    output[find_boundaries(instance_map, mode="inner")] = 0
    return output


def type_label_image(
    instance_map: np.ndarray,
    instance_types: np.ndarray,
) -> np.ndarray:
    """Render nuclear classes using the fixed palette on a black background."""
    output = np.zeros((*instance_map.shape, 3), dtype=np.uint8)
    for instance_id in range(1, int(instance_map.max()) + 1):
        mask = instance_map == instance_id
        if not mask.any():
            continue
        class_id = int(instance_types[instance_id]) if instance_id < len(instance_types) else -1
        color = TYPE_COLORS[class_id] if 0 < class_id < len(TYPE_COLORS) else UNKNOWN_TYPE_COLOR
        output[mask] = color
    output[find_boundaries(instance_map, mode="inner")] = 0
    return output


def instance_overlay(
    image: np.ndarray,
    instance_map: np.ndarray,
    instance_types: np.ndarray,
    alpha: float = 0.28,
) -> np.ndarray:
    output = image.astype(np.float32).copy()
    for instance_id in range(1, int(instance_map.max()) + 1):
        mask = instance_map == instance_id
        if not mask.any():
            continue
        class_id = int(instance_types[instance_id]) if instance_id < len(instance_types) else -1
        color = TYPE_COLORS[class_id] if 0 < class_id < len(TYPE_COLORS) else UNKNOWN_TYPE_COLOR
        output[mask] = (1.0 - alpha) * output[mask] + alpha * color
        # Type panels use class-coloured contours as well as a light fill.
        output[find_boundaries(mask, mode="inner")] = color
    return np.clip(output, 0, 255).astype(np.uint8)


def save_prediction_visualization(
    path: Path,
    image: np.ndarray,
    true_map: np.ndarray,
    true_types: np.ndarray,
    pred_map: np.ndarray,
    pred_types: np.ndarray,
    nucleus_probability: np.ndarray,
    hv_map: np.ndarray,
    type_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(21, 10), constrained_layout=True)
    panels = (
        (instance_label_image(true_map), "Ground truth instances", None, None),
        (instance_label_image(pred_map), "Predicted instances", None, None),
        (type_label_image(true_map, true_types), "Ground truth nuclear types", None, None),
        (type_label_image(pred_map, pred_types), "Predicted nuclear types", None, None),
        (image, "Input H&E", None, None),
        (nucleus_probability, "Nucleus probability", "magma", (0, 1)),
        (hv_map[..., 0], "Horizontal centroid distance", "coolwarm", (-1, 1)),
        (hv_map[..., 1], "Vertical centroid distance", "coolwarm", (-1, 1)),
    )
    for axis, (panel, title, cmap, limits) in zip(axes.flat, panels):
        kwargs = {"cmap": cmap} if cmap else {}
        if limits:
            kwargs.update(vmin=limits[0], vmax=limits[1])
        rendered = axis.imshow(panel, **kwargs)
        axis.set_title(title)
        axis.axis("off")
        if cmap:
            figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.02)
    legend = [
        Patch(facecolor=TYPE_COLORS[index] / 255.0, label=name)
        for index, name in enumerate(type_names[1:], start=1)
    ]
    if np.any(true_types[1:] < 0) or np.any(pred_types[1:] <= 0):
        legend.append(Patch(facecolor=UNKNOWN_TYPE_COLOR / 255.0, label="unknown / ambiguous"))
    figure.legend(handles=legend, loc="lower center", ncol=max(1, len(legend)), frameon=False)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_training_curves(history_path: Path, output_path: Path) -> None:
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    if not rows:
        return
    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(epochs, [row["train"]["loss"] for row in rows], label="train")
    axes[0, 0].plot(epochs, [row["val"]["loss"] for row in rows], label="validation")
    axes[0, 0].set_title("Total loss")
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, [row["train"]["dice"] for row in rows], label="train")
    axes[0, 1].plot(epochs, [row["val"]["dice"] for row in rows], label="validation")
    axes[0, 1].set_title("Nuclear pixel Dice")
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, [row["train"]["type_pixel_accuracy"] for row in rows], label="train")
    axes[1, 0].plot(epochs, [row["val"]["type_pixel_accuracy"] for row in rows], label="validation")
    axes[1, 0].set_title("Foreground type pixel accuracy")
    axes[1, 0].legend()
    for name in ("np_ce", "np_dice", "hv_mse", "hv_gradient", "type_ce", "type_dice"):
        axes[1, 1].plot(epochs, [row["val"][name] for row in rows], label=name)
    axes[1, 1].set_title("Validation loss components")
    axes[1, 1].legend(fontsize=8, ncol=2)
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
