"""Train/evaluate Cartesian, PDE, and hybrid scan ablations on GLySAC."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
import zipfile
from collections import defaultdict
from dataclasses import replace
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import (
    RandomPatchDataset,
    build_split,
    detect_type_encoding,
    discover_records,
    read_annotation,
    read_ignore_map,
    read_rgb,
)
from .engine import (
    _loss_weights,
    _sliding_positions,
    ensure_pretrained_checkpoint,
    seed_everything,
)
from .geometric_config import PDEGeometricConfig
from .geometric_loss import PDEGeometricLoss
from .geometric_model import (
    PDEGeometricMambaUNet,
    verify_geometric_output_contract,
)
from .geometric_postprocess import mask_pde_watershed
from .metrics import MetricAccumulator
from .model import load_official_pretraining
from .pde_data import PDETargetDataset
from .pde_field import make_poisson_field
from .pde_scan import build_pde_permutations
from .postprocess import instance_type_map


def _seed_worker(worker_id: int, seed: int) -> None:
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker = torch.utils.data.get_worker_info()
    if worker is not None:
        worker.dataset.rng = np.random.default_rng(worker_seed)


def create_pde_loaders(split, config, encoding):
    patches = 2 if config.smoke_test else config.patches_per_tile
    train_base = RandomPatchDataset(
        split.train, config.patch_size, patches, config.seed, encoding,
        config.min_foreground_fraction, augment=True,
    )
    val_base = RandomPatchDataset(
        split.val, config.patch_size, patches, config.seed, encoding,
        config.min_foreground_fraction, augment=False,
    )
    train_set = PDETargetDataset(
        train_base, iterations=config.pde_target_iterations
    )
    val_set = PDETargetDataset(
        val_base, iterations=config.pde_target_iterations
    )
    common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        worker_init_fn=partial(_seed_worker, seed=config.seed),
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=True,
        generator=generator, **common,
    )
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_base, train_loader, val_loader


@torch.inference_mode()
def validation_loss(model, loader, criterion, device, amp):
    model.eval()
    totals = defaultdict(float)
    for images, instances, types, field, _ in tqdm(
        loader, desc="Geometric validation loss", leave=False
    ):
        images = images.to(device, non_blocking=True)
        instances = instances.to(device, non_blocking=True)
        types = types.to(device, non_blocking=True)
        field = field.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            loss, parts = criterion(
                model(images), instances, types, field
            )
        totals["loss"] += float(loss)
        for key, value in parts.items():
            totals[key] += value
    divisor = max(len(loader), 1)
    return {
        key: value / divisor for key, value in totals.items()
    }


@torch.inference_mode()
def predict_tile(model, image, config, device):
    model.eval()
    size = config.patch_size
    original_height, original_width = image.shape[:2]
    pad_height = max(0, size - original_height)
    pad_width = max(0, size - original_width)
    padded = (
        np.pad(
            image,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode="reflect",
        )
        if pad_height or pad_width else image
    )
    height, width = padded.shape[:2]
    ys = _sliding_positions(height, size, config.infer_overlap)
    xs = _sliding_positions(width, size, config.infer_overlap)
    window = np.maximum(
        np.outer(np.hanning(size), np.hanning(size)).astype(np.float32),
        0.05,
    )
    mask_accumulator = np.zeros((height, width), np.float32)
    field_accumulator = np.zeros((height, width), np.float32)
    type_accumulator = np.zeros(
        (len(config.type_classes), height, width), np.float32
    )
    normalizer = np.zeros((height, width), np.float32)
    coordinates = [(y, x) for y in ys for x in xs]
    for start in range(0, len(coordinates), config.infer_batch_size):
        batch_coordinates = coordinates[
            start : start + config.infer_batch_size
        ]
        batch = np.stack(
            [
                padded[y : y + size, x : x + size].transpose(2, 0, 1)
                for y, x in batch_coordinates
            ]
        )
        output = model(torch.from_numpy(batch).float().to(device))
        masks = output["np"].softmax(1)[:, 1].cpu().numpy()
        fields = output["field"].sigmoid()[:, 0].cpu().numpy()
        types = output["tp"].softmax(1).cpu().numpy()
        for index, (y, x) in enumerate(batch_coordinates):
            area = np.s_[y : y + size, x : x + size]
            mask_accumulator[area] += masks[index] * window
            field_accumulator[area] += fields[index] * window
            type_accumulator[:, area[0], area[1]] += types[index] * window
            normalizer[area] += window
    normalizer = np.maximum(normalizer, 1.0e-6)
    mask_probability = mask_accumulator / normalizer
    field = field_accumulator / normalizer
    pixel_types = (type_accumulator / normalizer).argmax(0).astype(np.uint8)
    instances = mask_pde_watershed(
        mask_probability, field,
        nucleus_threshold=config.nucleus_threshold,
        marker_threshold=config.marker_threshold,
        min_distance=config.minimum_peak_distance,
        min_size=config.minimum_object_size,
        smoothing_sigma=config.field_smoothing_sigma,
    )
    crop = np.s_[:original_height, :original_width]
    return (
        instances[crop], pixel_types[crop],
        mask_probability[crop], field[crop],
    )


def _write_metrics(
    output_dir, split_name, summary, per_image, pixel, instance, panoptic
):
    per_image.to_csv(
        output_dir / f"geometric_per_image_{split_name}.csv", index=False
    )
    pixel.to_csv(
        output_dir / f"geometric_classwise_pixel_{split_name}.csv",
        index=False,
    )
    instance.to_csv(
        output_dir / f"geometric_classwise_instance_{split_name}.csv",
        index=False,
    )
    panoptic.to_csv(
        output_dir / f"geometric_classwise_panoptic_{split_name}.csv",
        index=False,
    )
    (output_dir / f"geometric_metrics_{split_name}.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )


@torch.inference_mode()
def evaluate(
    model, rows, split_name, config, encoding, device, save_predictions
):
    accumulator = MetricAccumulator(
        config.match_iou,
        config.magnification,
        class_names={index: name for index, name in enumerate(config.type_classes)},
    )
    prediction_dir = config.output_dir / f"geometric_predictions_{split_name}"
    if save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    for record in tqdm(rows, desc=f"Geometric evaluate {split_name}"):
        image = read_rgb(record.image)
        true_instance, true_type = read_annotation(record.label, encoding)
        ignore_map = read_ignore_map(record.label)
        if config.smoke_test:
            height, width = image.shape[:2]
            size = min(config.patch_size, height, width)
            top, left = (height - size) // 2, (width - size) // 2
            bounds = np.s_[top : top + size, left : left + size]
            image = image[bounds]
            true_instance = true_instance[bounds]
            true_type = true_type[bounds]
            ignore_map = ignore_map[bounds]
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        predicted_instance, pixel_types, mask_probability, field = (
            predict_tile(model, image, config, device)
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if ignore_map.any():
            true_instance = np.where(ignore_map, 0, true_instance)
            true_type = np.where(ignore_map, 0, true_type)
            predicted_instance = np.where(
                ignore_map, 0, predicted_instance
            ).astype(np.int32)
            pixel_types = np.where(ignore_map, 0, pixel_types).astype(np.uint8)
            mask_probability = np.where(ignore_map, 0.0, mask_probability)
            field = np.where(ignore_map, 0.0, field)
        predicted_type = instance_type_map(predicted_instance, pixel_types)
        accumulator.update(
            record.stem, true_instance, true_type,
            predicted_instance, predicted_type, elapsed,
        )
        if save_predictions:
            np.savez_compressed(
                prediction_dir / f"{record.stem}.npz",
                inst_map=predicted_instance,
                type_map=predicted_type,
                type_pixel_map=pixel_types,
                mask_probability=mask_probability,
                pde_field=field,
                ignore_map=ignore_map.astype(np.uint8),
                inference_seconds=np.asarray(elapsed, np.float64),
            )
    summary, per_image, pixel, instance, panoptic = accumulator.finalize()
    _write_metrics(
        config.output_dir, split_name, summary,
        per_image, pixel, instance, panoptic,
    )
    return summary


def _instance_rgb(instances, seed):
    colors = np.zeros((int(instances.max()) + 1, 3), np.uint8)
    if instances.max():
        colors[1:] = np.random.default_rng(seed).integers(
            45, 256, size=(int(instances.max()), 3), dtype=np.uint8
        )
    return colors[instances]


def save_validation_visualizations(records, config, encoding):
    count = min(config.visualization_count, len(records))
    selected = records[:count]
    figure, axes = plt.subplots(
        count, 9, figsize=(34, 4.0 * count), squeeze=False
    )
    titles = [
        "RGB", "GT instances", "GT PDE", "Predicted mask",
        "Predicted PDE", "Watershed", "Gradient magnitude",
        "Normal direction", "Tangential direction",
    ]
    for row, record in enumerate(selected):
        image = read_rgb(record.image)
        true_instance, _ = read_annotation(record.label, encoding)
        true_field = make_poisson_field(
            true_instance, iterations=config.pde_target_iterations
        )
        with np.load(
            config.output_dir / "geometric_predictions_val" /
            f"{record.stem}.npz"
        ) as prediction:
            instances = prediction["inst_map"]
            mask_probability = prediction["mask_probability"]
            field = prediction["pde_field"]
        gradient_y, gradient_x = np.gradient(field.astype(np.float32))
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        denominator = magnitude + 1.0e-6
        normal_x, normal_y = gradient_x / denominator, gradient_y / denominator
        tangent_x, tangent_y = -normal_y, normal_x
        foreground = mask_probability >= config.nucleus_threshold
        panels = [
            image, _instance_rgb(true_instance, config.seed), true_field,
            mask_probability, field, _instance_rgb(instances, config.seed),
            magnitude,
        ]
        axes[row, 0].imshow(panels[0])
        axes[row, 1].imshow(panels[1])
        axes[row, 2].imshow(panels[2], cmap="magma", vmin=0, vmax=1)
        axes[row, 3].imshow(panels[3], cmap="gray", vmin=0, vmax=1)
        axes[row, 4].imshow(panels[4], cmap="magma", vmin=0, vmax=1)
        axes[row, 5].imshow(panels[5])
        axes[row, 6].imshow(panels[6], cmap="viridis")
        for column, (vx, vy) in enumerate(
            ((normal_x, normal_y), (tangent_x, tangent_y)), start=7
        ):
            axes[row, column].imshow(
                np.where(foreground, field, 0), cmap="magma", vmin=0, vmax=1
            )
            step = max(1, min(field.shape) // 24)
            yy, xx = np.mgrid[
                0 : field.shape[0] : step, 0 : field.shape[1] : step
            ]
            keep = foreground[::step, ::step]
            axes[row, column].quiver(
                xx[keep], yy[keep], vx[::step, ::step][keep],
                -vy[::step, ::step][keep], color="cyan",
                angles="xy", scale_units="xy", scale=0.12,
                width=0.003,
            )
        axes[row, 0].set_ylabel(record.stem, fontsize=8)
        for column, title in enumerate(titles):
            if row == 0:
                axes[row, column].set_title(title)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.tight_layout()
    path = config.output_dir / "geometric_validation_visualizations.png"
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _type_rgb(type_map):
    """Colorize class map with black background and fixed bright class colors."""
    palette = np.asarray(
        [
            [0, 0, 0],
            [0, 230, 255],
            [255, 80, 180],
            [255, 220, 0],
            [80, 255, 120],
        ],
        dtype=np.uint8,
    )
    labels = np.asarray(type_map, dtype=np.int64)
    safe = np.clip(labels, 0, len(palette) - 1)
    return palette[safe]


def save_task_visualizations(records, config, encoding):
    """Save task-focused masks/types without replacing the PDE visualization."""
    count = min(config.visualization_count, len(records))
    selected = records[:count]
    figure, axes = plt.subplots(
        count, 6, figsize=(22, 3.8 * count), squeeze=False,
        facecolor="black",
    )
    titles = [
        "Original image",
        "GT binary mask",
        "Predicted binary (pre-watershed)",
        "Predicted binary (post-watershed)",
        "GT type map",
        "Predicted type map",
    ]
    for row, record in enumerate(selected):
        image = read_rgb(record.image)
        true_instance, true_type = read_annotation(record.label, encoding)
        with np.load(
            config.output_dir / "geometric_predictions_val" /
            f"{record.stem}.npz"
        ) as prediction:
            predicted_instance = prediction["inst_map"]
            predicted_type = prediction["type_map"]
            mask_probability = prediction["mask_probability"]

        gt_binary = true_instance > 0
        pred_before = mask_probability >= config.nucleus_threshold
        pred_after = predicted_instance > 0

        axes[row, 0].imshow(image)
        axes[row, 1].imshow(gt_binary, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(pred_before, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(pred_after, cmap="gray", vmin=0, vmax=1)
        axes[row, 4].imshow(_type_rgb(true_type))
        axes[row, 5].imshow(_type_rgb(predicted_type))

        axes[row, 0].set_ylabel(record.stem, fontsize=8, color="white")
        for column, title in enumerate(titles):
            axis = axes[row, column]
            axis.set_facecolor("black")
            if row == 0:
                axis.set_title(title, color="white", fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

    figure.tight_layout()
    path = config.output_dir / "geometric_task_visualizations.png"
    figure.savefig(
        path, dpi=170, bbox_inches="tight",
        facecolor="black", edgecolor="black",
    )
    plt.close(figure)
    return path


@torch.inference_mode()
def save_scan_debug(model, record, config, device):
    if not model.guidance_enabled:
        note = config.output_dir / "scan_debug_not_applicable.txt"
        note.write_text("Cartesian ablation does not construct PDE scan orders.\n")
        return note
    image = read_rgb(record.image)
    size = config.patch_size
    top = max(0, (image.shape[0] - size) // 2)
    left = max(0, (image.shape[1] - size) // 2)
    patch = image[top : top + size, left : left + size]
    model(torch.from_numpy(patch.transpose(2, 0, 1)[None]).float().to(device))
    permutations = model.last_scan_debug(16, 16)
    normal_rank = permutations.normal_inverse[0].reshape(16, 16).cpu()
    tangent_rank = permutations.tangent_inverse[0].reshape(16, 16).cpu()
    potential = permutations.potential[0].cpu()
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(potential, cmap="magma", vmin=0, vmax=1)
    axes[0].set_title("Predicted guide (16×16)")
    axes[1].imshow(normal_rank, cmap="turbo")
    axes[1].set_title("Normal scan rank")
    axes[2].imshow(tangent_rank, cmap="turbo")
    axes[2].set_title("Window-local tangential scan rank")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    path = config.output_dir / "geometric_scan_order_16x16.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def save_synthetic_scan_debug(config):
    """Visualize ordering on a clean irregular field independent of training."""
    y, x = np.mgrid[:16, :16]
    first = np.exp(-(((x - 4.5) / 3.2) ** 2 + ((y - 6.0) / 5.0) ** 2))
    second = np.exp(-(((x - 11.5) / 2.1) ** 2 + ((y - 10.5) / 3.0) ** 2))
    notch = np.where((x > 5) & (y < 7), 0.35, 1.0)
    field = np.clip(np.maximum(first * notch, second), 0, 1).astype(np.float32)
    permutations = build_pde_permutations(
        torch.from_numpy(field)[None, None], 16, 16,
        num_pde_bins=config.num_pde_bins,
        window_size=config.pde_scan_window,
    )
    normal_rank = permutations.normal_inverse[0].reshape(16, 16)
    tangent_rank = permutations.tangent_inverse[0].reshape(16, 16)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, panel, title, cmap in (
        (axes[0], field, "Irregular synthetic PDE field", "magma"),
        (axes[1], normal_rank, "Normal scan rank", "turbo"),
        (
            axes[2], tangent_rank,
            "Window-local tangential scan rank", "turbo",
        ),
    ):
        axis.imshow(panel, cmap=cmap)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    path = config.output_dir / "geometric_scan_order_synthetic_16x16.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def save_scan_diagnostics(model, config):
    path = config.output_dir / "geometric_scan_diagnostics.json"
    if not model.guidance_enabled or model._last_scan_cache is None:
        path.write_text(json.dumps({
            "scan_mode": "cartesian",
            "pde_ordering_constructed": False,
        }, indent=2))
        return path
    base = model._last_scan_cache.get(16, 16)
    guidance = model._last_scan_cache.guidance
    generator = torch.Generator(device=guidance.device).manual_seed(config.seed)
    perturbed = (guidance + 1.0e-3 * torch.randn(
        guidance.shape, generator=generator,
        device=guidance.device, dtype=guidance.dtype,
    )).clamp(0, 1)
    alternate = build_pde_permutations(
        perturbed, 16, 16,
        num_pde_bins=config.num_pde_bins,
        window_size=config.pde_scan_window,
    )

    def statistics(permutation, inverse, alternate_inverse):
        sequence = permutation[0].float()
        y = torch.div(sequence, 16, rounding_mode="floor")
        x = sequence.remainder(16)
        steps = torch.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
        rank_change = (
            inverse[0].float() - alternate_inverse[0].float()
        ).abs()
        return {
            "mean_consecutive_spatial_jump_tokens": float(steps.mean()),
            "max_consecutive_spatial_jump_tokens": float(steps.max()),
            "mean_absolute_rank_change_under_1e-3_noise": float(
                rank_change.mean()
            ),
            "fraction_tokens_same_rank_under_1e-3_noise": float(
                (rank_change == 0).float().mean()
            ),
        }

    report = {
        "resolution": [16, 16],
        "normal": statistics(
            base.normal, base.normal_inverse, alternate.normal_inverse
        ),
        "tangential": statistics(
            base.tangent, base.tangent_inverse,
            alternate.tangent_inverse,
        ),
        "scientific_warnings": [
            "Normal potential sorting groups equal-potential tokens across "
            "different nuclei and therefore is not a local streamline.",
            "Window-local tangential ordering can still jump between "
            "disconnected nuclei inside one window.",
            "Discrete ranks can change under small guidance perturbations; "
            "the hybrid zero gate protects pretrained behavior early.",
            "The permutation is non-differentiable and receives detached "
            "model-predicted guidance, never ground truth guidance.",
        ],
    }
    path.write_text(json.dumps(report, indent=2))
    return path


def _archive_results(output_dir):
    path = output_dir.parent / f"{output_dir.name}_results.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(output_dir.rglob("*")):
            if item.is_file() and item.suffix != ".pth":
                archive.write(item, item.relative_to(output_dir))
    return path


@torch.inference_mode()
def _median_forward_ms(model, device, patch_size, repeats):
    sample = torch.zeros(1, 3, patch_size, patch_size, device=device)
    was_training = model.training
    model.eval()
    model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000)
    model.train(was_training)
    return float(np.median(timings))


def _overhead_report(model, config, device):
    guided = model.parameter_report()["guided_ss2d_blocks"]
    repeats = 2 if config.smoke_test else 5
    selected_ms = _median_forward_ms(
        model, device, config.patch_size, repeats
    )
    cartesian_config = replace(config, scan_mode="cartesian")
    cartesian = PDEGeometricMambaUNet(cartesian_config).to(device)
    cartesian_ms = _median_forward_ms(
        cartesian, device, config.patch_size, repeats
    )
    del cartesian
    if device.type == "cuda":
        torch.cuda.empty_cache()
    guidance_enabled = config.scan_mode != "cartesian"
    report = {
        "scan_mode": config.scan_mode,
        "guided_stages": list(config.pde_scan_stages),
        "guided_ss2d_blocks": guided,
        "permutation_cache_scope": (
            "one forward pass" if guidance_enabled else "not applicable"
        ),
        "sorted_resolutions": ["16x16", "8x8"] if guidance_enabled else [],
        "sort_complexity": (
            "O(B*L*log(L)) once per distinct guided resolution"
            if guidance_enabled else "not applicable"
        ),
        "measured_forward_ms_batch1": selected_ms,
        "measured_cartesian_forward_ms_batch1": cartesian_ms,
        "measured_full_model_overhead_ratio": selected_ms / cartesian_ms,
        "benchmark_repeats": repeats,
        "selective_scan_cost": (
            "approximately 2x SS2D scan work only in guided blocks; "
            "unguided blocks unchanged"
            if config.scan_mode == "hybrid" else
            "same scan count as Cartesian in guided blocks plus sort/gather"
            if guidance_enabled else
            "unchanged Cartesian SS2D"
        ),
        "warning": (
            "Window-local level-set ordering is approximate and may jump "
            "between disconnected nuclei inside a window."
            if guidance_enabled else None
        ),
    }
    (config.output_dir / "computational_overhead.json").write_text(
        json.dumps(report, indent=2)
    )
    return report


def run_experiment(config: PDEGeometricConfig) -> Path:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed, config.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not config.smoke_test:
        raise RuntimeError("Full geometric Mamba training requires CUDA")

    records = discover_records(config.data_root, config.split_csv)
    encoding = detect_type_encoding(records, config.type_encoding)
    split = build_split(records, config, config.output_dir)
    train_base, train_loader, val_loader = create_pde_loaders(
        split, config, encoding
    )
    model = PDEGeometricMambaUNet(config)
    checkpoint = ensure_pretrained_checkpoint(config)
    load_report = load_official_pretraining(
        model, checkpoint, config.output_dir, config.checkpoint_dataset
    )
    model.to(device)
    shapes = verify_geometric_output_contract(
        model, device, config.patch_size
    )
    report = model.parameter_report()
    report.update({
        "pretrained_lr": config.pretrained_learning_rate,
        "new_lr": config.learning_rate,
        "dummy_output_shapes": shapes,
    })
    (config.output_dir / "parameter_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print("Parameter/output report:", report)
    print(
        "Pretrained tensors loaded:", len(load_report["loaded"]),
        "| missing target tensors:", len(load_report["missing_target"]),
    )
    overhead = _overhead_report(model, config, device)
    print("Measured computational overhead:", overhead)
    # Keep stochastic training repeatable after constructing the benchmark-only
    # Cartesian model.
    seed_everything(config.seed, config.deterministic)

    binary_weights, type_weights, type_counts = _loss_weights(
        train_base, device, len(config.type_classes)
    )
    print("Training type pixels:", type_counts.tolist())
    criterion = PDEGeometricLoss(binary_weights, type_weights, config)
    pretrained, new = model.parameter_groups()
    optimizer = torch.optim.Adam(
        [
            {"params": pretrained, "lr": config.pretrained_learning_rate},
            {"params": new, "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=75, gamma=0.1
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.amp and device.type == "cuda"
    )
    start_epoch, best_val_loss, bad_validations = 0, float("inf"), 0
    history: list[dict] = []
    if config.resume_checkpoint is not None:
        state = torch.load(
            config.resume_checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_val_loss = float(state.get("best_val_loss", float("inf")))

    epochs = 1 if config.smoke_test else config.epochs
    eval_every = 1 if config.smoke_test else config.eval_every
    for epoch in range(start_epoch, epochs):
        model.train()
        totals = defaultdict(float)
        progress = tqdm(
            train_loader, desc=f"Geometric epoch {epoch+1}/{epochs}"
        )
        for step, (images, instances, types, field, _) in enumerate(progress):
            images = images.to(device, non_blocking=True)
            instances = instances.to(device, non_blocking=True)
            types = types.to(device, non_blocking=True)
            field = field.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=config.amp and device.type == "cuda",
            ):
                loss, parts = criterion(
                    model(images), instances, types, field
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch+1}, step {step+1}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            for key, value in parts.items():
                totals[key] += value
            progress.set_postfix(
                loss=f"{totals['total']/(step+1):.4f}"
            )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            **{
                f"train_{key}": value / max(len(train_loader), 1)
                for key, value in totals.items()
            },
            "pretrained_lr": optimizer.param_groups[0]["lr"],
            "new_lr": optimizer.param_groups[1]["lr"],
        }
        do_evaluate = (
            (epoch + 1) % eval_every == 0
            or epoch + 1 == epochs
            or epoch == start_epoch
        )
        if do_evaluate:
            losses = validation_loss(
                model, val_loader, criterion, device, config.amp
            )
            row.update({f"val_{key}": value for key, value in losses.items()})
            val_summary = evaluate(
                model, split.val, "val", config, encoding, device, False
            )
            row.update({
                f"val_metric_{key}": value
                for key, value in val_summary.items()
            })
            val_loss = losses["loss"]
            if val_loss < best_val_loss:
                best_val_loss, bad_validations = val_loss, 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "config": config.to_dict(),
                    },
                    config.output_dir / "geometric_best_checkpoint.pth",
                )
                print(f"New best validation loss: {best_val_loss:.6f}")
            else:
                bad_validations += 1
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": config.to_dict(),
            },
            config.output_dir / "geometric_latest_checkpoint.pth",
        )
        history.append(row)
        pd.DataFrame(history).to_csv(
            config.output_dir / "geometric_training_history.csv", index=False
        )
        if (
            do_evaluate
            and config.early_stopping_patience is not None
            and bad_validations >= config.early_stopping_patience
        ):
            print(
                f"Early stopping after {bad_validations} validation events"
            )
            break

    best_path = config.output_dir / "geometric_best_checkpoint.pth"
    if not best_path.exists():
        raise RuntimeError("No best checkpoint was written")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    validation_summary = evaluate(
        model, split.val, "val", config, encoding, device, True
    )
    test_summary = evaluate(
        model, split.test, "test", config, encoding, device, True
    )
    visualization = save_validation_visualizations(
        split.val, config, encoding
    )
    task_visualization = save_task_visualizations(
        split.val, config, encoding
    )
    scan_debug = save_scan_debug(model, split.val[0], config, device)
    synthetic_scan_debug = save_synthetic_scan_debug(config)
    scan_diagnostics = save_scan_diagnostics(model, config)
    metadata = {
        **config.to_dict(),
        "architecture": "native Mamba-UNet with NP/PDE/type heads",
        "guidance_source": "model_prediction_detached_for_discrete_sort",
        "ground_truth_guidance_at_inference": False,
        "checkpoint_selection": "minimum_validation_loss",
        "best_epoch": int(best["epoch"]) + 1,
        "best_validation_loss": float(best["best_val_loss"]),
        "validation_summary": validation_summary,
        "test_summary": test_summary,
    }
    (config.output_dir / "geometric_run_config.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False)
    )
    archive = _archive_results(config.output_dir)
    print("\nFINAL TEST SUMMARY\n", json.dumps(test_summary, indent=2))
    print("Saved validation visualization:", visualization)
    print("Saved task visualization:", task_visualization)
    print("Saved scan-order debug visualization:", scan_debug)
    print("Saved synthetic scan-order visualization:", synthetic_scan_debug)
    print("Saved scan diagnostics:", scan_diagnostics)
    print("Saved results archive:", archive)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return config.output_dir


def build_parser():
    parser = argparse.ArgumentParser(
        description="PDE-guided geometric Mamba GLySAC experiment"
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/pde_geometric_mamba_glysac.json"),
    )
    parser.add_argument(
        "--scan-mode",
        choices=("cartesian", "pde", "hybrid", "normal", "tangential"),
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    config = PDEGeometricConfig.from_json(args.config)
    if args.scan_mode is not None:
        config.scan_mode = args.scan_mode
        if args.output_dir is None:
            name = config.output_dir.name
            for mode in ("hybrid", "pde", "cartesian", "normal", "tangential"):
                suffix = f"_{mode}"
                if name.endswith(suffix):
                    name = name[: -len(suffix)] + f"_{args.scan_mode}"
                    break
            else:
                name = f"{name}_{args.scan_mode}"
            config.output_dir = config.output_dir.with_name(name)
    for name in ("data_root", "output_dir", "resume_checkpoint"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    if args.smoke_test:
        config.smoke_test = True
    output = run_experiment(config)
    print("Geometric experiment complete:", output)


if __name__ == "__main__":
    main()
