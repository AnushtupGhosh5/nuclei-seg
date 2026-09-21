from __future__ import annotations

import argparse
import json
import random
import time
import zipfile
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import MambaUNetConfig
from .data import (
    RandomPatchDataset,
    build_split,
    detect_type_encoding,
    discover_records,
    read_annotation,
    read_rgb,
)
from .engine import (
    _loss_weights,
    _sliding_positions,
    ensure_pretrained_checkpoint,
    seed_everything,
)
from .metrics import MetricAccumulator
from .model import load_official_pretraining
from .pde_field import make_poisson_field, poisson_watershed
from .pde_model import MambaUNetPoissonType
from .postprocess import instance_type_map


class PoissonTargetDataset(Dataset):
    """Reuse the validated GLySAC crop pipeline, replacing HV with a PDE field."""

    def __init__(self, base: RandomPatchDataset, iterations: int = 48) -> None:
        self.base = base
        self.iterations = iterations

    @property
    def rng(self):
        return self.base.rng

    @rng.setter
    def rng(self, value):
        self.base.rng = value

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, instances, types, _, stem = self.base[index]
        field = make_poisson_field(instances.numpy(), self.iterations)
        return image, instances, types, torch.from_numpy(field), stem


def _seed_worker(worker_id: int, seed: int) -> None:
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker = torch.utils.data.get_worker_info()
    if worker is not None:
        worker.dataset.rng = np.random.default_rng(worker_seed)
def make_loaders(split, config, encoding):
    patches = 2 if config.smoke_test else config.patches_per_tile
    train_base = RandomPatchDataset(
        split.train, config.patch_size, patches, config.seed, encoding,
        config.min_foreground_fraction, augment=True,
    )
    val_base = RandomPatchDataset(
        split.val, config.patch_size, patches, config.seed, encoding,
        config.min_foreground_fraction, augment=False,
    )
    train_set = PoissonTargetDataset(train_base)
    val_set = PoissonTargetDataset(val_base)
    common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        worker_init_fn=partial(_seed_worker, seed=config.seed),
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=True, generator=generator, **common
    )
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_base, train_loader, val_loader


def dice_loss(logits: torch.Tensor, target: torch.Tensor, classes: int) -> torch.Tensor:
    probability = logits.softmax(1)
    one_hot = F.one_hot(target, classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    score = (2 * (probability * one_hot).sum(dims) + 1.0) / (
        (probability + one_hot).sum(dims) + 1.0
    )
    return 1.0 - score[1:].mean()
class PoissonLoss:
    def __init__(self, type_weights: torch.Tensor) -> None:
        self.type_weights = type_weights

    def __call__(self, output, instances, types, field_target):
        predicted = output["field"].float().sigmoid().squeeze(1)
        residual = F.smooth_l1_loss(
            predicted, field_target.float(), reduction="none", beta=0.1
        )
        foreground = (instances > 0).float()
        weights = 1.0 + foreground
        field_loss = (residual * weights).sum() / weights.sum().clamp_min(1.0)

        type_logits = output["tp"].float()
        type_loss = F.cross_entropy(
            type_logits, types, weight=self.type_weights
        ) + dice_loss(type_logits, types, 4)
        total = field_loss + type_loss
        return total, {
            "field": float(field_loss.detach()),
            "type": float(type_loss.detach()),
        }


@torch.inference_mode()
def validation_loss(model, loader, criterion, device, amp):
    model.eval()
    total = 0.0
    parts = defaultdict(float)
    for images, instances, types, field, _ in tqdm(
        loader, desc="PDE validation", leave=False
    ):
        images = images.to(device, non_blocking=True)
        instances = instances.to(device)
        types = types.to(device)
        field = field.to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            output = model(images)
            loss, values = criterion(output, instances, types, field)
        total += float(loss)
        for key, value in values.items():
            parts[key] += value
    n = max(len(loader), 1)
    return total / n, {key: value / n for key, value in parts.items()}
@torch.inference_mode()
def predict_tile(model, image, config, device):
    model.eval()
    size, overlap = config.patch_size, config.infer_overlap
    original_h, original_w = image.shape[:2]
    pad_h, pad_w = max(0, size-original_h), max(0, size-original_w)
    padded = np.pad(
        image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
    ) if pad_h or pad_w else image
    height, width = padded.shape[:2]
    ys = _sliding_positions(height, size, overlap)
    xs = _sliding_positions(width, size, overlap)
    window = np.maximum(
        np.outer(np.hanning(size), np.hanning(size)).astype(np.float32), 0.05
    )
    field_acc = np.zeros((height, width), np.float32)
    type_acc = np.zeros((4, height, width), np.float32)
    norm = np.zeros((height, width), np.float32)
    coords = [(y, x) for y in ys for x in xs]
    for start in range(0, len(coords), config.infer_batch_size):
        batch_coords = coords[start:start+config.infer_batch_size]
        batch = np.stack([
            padded[y:y+size, x:x+size].transpose(2, 0, 1)
            for y, x in batch_coords
        ])
        tensor = torch.from_numpy(batch).float().to(device)
        output = model(tensor)
        fields = output["field"].sigmoid().squeeze(1).cpu().numpy()
        types = output["tp"].softmax(1).cpu().numpy()
        for i, (y, x) in enumerate(batch_coords):
            field_acc[y:y+size, x:x+size] += fields[i] * window
            type_acc[:, y:y+size, x:x+size] += types[i] * window
            norm[y:y+size, x:x+size] += window
    field = field_acc / np.maximum(norm, 1e-6)
    pixel_types = (type_acc / np.maximum(norm, 1e-6)).argmax(0).astype(np.uint8)
    instances = poisson_watershed(field)
    return (
        instances[:original_h, :original_w],
        pixel_types[:original_h, :original_w],
        field[:original_h, :original_w],
    )


@torch.inference_mode()
def evaluate(model, rows, split_name, config, encoding, device, save_predictions):
    accumulator = MetricAccumulator(config.match_iou, config.magnification)
    prediction_dir = config.output_dir / f"pde_predictions_{split_name}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for record in tqdm(rows, desc=f"PDE evaluate {split_name}"):
        image = read_rgb(record.image)
        true_instance, true_type = read_annotation(record.label, encoding)
        if config.smoke_test:
            height, width = image.shape[:2]
            size = min(config.patch_size, height, width)
            top = (height - size) // 2
            left = (width - size) // 2
            bounds = np.s_[top : top + size, left : left + size]
            image = image[bounds]
            true_instance = true_instance[bounds]
            true_type = true_type[bounds]
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        predicted_instance, pixel_types, field = predict_tile(
            model, image, config, device
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
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
                pde_field=field,
                inference_seconds=np.asarray(elapsed, dtype=np.float64),
            )
    summary, per_image, pixel, instance, panoptic = accumulator.finalize()
    summary.update(_postprocess_metadata())
    _write_metric_artifacts(
        config.output_dir, split_name, summary,
        per_image, pixel, instance, panoptic,
    )
    return summary


def _write_metric_artifacts(
    output_dir, split_name, summary, per_image, pixel, instance, panoptic
):
    """Write the same complete metric tables as the NP/HV baseline."""
    per_image.to_csv(
        output_dir / f"pde_per_image_{split_name}.csv", index=False
    )
    pixel.to_csv(
        output_dir / f"pde_classwise_pixel_{split_name}.csv", index=False
    )
    instance.to_csv(
        output_dir / f"pde_classwise_instance_{split_name}.csv", index=False
    )
    panoptic.to_csv(
        output_dir / f"pde_classwise_panoptic_{split_name}.csv", index=False
    )
    (output_dir / f"pde_metrics_{split_name}.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )


def _postprocess_metadata():
    return {
        "postprocess_smoothing_sigma": 1.5,
        "postprocess_foreground_threshold": 0.12,
        "postprocess_marker_threshold": 0.22,
        "postprocess_min_distance": 9,
        "postprocess_min_size": 10,
        "postprocess_selection_split": "validation_only_seed_42",
    }


def _save_pde_visualizations(records, config, number=5):
    """Render five test cases without pretending the PDE model predicts HV."""
    type_cmap = ListedColormap(
        ["#000000", "#F4A261", "#2A9D8F", "#E63946"]
    )
    type_norm = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5, 3.5], type_cmap.N
    )
    derivative_cmap = plt.colormaps["coolwarm"].copy()
    derivative_cmap.set_bad("black")
    selected = records[: min(number, len(records))]
    figure, axes = plt.subplots(
        len(selected), 6,
        figsize=(25, 4.2 * len(selected)), squeeze=False,
    )
    titles = [
        "Original RGB", "Predicted type", "Predicted instances",
        "PDE scalar field", "PDE horizontal derivative",
        "PDE vertical derivative",
    ]
    for row, record in enumerate(selected):
        image = read_rgb(record.image)
        prediction_path = (
            config.output_dir / "pde_predictions_test" /
            f"{record.stem}.npz"
        )
        with np.load(prediction_path) as prediction:
            instances = prediction["inst_map"]
            types = prediction["type_map"]
            field = prediction["pde_field"]
        vertical, horizontal = np.gradient(field.astype(np.float32))
        colors = np.zeros(
            (int(instances.max()) + 1, 3), dtype=np.uint8
        )
        if instances.max():
            colors[1:] = np.random.default_rng(
                config.seed
            ).integers(
                45, 256, size=(int(instances.max()), 3), dtype=np.uint8
            )
        foreground = instances > 0
        panels = [
            image,
            types,
            colors[instances],
            field,
            np.ma.masked_where(~foreground, horizontal),
            np.ma.masked_where(~foreground, vertical),
        ]
        axes[row, 0].imshow(panels[0])
        axes[row, 1].imshow(
            panels[1], cmap=type_cmap, norm=type_norm,
            interpolation="nearest",
        )
        axes[row, 2].imshow(panels[2], interpolation="nearest")
        axes[row, 3].imshow(panels[3], cmap="magma", vmin=0, vmax=1)
        derivative_limit = max(
            float(np.max(np.abs(horizontal))),
            float(np.max(np.abs(vertical))), 1.0e-6,
        )
        axes[row, 4].imshow(
            panels[4], cmap=derivative_cmap,
            vmin=-derivative_limit, vmax=derivative_limit,
        )
        axes[row, 5].imshow(
            panels[5], cmap=derivative_cmap,
            vmin=-derivative_limit, vmax=derivative_limit,
        )
        axes[row, 0].set_ylabel(record.stem, fontsize=9)
        for column, title in enumerate(titles):
            if row == 0:
                axes[row, column].set_title(title)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.suptitle(
        "Type colors: black=background | orange=other | "
        "teal=lymphocyte | red=epithelial",
        y=1.002,
    )
    figure.tight_layout()
    path = config.output_dir / "pde_test_visualizations_5.png"
    figure.savefig(
        path, dpi=180, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)
    return path


def _archive_pde_results(output_dir):
    """Archive report artifacts and predictions without duplicating checkpoints."""
    archive = output_dir.parent / f"{output_dir.name}_results.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED
    ) as handle:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.suffix != ".pth":
                handle.write(path, path.relative_to(output_dir))
    return archive


def export_existing_predictions(config):
    """Create missing reports from a completed run without model inference."""
    records = discover_records(config.data_root, config.split_csv)
    encoding = detect_type_encoding(records, config.type_encoding)
    split = build_split(records, config, config.output_dir)
    accumulator = MetricAccumulator(config.match_iou, config.magnification)
    prediction_dir = config.output_dir / "pde_predictions_test"
    for record in tqdm(split.test, desc="Exporting saved PDE predictions"):
        path = prediction_dir / f"{record.stem}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing saved test prediction: {path}"
            )
        true_instance, true_type = read_annotation(record.label, encoding)
        with np.load(path) as prediction:
            field = prediction["pde_field"]
            pixel_types = prediction["type_pixel_map"]
            elapsed = (
                float(prediction["inference_seconds"])
                if "inference_seconds" in prediction.files else float("nan")
            )
        # Re-run only deterministic post-processing. This repairs predictions
        # made before Gaussian smoothing was added; no model inference occurs.
        predicted_instance = poisson_watershed(field)
        predicted_type = instance_type_map(
            predicted_instance, pixel_types
        )
        save_values = {
            "inst_map": predicted_instance,
            "type_map": predicted_type,
            "type_pixel_map": pixel_types,
            "pde_field": field,
        }
        if np.isfinite(elapsed):
            save_values["inference_seconds"] = np.asarray(
                elapsed, dtype=np.float64
            )
        np.savez_compressed(path, **save_values)
        accumulator.update(
            record.stem, true_instance, true_type,
            predicted_instance, predicted_type, elapsed,
        )
    summary, per_image, pixel, instance, panoptic = accumulator.finalize()
    if per_image["inference_seconds_including_postprocess"].isna().all():
        summary["inference_seconds_including_postprocess_image_mean"] = None
        summary["inference_seconds_total_including_postprocess"] = None
        summary["inference_images_per_second_including_postprocess"] = None
        summary["inference_timing_available"] = False
    summary.update(_postprocess_metadata())
    _write_metric_artifacts(
        config.output_dir, "test", summary,
        per_image, pixel, instance, panoptic,
    )
    run_config_path = config.output_dir / "pde_run_config.json"
    run_metadata = (
        json.loads(run_config_path.read_text())
        if run_config_path.exists() else config.to_dict()
    )
    run_metadata["test_summary"] = summary
    run_metadata["postprocess"] = _postprocess_metadata()
    run_config_path.write_text(
        json.dumps(run_metadata, indent=2, allow_nan=False)
    )
    visualization = _save_pde_visualizations(split.test, config)
    archive = _archive_pde_results(config.output_dir)
    print("\nPDE TEST SUMMARY")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print("Saved visualization:", visualization)
    print("Saved results archive:", archive)
    return config.output_dir


def run_experiment(config: MambaUNetConfig) -> Path:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed, config.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not config.smoke_test:
        raise RuntimeError("Full PDE-Mamba training requires CUDA")

    records = discover_records(config.data_root, config.split_csv)
    encoding = detect_type_encoding(records, config.type_encoding)
    split = build_split(records, config, config.output_dir)
    train_base, train_loader, val_loader = make_loaders(
        split, config, encoding
    )

    checkpoint = ensure_pretrained_checkpoint(config)
    model = MambaUNetPoissonType()
    load_official_pretraining(
        model, checkpoint, config.output_dir, config.checkpoint_dataset
    )
    model.to(device)
    _, type_weights, _ = _loss_weights(train_base, device)
    criterion = PoissonLoss(type_weights)
    core, heads = model.parameter_groups()
    optimizer = torch.optim.Adam(
        [
            {"params": core, "lr": config.pretrained_learning_rate},
            {"params": heads, "lr": config.learning_rate},
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
    start_epoch = 0
    best_val_loss = float("inf")
    bad_validations = 0
    history = []

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
        running = defaultdict(float)
        progress = tqdm(
            train_loader, desc=f"PDE epoch {epoch+1}/{epochs}"
        )
        for step, (images, instances, types, field, _) in enumerate(progress):
            images = images.to(device, non_blocking=True)
            instances = instances.to(device)
            types = types.to(device)
            field = field.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=config.amp and device.type == "cuda",
            ):
                output = model(images)
                loss, parts = criterion(
                    output, instances, types, field
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite PDE loss at epoch {epoch+1}, step {step+1}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running["loss"] += float(loss.detach())
            for key, value in parts.items():
                running[key] += value
            progress.set_postfix(
                loss=f"{running['loss']/(step+1):.4f}"
            )

        scheduler.step()
        row = {
            "epoch": epoch + 1,
            **{
                key: value / max(len(train_loader), 1)
                for key, value in running.items()
            },
            "core_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        do_evaluate = (
            (epoch + 1) % eval_every == 0
            or epoch + 1 == epochs
            or epoch == start_epoch
        )
        if do_evaluate:
            val_loss, val_parts = validation_loss(
                model, val_loader, criterion, device, config.amp
            )
            row["val_loss"] = val_loss
            row.update({
                f"val_{key}_loss": value
                for key, value in val_parts.items()
            })
            val_summary = evaluate(
                model, split.val, "val",
                config, encoding, device, False
            )
            row["val_pq"] = val_summary["pq_image_mean"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_validations = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "config": config.to_dict(),
                    },
                    config.output_dir / "pde_best_checkpoint.pth",
                )
                print(
                    f"New PDE best val loss: {best_val_loss:.6f}"
                )
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
            config.output_dir / "pde_latest_checkpoint.pth",
        )
        history.append(row)
        pd.DataFrame(history).to_csv(
            config.output_dir / "pde_training_history.csv",
            index=False,
        )

        patience = config.early_stopping_patience
        if (
            do_evaluate
            and patience is not None
            and bad_validations >= patience
        ):
            print(
                f"PDE early stopping after "
                f"{bad_validations} validation events"
            )
            break

    best_path = config.output_dir / "pde_best_checkpoint.pth"
    if not best_path.exists():
        raise RuntimeError("No PDE best checkpoint was written")
    state = torch.load(
        best_path, map_location=device, weights_only=False
    )
    model.load_state_dict(state["model"])
    test_summary = evaluate(
        model, split.test, "test",
        config, encoding, device, True
    )
    metadata = {
        **config.to_dict(),
        "representation": "per-instance Poisson scalar field",
        "checkpoint_selection": "minimum validation loss",
        "test_summary": test_summary,
    }
    (config.output_dir / "pde_run_config.json").write_text(
        json.dumps(metadata, indent=2)
    )
    visualization = _save_pde_visualizations(split.test, config)
    archive = _archive_pde_results(config.output_dir)
    print("\nPDE TEST SUMMARY")
    print(json.dumps(test_summary, indent=2, allow_nan=False))
    print("Saved visualization:", visualization)
    print("Saved results archive:", archive)
    return config.output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PDE scalar-field Mamba-UNet experiment"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mamba_unet_glysac.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--export-existing", action="store_true",
        help="Generate reports from saved test predictions without training",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = MambaUNetConfig.from_json(args.config)
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    else:
        config.output_dir = Path(
            "outputs/mamba_unet_pde_glysac"
        )
    if args.smoke_test:
        config.smoke_test = True
    output = (
        export_existing_predictions(config)
        if args.export_existing else run_experiment(config)
    )
    print("PDE experiment complete:", output)


if __name__ == "__main__":
    main()
