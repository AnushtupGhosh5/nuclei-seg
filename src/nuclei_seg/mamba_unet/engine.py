from __future__ import annotations

import gc
import hashlib
import json
import random
import shutil
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import MambaUNetConfig
from .data import (
    DataSplit,
    RandomPatchDataset,
    Record,
    build_split,
    create_loaders,
    detect_type_encoding,
    discover_records,
    read_annotation,
    read_rgb,
)
from .loss import SmileStyleLoss, class_weights
from .metrics import MetricAccumulator
from .model import MambaUNetNP_HV_Type, load_official_pretraining, verify_output_contract
from .postprocess import instance_type_map, smile_watershed


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_pretrained_checkpoint(config: MambaUNetConfig) -> Path:
    path = config.pretrained_checkpoint
    if path.exists() and _sha256(path) == config.checkpoint_sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    print("Downloading verified VMamba-T checkpoint:", config.checkpoint_url)
    urllib.request.urlretrieve(config.checkpoint_url, temporary)
    actual = _sha256(temporary)
    if actual != config.checkpoint_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {actual}")
    temporary.replace(path)
    return path


def _loss_weights(
    dataset: RandomPatchDataset,
    device: torch.device,
    num_type_classes: int = 4,
):
    type_counts = np.zeros(num_type_classes, dtype=np.int64)
    binary_counts = np.zeros(2, dtype=np.int64)
    for _, instances, types, _ in dataset.tiles:
        type_counts += np.bincount(
            types.ravel(), minlength=num_type_classes
        )[:num_type_classes]
        binary_counts += np.bincount((instances > 0).ravel(), minlength=2)[:2]
    return class_weights(binary_counts, device), class_weights(type_counts, device), type_counts


@torch.inference_mode()
def validation_loss(
    model: MambaUNetNP_HV_Type,
    loader: DataLoader,
    criterion: SmileStyleLoss,
    device: torch.device,
    amp: bool,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total = 0.0
    parts_total: dict[str, float] = defaultdict(float)
    for images, instances, types, hv_map, _ in tqdm(loader, desc="Validation loss", leave=False):
        images = images.to(device, non_blocking=True)
        instances, types, hv_map = instances.to(device), types.to(device), hv_map.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            output = model(images)
            loss, parts = criterion(output, instances, types, hv_map)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        total += loss.item()
        for key, value in parts.items():
            parts_total[key] += value
    count = len(loader)
    return total / count, {key: value / count for key, value in parts_total.items()}


def _sliding_positions(length: int, size: int, overlap: int) -> list[int]:
    step = size - overlap
    positions = list(range(0, max(1, length - size + 1), step))
    last = max(0, length - size)
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


@torch.inference_mode()
def predict_tile(
    model: MambaUNetNP_HV_Type,
    image: np.ndarray,
    config: MambaUNetConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    size, overlap = config.patch_size, config.infer_overlap
    original_height, original_width = image.shape[:2]
    pad_h, pad_w = max(0, size - original_height), max(0, size - original_width)
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect") if pad_h or pad_w else image
    height, width = padded.shape[:2]
    y_positions = _sliding_positions(height, size, overlap)
    x_positions = _sliding_positions(width, size, overlap)
    one_dimensional = np.hanning(size).astype(np.float32)
    weight = np.maximum(np.outer(one_dimensional, one_dimensional), 0.05)
    binary_accumulator = np.zeros((2, height, width), np.float32)
    type_accumulator = np.zeros((4, height, width), np.float32)
    hv_accumulator = np.zeros((2, height, width), np.float32)
    normalizer = np.zeros((height, width), np.float32)
    coordinates = [(y, x) for y in y_positions for x in x_positions]
    for start in range(0, len(coordinates), config.infer_batch_size):
        batch_coordinates = coordinates[start : start + config.infer_batch_size]
        batch = np.stack(
            [padded[y : y + size, x : x + size].transpose(2, 0, 1) for y, x in batch_coordinates]
        )
        batch_tensor = torch.from_numpy(batch).float().to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=config.amp and device.type == "cuda"
        ):
            output = model(batch_tensor)
        binary = output["np"].softmax(1).float().cpu().numpy()
        types = output["tp"].softmax(1).float().cpu().numpy()
        hv_map = output["hv"].float().cpu().numpy()
        for index, (y, x) in enumerate(batch_coordinates):
            binary_accumulator[:, y : y + size, x : x + size] += binary[index] * weight
            type_accumulator[:, y : y + size, x : x + size] += types[index] * weight
            hv_accumulator[:, y : y + size, x : x + size] += hv_map[index] * weight
            normalizer[y : y + size, x : x + size] += weight
    nucleus_probability = (binary_accumulator / normalizer)[1]
    pixel_types = (type_accumulator / normalizer).argmax(0).astype(np.uint8)
    hv_map = (hv_accumulator / normalizer).transpose(1, 2, 0)
    instances = smile_watershed(nucleus_probability, hv_map)
    return (
        instances[:original_height, :original_width],
        pixel_types[:original_height, :original_width],
        hv_map[:original_height, :original_width],
        nucleus_probability[:original_height, :original_width],
    )


def evaluate(
    model: MambaUNetNP_HV_Type,
    rows: list[Record],
    split_name: str,
    config: MambaUNetConfig,
    encoding: str,
    device: torch.device,
    save_predictions: bool,
):
    accumulator = MetricAccumulator(config.match_iou, config.magnification)
    prediction_dir = config.output_dir / f"predictions_{split_name}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for record in tqdm(rows, desc=f"Evaluating {split_name}"):
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
        predicted_instance, pixel_types, predicted_hv, nucleus_probability = predict_tile(
            model, image, config, device
        )
        predicted_type = instance_type_map(predicted_instance, pixel_types)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        accumulator.update(
            record.stem,
            true_instance,
            true_type,
            predicted_instance,
            predicted_type,
            elapsed,
        )
        if save_predictions:
            np.savez_compressed(
                prediction_dir / f"{record.stem}.npz",
                inst_map=predicted_instance,
                type_map=predicted_type,
                type_pixel_map=pixel_types,
                hv_map=predicted_hv,
                nucleus_probability=nucleus_probability,
            )
    summary, per_image, pixel, instance, panoptic = accumulator.finalize()
    per_image.to_csv(config.output_dir / f"per_image_{split_name}.csv", index=False)
    pixel.to_csv(config.output_dir / f"classwise_pixel_{split_name}.csv", index=False)
    instance.to_csv(config.output_dir / f"classwise_instance_{split_name}.csv", index=False)
    panoptic.to_csv(config.output_dir / f"classwise_panoptic_{split_name}.csv", index=False)
    (config.output_dir / f"metrics_{split_name}.json").write_text(json.dumps(summary, indent=2))
    return summary, per_image, pixel, instance, panoptic


def _save_visualizations(
    records: list[Record], config: MambaUNetConfig, number: int = 5
) -> Path:
    type_cmap = ListedColormap(["#000000", "#F4A261", "#2A9D8F", "#E63946"])
    type_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], type_cmap.N)
    hv_cmap = plt.colormaps["coolwarm"].copy()
    hv_cmap.set_bad("black")

    selected = records[: min(number, len(records))]
    figure, axes = plt.subplots(len(selected), 5, figsize=(22, 4.2 * len(selected)), squeeze=False)
    titles = ["Original RGB", "Predicted type", "Predicted instances", "Horizontal HV", "Vertical HV"]
    for row, record in enumerate(selected):
        image = read_rgb(record.image)
        with np.load(config.output_dir / "predictions_test" / f"{record.stem}.npz") as prediction:
            instances, types, hv_map = prediction["inst_map"], prediction["type_map"], prediction["hv_map"]
        colors = np.zeros((int(instances.max()) + 1, 3), dtype=np.uint8)
        if instances.max():
            colors[1:] = np.random.default_rng(config.seed).integers(
                45, 256, size=(int(instances.max()), 3), dtype=np.uint8
            )
        panels = [
            image,
            types,
            colors[instances],
            np.ma.masked_where(instances == 0, np.clip(hv_map[..., 0], -1, 1)),
            np.ma.masked_where(instances == 0, np.clip(hv_map[..., 1], -1, 1)),
        ]
        axes[row, 0].imshow(panels[0])
        axes[row, 1].imshow(panels[1], cmap=type_cmap, norm=type_norm, interpolation="nearest")
        axes[row, 2].imshow(panels[2], interpolation="nearest")
        axes[row, 3].imshow(panels[3], cmap=hv_cmap, vmin=-1, vmax=1)
        axes[row, 4].imshow(panels[4], cmap=hv_cmap, vmin=-1, vmax=1)
        axes[row, 0].set_ylabel(record.stem, fontsize=9)
        for column in range(5):
            if row == 0:
                axes[row, column].set_title(titles[column])
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.suptitle(
        "Type colors: black=background | orange=other | teal=lymphocyte | red=epithelial",
        y=1.002,
    )
    figure.tight_layout()
    path = config.output_dir / "test_visualizations_5.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def run_experiment(config: MambaUNetConfig) -> Path:
    config.validate()
    if not config.data_root.exists():
        raise FileNotFoundError(config.data_root)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed, config.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not config.smoke_test:
        raise RuntimeError("Full Mamba-UNet training requires CUDA; use smoke_test only for CPU checks")

    records = discover_records(config.data_root, config.split_csv)
    encoding = detect_type_encoding(records, config.type_encoding)
    split = build_split(records, config, config.output_dir)
    print("Effective split:", len(split.train), len(split.val), len(split.test))
    print("Split identity SHA-256:", split.identity_sha256)
    train_dataset, train_loader, val_loader = create_loaders(split, config, encoding)

    checkpoint = ensure_pretrained_checkpoint(config)
    model = MambaUNetNP_HV_Type()
    load_official_pretraining(model, checkpoint, config.output_dir, config.checkpoint_dataset)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("All Mamba-UNet parameters must remain trainable")
    model.to(device)
    verify_output_contract(model, device, config.patch_size)
    parameter_report = model.parameter_report()
    parameter_report.update(
        {
            "pretrained_core_lr": config.pretrained_learning_rate,
            "new_task_heads_lr": config.learning_rate,
        }
    )
    (config.output_dir / "parameter_counts.json").write_text(json.dumps(parameter_report, indent=2))
    print("Parameter report:", parameter_report)

    binary_weights, type_weights, type_counts = _loss_weights(train_dataset, device)
    print("Training type pixels:", type_counts.tolist())
    print("Binary weights:", binary_weights.detach().cpu().numpy())
    print("Type weights:", type_weights.detach().cpu().numpy())
    criterion = SmileStyleLoss(binary_weights, type_weights)
    core_parameters, head_parameters = model.parameter_groups()
    optimizer = torch.optim.Adam(
        [
            {"params": core_parameters, "lr": config.pretrained_learning_rate, "name": "official_core"},
            {"params": head_parameters, "lr": config.learning_rate, "name": "task_heads"},
        ],
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)
    start_epoch, best_val_loss = 0, float("inf")
    if config.resume_checkpoint is not None:
        state = torch.load(config.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_val_loss = state.get("best_val_loss", float("inf"))

    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    history: list[dict] = []
    bad_validations = 0
    epochs = 1 if config.smoke_test else config.epochs
    eval_every = 1 if config.smoke_test else config.eval_every
    for epoch in range(start_epoch, epochs):
        model.train()
        running: dict[str, float] = defaultdict(float)
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for step, (images, instances, types, hv_map, _) in enumerate(progress):
            images = images.to(device, non_blocking=True)
            instances, types, hv_map = instances.to(device), types.to(device), hv_map.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=config.amp and device.type == "cuda"
            ):
                output = model(images)
                loss, parts = criterion(output, instances, types, hv_map)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}, batch {step + 1}: {parts}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite gradient at epoch {epoch + 1}, batch {step + 1}")
            scaler.step(optimizer)
            scaler.update()
            running["loss"] += loss.item()
            for key, value in parts.items():
                running[key] += value
            progress.set_postfix(loss=f"{running['loss'] / (step + 1):.3f}")
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            **{key: value / len(train_loader) for key, value in running.items()},
            "pretrained_core_lr": optimizer.param_groups[0]["lr"],
            "task_head_lr": optimizer.param_groups[1]["lr"],
        }
        do_evaluate = (epoch + 1) % eval_every == 0 or epoch + 1 == epochs or epoch == start_epoch
        if do_evaluate:
            val_loss, val_parts = validation_loss(model, val_loader, criterion, device, config.amp)
            val_summary, *_ = evaluate(model, split.val, "val", config, encoding, device, False)
            row.update(
                {
                    "val_loss": val_loss,
                    "val_pq": val_summary["pq_image_mean"],
                    **{f"val_{key}_loss": value for key, value in val_parts.items()},
                }
            )
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
                    config.output_dir / "best_checkpoint.pth",
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
            config.output_dir / "latest_checkpoint.pth",
        )
        history.append(row)
        pd.DataFrame(history).to_csv(config.output_dir / "training_history.csv", index=False)
        patience = config.early_stopping_patience
        if do_evaluate and patience is not None and bad_validations >= patience:
            print(f"Early stopping after {bad_validations} validation events")
            break

    best_path = config.output_dir / "best_checkpoint.pth"
    if not best_path.exists():
        raise RuntimeError("No best checkpoint was written")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    summary, *_ = evaluate(model, split.test, "test", config, encoding, device, True)
    run_metadata = {
        **config.to_dict(),
        "device": str(device),
        "split_identity_sha256": split.identity_sha256,
        "upstream_commit": config.upstream_commit,
        "type_encoding_resolved": encoding,
        "test_summary": summary,
    }
    (config.output_dir / "run_config.json").write_text(json.dumps(run_metadata, indent=2))
    visualization = _save_visualizations(split.test, config)
    print("Saved visualization:", visualization)
    archive = shutil.make_archive(str(config.output_dir), "zip", root_dir=config.output_dir)
    print("Saved experiment archive:", archive)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return config.output_dir
