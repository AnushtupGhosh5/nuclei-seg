from __future__ import annotations

import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .datasets import DATASET_INFO, NucleiPatchDataset, load_manifest, split_records
from .losses import HoVerNetLoss
from .metrics import (
    evaluate_instances,
    hover_type_classification_counts,
    instance_types_from_map,
    type_classification_counts,
)
from .model import OriginalUNet
from .hovernet import HoVerNetFast
from .postprocess import post_process_instances
from .visualization import instance_overlay, save_prediction_visualization, save_training_curves


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Stable algorithms make fresh runs repeatable on the same software and
    # hardware stack. Exact equality across different GPU models is not
    # guaranteed by CUDA.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def profile_model(
    model: nn.Module, device: torch.device, input_size: int
) -> dict[str, int | float | list[int]]:
    """Profile parameters, Conv/Linear FLOPs, output geometry, and tile latency."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    flops = 0

    def count_flops(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal flops
        if isinstance(module, nn.Conv2d):
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (
                module.in_channels // module.groups
            )
            flops += output.numel() * kernel_ops * 2
            if module.bias is not None:
                flops += output.numel()
        elif isinstance(module, nn.ConvTranspose2d):
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (
                module.out_channels // module.groups
            )
            flops += inputs[0].numel() * kernel_ops * 2
            if module.bias is not None:
                flops += output.numel()
        elif isinstance(module, nn.Linear):
            flops += output.numel() * module.in_features * 2
            if module.bias is not None:
                flops += output.numel()

    hooks = [
        module.register_forward_hook(count_flops)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))
    ]
    was_training = model.training
    model.eval()
    dummy = torch.zeros(1, 3, input_size, input_size, device=device)
    output = model(dummy)
    output_shapes = {
        "np_output_shape": list(output.np_logits.shape),
        "hv_output_shape": list(output.hv_map.shape),
        "type_output_shape": list(output.type_logits.shape),
    }
    for hook in hooks:
        hook.remove()

    warmup_runs, timed_runs = ((3, 10) if device.type == "cuda" else (1, 3))
    for _ in range(warmup_runs):
        model(dummy)
    _synchronize(device)
    timings = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        model(dummy)
        _synchronize(device)
        timings.append(time.perf_counter() - start)
    model.train(was_training)
    mean_seconds = float(np.mean(timings))
    del output, dummy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "parameter_size_mb": float(parameter_bytes / 1024**2),
        # Multiply-accumulate is counted as two floating-point operations.
        "gflops": float(flops / 1.0e9),
        "profile_input_shape": [1, 3, input_size, input_size],
        **output_shapes,
        "tile_inference_mean_ms": mean_seconds * 1000.0,
        "tile_inference_std_ms": float(np.std(timings) * 1000.0),
        "tiles_per_second": float(1.0 / mean_seconds),
        "benchmark_warmup_runs": warmup_runs,
        "benchmark_timed_runs": timed_runs,
    }


def write_model_summary(path: Path, model: nn.Module, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        str(model),
        "",
        "Profile",
        "-------",
        *(f"{key}: {value}" for key, value in profile.items()),
        "",
        "GFLOPs count Conv2d/ConvTranspose2d/Linear operations and count one MAC as two FLOPs.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _device_batch(batch: dict, device: torch.device) -> tuple[Tensor, dict[str, Tensor]]:
    image = batch["image"].to(device, non_blocking=True)
    target = {
        key: batch[key].to(device, non_blocking=True)
        for key in ("np_map", "type_map", "hv_map")
    }
    return image, target


def _create_model(architecture: str, num_types: int, channels: tuple[int, ...]):
    if architecture == "hovernet_fast":
        return HoVerNetFast(num_types)
    if architecture == "unet":
        return OriginalUNet(num_types, channels)
    raise ValueError(f"Unknown architecture: {architecture}")


def _set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> None:
    if hasattr(model, "freeze"):
        model.freeze = not trainable
    names = ("conv0", "d0", "d1", "d2", "d3")
    for name in names:
        module = getattr(model, name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = trainable


def _make_optimizer(model: torch.nn.Module, args):
    if args.architecture == "hovernet_fast":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return optimizer, scheduler


def _center_crop_target(target: dict[str, Tensor], shape: tuple[int, int]) -> dict[str, Tensor]:
    current = target["np_map"].shape[-2:]
    if current == shape:
        return target
    top, left = (current[0] - shape[0]) // 2, (current[1] - shape[1]) // 2
    if top < 0 or left < 0:
        raise ValueError(f"Model output {shape} is larger than target {current}")
    return {
        key: value[..., top : top + shape[0], left : left + shape[1]]
        for key, value in target.items()
    }


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: HoVerNetLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    accumulation_steps: int = 1,
) -> dict[str, float]:
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be at least 1")
    training = optimizer is not None
    model.train(training)
    totals: defaultdict[str, float] = defaultdict(float)
    intersection = predicted = actual = 0
    type_correct = type_total = 0
    num_batches = len(loader)
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        image, target = _device_batch(batch, device)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = model(image)
            target = _center_crop_target(target, output.np_logits.shape[-2:])
            losses = criterion(output, target)
        if training:
            group_start = (batch_index // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, num_batches - group_start)
            scaler.scale(losses["loss"] / group_size).backward()
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == num_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        batch_size = image.shape[0]
        for name, value in losses.items():
            totals[name] += float(value.detach()) * batch_size
        np_pred = output.np_logits.argmax(1)
        np_true = target["np_map"]
        intersection += int(((np_pred == 1) & (np_true == 1)).sum())
        predicted += int((np_pred == 1).sum())
        actual += int((np_true == 1).sum())
        valid_type = target["type_map"] > 0
        type_correct += int(((output.type_logits.argmax(1) == target["type_map"]) & valid_type).sum())
        type_total += int(valid_type.sum())
    count = len(loader.dataset)
    metrics = {name: value / max(count, 1) for name, value in totals.items()}
    metrics["dice"] = (2 * intersection + 1.0e-6) / (predicted + actual + 1.0e-6)
    metrics["type_pixel_accuracy"] = type_correct / max(type_total, 1)
    return metrics


def train(args) -> Path:
    seed_everything(args.seed)
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    info = DATASET_INFO[args.dataset]
    records = load_manifest(data_dir, args.dataset)
    train_records, val_records, _ = split_records(records, args.seed, args.val_fraction)
    if not train_records or not val_records:
        raise RuntimeError("Training/validation split is empty")

    train_data = NucleiPatchDataset(
        train_records,
        args.patch_size,
        args.train_patches_per_image,
        training=True,
        foreground_probability=args.foreground_probability,
    )
    val_data = NucleiPatchDataset(
        val_records, args.patch_size, args.val_patches_per_image, training=False
    )
    def make_loaders(train_batch_size: int) -> tuple[DataLoader, DataLoader]:
        common = dict(
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.workers > 0,
        )
        return (
            DataLoader(
                train_data, batch_size=train_batch_size, shuffle=True, drop_last=True, **common
            ),
            DataLoader(val_data, batch_size=args.val_batch_size, shuffle=False, **common),
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = _create_model(args.architecture, info["num_types"], tuple(args.encoder_channels)).to(device)
    run_dir = output_dir / "models" / args.dataset / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_path = run_dir / "last.pt"
    resuming = args.resume and resume_path.exists()
    if args.pretrained_checkpoint and not resuming:
        pretrained = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=False)
        state = pretrained.get("desc", pretrained.get("model", pretrained))
        current = model.state_dict()
        compatible = {key: value for key, value in state.items() if key in current and current[key].shape == value.shape}
        incompatible = model.load_state_dict(compatible, strict=False)
        print(
            f"Loaded pretrained weights: tensors={len(compatible)} missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    model_profile = profile_model(model, device, args.patch_size)
    write_model_summary(run_dir / "model_summary.txt", model, model_profile)
    (run_dir / "model_profile.json").write_text(json.dumps(model_profile, indent=2))
    print(
        f"Model: parameters={model_profile['total_parameters']:,} "
        f"GFLOPs={model_profile['gflops']:.3f} "
        f"tile_latency={model_profile['tile_inference_mean_ms']:.2f}ms "
        f"tiles/s={model_profile['tiles_per_second']:.2f}",
        flush=True,
    )
    criterion = HoVerNetLoss()
    optimizer, scheduler = _make_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history_path = run_dir / "history.jsonl"
    config = vars(args).copy()
    config.update(
        device=str(device),
        num_types=info["num_types"],
        type_names=info["type_names"],
        train_images=len(train_records),
        val_images=len(val_records),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        model_profile=model_profile,
        unbounded_hv=True,
        architecture=args.architecture,
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    start_epoch = 0
    best_loss = float("inf")
    epochs_without_improvement = 0
    if resuming:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            phase_epoch = int(checkpoint["epoch"])
            if phase_epoch > args.freeze_encoder_epochs:
                phase_epoch -= args.freeze_encoder_epochs
            scheduler.last_epoch = phase_epoch
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        previous_freeze_epochs = int(
            checkpoint.get("config", {}).get("freeze_encoder_epochs", args.freeze_encoder_epochs)
        )
        was_encoder_trainable = start_epoch > previous_freeze_epochs
        next_encoder_trainable = start_epoch + 1 > args.freeze_encoder_epochs
        if args.architecture == "hovernet_fast" and next_encoder_trainable and not was_encoder_trainable:
            optimizer, scheduler = _make_optimizer(model, args)
            print(
                "Resume crosses the frozen/unfrozen boundary; reset Adam and StepLR for phase two",
                flush=True,
            )
        best_path = run_dir / "best.pt"
        if best_path.exists():
            best_loss = float(
                torch.load(best_path, map_location="cpu", weights_only=False)
                .get("val_metrics", {})
                .get("loss", float("inf"))
            )
        print(f"Resuming {resume_path} from completed epoch {start_epoch}", flush=True)

    initial_batch = (
        args.frozen_batch_size
        if start_epoch < args.freeze_encoder_epochs and args.architecture == "hovernet_fast"
        else args.batch_size
    )
    train_loader, val_loader = make_loaders(initial_batch)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        encoder_trainable = epoch > args.freeze_encoder_epochs
        accumulation_steps = (
            args.frozen_gradient_accumulation_steps
            if not encoder_trainable and args.architecture == "hovernet_fast"
            else args.gradient_accumulation_steps
        )
        _set_encoder_trainable(model, encoder_trainable)
        if (
            args.architecture == "hovernet_fast"
            and args.freeze_encoder_epochs > 0
            and epoch == args.freeze_encoder_epochs + 1
        ):
            # The reference recipe starts phase two with a fresh Adam state and
            # a fresh 1e-4 StepLR schedule.
            optimizer, scheduler = _make_optimizer(model, args)
            train_loader, val_loader = make_loaders(args.batch_size)
        epoch_start = time.perf_counter()
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            accumulation_steps=accumulation_steps,
        )
        train_seconds = time.perf_counter() - epoch_start
        validation_start = time.perf_counter()
        val_metrics = _run_epoch(model, val_loader, criterion, device, None, scaler)
        validation_seconds = time.perf_counter() - validation_start
        scheduler.step()
        row = {
            "epoch": epoch,
            "encoder_trainable": encoder_trainable,
            "gradient_accumulation_steps": accumulation_steps,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "epoch_seconds": train_seconds + validation_seconds,
            "train": train_metrics,
            "val": val_metrics,
        }
        with history_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        np_loss = train_metrics["np_ce"] + train_metrics["np_dice"]
        hv_loss = train_metrics["hv_mse"] + train_metrics["hv_gradient"]
        type_loss = train_metrics["type_ce"] + train_metrics["type_dice"]
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"(np={np_loss:.3f} hv={hv_loss:.3f} type={type_loss:.3f}) "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_type_acc={val_metrics['type_pixel_accuracy']:.4f} "
            f"accumulation={accumulation_steps} "
            f"train_time={train_seconds:.1f}s val_time={validation_seconds:.1f}s "
            f"epoch_time={train_seconds + validation_seconds:.1f}s",
            flush=True,
        )
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}: validation loss did not improve for "
                f"{epochs_without_improvement} epochs (best={best_loss:.4f})",
                flush=True,
            )
            break
    save_training_curves(history_path, run_dir / "training_curves.png")
    return run_dir / "best.pt"


def _tile_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    if positions[-1] != length - patch_size:
        positions.append(length - patch_size)
    return positions


@torch.inference_mode()
def tiled_predict(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    patch_size: int,
    overlap: int,
    batch_size: int,
    output_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if output_size is None:
        output_size = patch_size - overlap
    if not 0 < output_size <= patch_size:
        raise ValueError("output_size must be positive and no larger than patch_size")
    original_height, original_width = image.shape[:2]
    margin_before = (patch_size - output_size) // 2
    margin_after = patch_size - output_size - margin_before
    rows = int(np.ceil(original_height / output_size))
    columns = int(np.ceil(original_width / output_size))
    canvas_height, canvas_width = rows * output_size, columns * output_size
    image = np.pad(
        image,
        (
            (margin_before, margin_after + canvas_height - original_height),
            (margin_before, margin_after + canvas_width - original_width),
            (0, 0),
        ),
        mode="reflect",
    )
    height, width = canvas_height, canvas_width
    locations = [
        (y, x)
        for y in range(0, height, output_size)
        for x in range(0, width, output_size)
    ]
    num_types = getattr(model, "num_types", None)
    if num_types is None:
        num_types = model.type_decoder.head.out_channels
    np_sum = np.zeros((2, height, width), dtype=np.float32)
    hv_sum = np.zeros((2, height, width), dtype=np.float32)
    type_sum = np.zeros((num_types, height, width), dtype=np.float32)
    model.eval()
    for start in range(0, len(locations), batch_size):
        batch_locations = locations[start : start + batch_size]
        patches = np.stack(
            [image[y : y + patch_size, x : x + patch_size].transpose(2, 0, 1) for y, x in batch_locations]
        ).astype(np.float32) / 255.0
        output = model(torch.from_numpy(patches).to(device))
        np_batch = output.np_logits.softmax(1).cpu().numpy()
        hv_batch = output.hv_map.cpu().numpy()
        type_batch = output.type_logits.softmax(1).cpu().numpy()
        prediction_size = np_batch.shape[-1]
        if prediction_size == output_size:
            source_start = 0
        elif prediction_size == patch_size:
            source_start = margin_before
        else:
            raise ValueError(
                f"Expected model output size {output_size} or {patch_size}, got {prediction_size}"
            )
        for index, (y, x) in enumerate(batch_locations):
            source = np.s_[
                source_start : source_start + output_size,
                source_start : source_start + output_size,
            ]
            area = np.s_[y : y + output_size, x : x + output_size]
            np_sum[:, area[0], area[1]] = np_batch[index, :, source[0], source[1]]
            hv_sum[:, area[0], area[1]] = hv_batch[index, :, source[0], source[1]]
            type_sum[:, area[0], area[1]] = type_batch[index, :, source[0], source[1]]
    crop = np.s_[:original_height, :original_width]
    np_probability = np_sum[1][crop]
    hv_map = hv_sum[:, crop[0], crop[1]].transpose(1, 2, 0)
    type_probability = type_sum[:, crop[0], crop[1]].transpose(1, 2, 0)
    return np_probability, hv_map, type_probability


def evaluate(args) -> Path:
    seed_everything(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    dataset = args.dataset or config["dataset"]
    info = DATASET_INFO[dataset]
    channels = tuple(config.get("encoder_channels", (64, 128, 256, 512, 1024)))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    # Checkpoints created before the corrected objective used a tanh HV head.
    architecture = args.architecture or config.get("architecture", "unet")
    if architecture == "unet":
        model = OriginalUNet(
            info["num_types"], channels, bound_hv=not config.get("unbounded_hv", False)
        ).to(device)
    else:
        model = _create_model(architecture, info["num_types"], channels).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint.get("desc", checkpoint)))

    records = load_manifest(Path(args.data_dir), dataset)
    _, _, test_records = split_records(records, args.seed, config.get("val_fraction", 0.2))
    result_dir = Path(args.output_dir) / "results" / dataset / args.run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    model_profile = profile_model(model, device, args.patch_size)
    write_model_summary(result_dir / "model_summary.txt", model, model_profile)
    (result_dir / "model_profile.json").write_text(json.dumps(model_profile, indent=2))
    print(
        f"Model: parameters={model_profile['total_parameters']:,} "
        f"GFLOPs={model_profile['gflops']:.3f} "
        f"tile_latency={model_profile['tile_inference_mean_ms']:.2f}ms "
        f"tiles/s={model_profile['tiles_per_second']:.2f}",
        flush=True,
    )
    rows = []
    pooled_type_counts = np.zeros((info["num_types"], 3), dtype=np.int64)
    hover_type_counts = np.zeros((info["num_types"], 5), dtype=np.int64)
    evaluation_start = time.perf_counter()
    total_pixels = 0
    for index, record in enumerate(test_records, start=1):
        with np.load(record["path"]) as sample:
            image = sample["image"]
            true_map = sample["instance_map"].astype(np.int32)
            true_type_map = sample["type_map"].astype(np.int16)
        total_pixels += image.shape[0] * image.shape[1]
        _synchronize(device)
        inference_start = time.perf_counter()
        np_probability, hv_map, type_probability = tiled_predict(
            model,
            image,
            device,
            args.patch_size,
            args.overlap,
            args.batch_size,
            args.tile_output_size,
        )
        _synchronize(device)
        inference_seconds = time.perf_counter() - inference_start
        postprocess_start = time.perf_counter()
        pred_map, pred_types = post_process_instances(
            np_probability,
            hv_map,
            type_probability,
            nucleus_threshold=args.nucleus_threshold,
            boundary_threshold=args.boundary_threshold,
            min_size=args.min_size,
            type_boundary_weight=args.type_boundary_weight,
        )
        postprocess_seconds = time.perf_counter() - postprocess_start
        true_types = instance_types_from_map(true_map, true_type_map)
        metrics = evaluate_instances(true_map, pred_map, true_types, pred_types, info["type_names"])
        pooled_type_counts += type_classification_counts(
            true_map, pred_map, true_types, pred_types, info["num_types"]
        )
        hover_type_counts += hover_type_classification_counts(
            true_map, pred_map, true_types, pred_types, info["num_types"]
        )
        name = Path(record["path"]).stem
        rows.append(
            {
                "image": name,
                "inference_seconds": inference_seconds,
                "postprocess_seconds": postprocess_seconds,
                **metrics,
            }
        )
        prediction_path = result_dir / "predictions" / f"{name}.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            instance_map=pred_map,
            instance_types=pred_types,
            type_map=type_probability.argmax(-1).astype(np.int16),
            nucleus_probability=np_probability.astype(np.float16),
            hv_map=hv_map.astype(np.float16),
            type_probability=type_probability.astype(np.float16),
        )
        if index <= args.visualizations:
            overlay_path = result_dir / "overlays" / f"{name}.png"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(instance_overlay(image, pred_map, pred_types)).save(overlay_path)
            save_prediction_visualization(
                result_dir / "visualizations" / f"{name}.png",
                image,
                true_map,
                true_types,
                pred_map,
                pred_types,
                np_probability,
                hv_map,
                info["type_names"],
            )
        print(
            f"[{index}/{len(test_records)}] {rows[-1]['image']} PQ={metrics['pq']:.4f} "
            f"inference={inference_seconds:.3f}s postprocess={postprocess_seconds:.3f}s",
            flush=True,
        )

    fieldnames = list(rows[0]) if rows else ["image"]
    with (result_dir / "per_image_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in fieldnames[1:]
    } if rows else {}
    pooled_f1 = []
    for class_id, name in enumerate(info["type_names"][1:], start=1):
        tp, fp, fn = pooled_type_counts[class_id]
        score = float(2 * tp / max(2 * tp + fp + fn, 1))
        summary[f"pooled_f1_{name}"] = score
        if tp + fn > 0:
            pooled_f1.append(score)
    summary["pooled_type_macro_f1"] = float(np.mean(pooled_f1)) if pooled_f1 else 0.0
    hover_f1 = []
    for class_id, name in enumerate(info["type_names"][1:], start=1):
        tp, paired_fp, paired_fn, detection_fp, detection_fn = hover_type_counts[class_id]
        score = float(
            2 * tp
            / max(2 * tp + 2 * paired_fp + 2 * paired_fn + detection_fp + detection_fn, 1)
        )
        summary[f"hover_f1_{name}"] = score
        hover_f1.append(score)
    summary["hover_type_macro_f1"] = float(np.mean(hover_f1)) if hover_f1 else 0.0
    inference_total = float(sum(row["inference_seconds"] for row in rows))
    postprocess_total = float(sum(row["postprocess_seconds"] for row in rows))
    summary.update(
        test_images=len(rows),
        test_inference_seconds_total=inference_total,
        test_inference_seconds_mean=inference_total / max(len(rows), 1),
        test_postprocess_seconds_total=postprocess_total,
        test_throughput_images_per_second=len(rows) / max(inference_total, 1.0e-12),
        test_throughput_megapixels_per_second=(total_pixels / 1.0e6)
        / max(inference_total, 1.0e-12),
        test_wall_seconds=time.perf_counter() - evaluation_start,
        visualizations_created=min(args.visualizations, len(rows)),
        **model_profile,
    )
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return result_dir


def rescore(args) -> Path:
    """Recompute metrics from saved instance predictions without GPU inference."""
    info = DATASET_INFO[args.dataset]
    records = load_manifest(Path(args.data_dir), args.dataset)
    test_records = [record for record in records if record["split"] == "test"]
    result_dir = Path(args.output_dir) / "results" / args.dataset / args.run_name
    rows = []
    pooled = np.zeros((info["num_types"], 3), dtype=np.int64)
    hover = np.zeros((info["num_types"], 5), dtype=np.int64)
    for record in test_records:
        name = Path(record["path"]).stem
        with np.load(record["path"]) as sample:
            true_map = sample["instance_map"].astype(np.int32)
            true_type_map = sample["type_map"].astype(np.int16)
        with np.load(result_dir / "predictions" / f"{name}.npz") as prediction:
            pred_map = prediction["instance_map"].astype(np.int32)
            pred_types = prediction["instance_types"].astype(np.int16)
        true_types = instance_types_from_map(true_map, true_type_map)
        metrics = evaluate_instances(true_map, pred_map, true_types, pred_types, info["type_names"])
        pooled += type_classification_counts(true_map, pred_map, true_types, pred_types, info["num_types"])
        hover += hover_type_classification_counts(true_map, pred_map, true_types, pred_types, info["num_types"])
        rows.append({"image": name, **metrics})
    fieldnames = list(rows[0])
    with (result_dir / "per_image_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: float(np.mean([row[key] for row in rows])) for key in fieldnames[1:]}
    pooled_f1, hover_f1 = [], []
    for class_id, name in enumerate(info["type_names"][1:], start=1):
        tp, fp, fn = pooled[class_id]
        score = float(2 * tp / max(2 * tp + fp + fn, 1))
        summary[f"pooled_f1_{name}"] = score
        pooled_f1.append(score)
        tp, paired_fp, paired_fn, detection_fp, detection_fn = hover[class_id]
        score = float(2 * tp / max(2 * tp + 2 * paired_fp + 2 * paired_fn + detection_fp + detection_fn, 1))
        summary[f"hover_f1_{name}"] = score
        hover_f1.append(score)
    summary["pooled_type_macro_f1"] = float(np.mean(pooled_f1))
    summary["hover_type_macro_f1"] = float(np.mean(hover_f1))
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return result_dir


def regenerate_visualizations(args) -> Path:
    """Regenerate detailed figures from cached predictions without inference."""
    info = DATASET_INFO[args.dataset]
    records = load_manifest(Path(args.data_dir), args.dataset)
    test_records = [record for record in records if record["split"] == "test"]
    result_dir = Path(args.output_dir) / "results" / args.dataset / args.run_name
    prediction_dir = result_dir / "predictions"
    if not prediction_dir.exists():
        raise FileNotFoundError(f"No saved predictions found at {prediction_dir}")
    count = min(args.visualizations, len(test_records))
    for record in test_records[:count]:
        name = Path(record["path"]).stem
        with np.load(record["path"]) as sample:
            image = sample["image"]
            true_map = sample["instance_map"].astype(np.int32)
            true_type_map = sample["type_map"].astype(np.int16)
        with np.load(prediction_dir / f"{name}.npz") as prediction:
            pred_map = prediction["instance_map"].astype(np.int32)
            pred_types = prediction["instance_types"].astype(np.int16)
            nucleus_probability = prediction["nucleus_probability"].astype(np.float32)
            hv_map = prediction["hv_map"].astype(np.float32)
        true_types = instance_types_from_map(true_map, true_type_map)
        save_prediction_visualization(
            result_dir / "visualizations" / f"{name}.png",
            image,
            true_map,
            true_types,
            pred_map,
            pred_types,
            nucleus_probability,
            hv_map,
            info["type_names"],
        )
    print(f"Regenerated {count} visualizations in {result_dir / 'visualizations'}")
    return result_dir / "visualizations"
