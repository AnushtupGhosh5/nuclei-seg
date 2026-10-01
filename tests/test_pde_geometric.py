from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

from nuclei_seg.mamba_unet.geometric_config import PDEGeometricConfig
from nuclei_seg.mamba_unet.geometric_checkpoint import CHECKPOINT_SELECTION, LossCheckpointTracker, should_early_stop
from nuclei_seg.mamba_unet.geometric_loss import PDEGeometricLoss
from nuclei_seg.mamba_unet.data import type_class_counts
from nuclei_seg.mamba_unet.geometric_experiment import geometric_loss_weights, guided_stage_resolutions, predict_tile, run_experiment, run_metadata, save_scan_diagnostics, validation_loss
from nuclei_seg.mamba_unet.geometric_model import (
    PDEGeometricMambaUNet,
    verify_geometric_output_contract,
)
from nuclei_seg.mamba_unet.geometric_postprocess import mask_pde_watershed
from nuclei_seg.mamba_unet.model import MambaUNetNP_HV_Type
from nuclei_seg.mamba_unet.pde_field import make_poisson_field
from nuclei_seg.mamba_unet.pde_scan import (
    PDEPermutationCache,
    build_pde_permutations,
    gather_sequence,
    inverse_permutation,
    restore_sequence,
    scan_diagnostics,
)
from nuclei_seg.mamba_unet.pde_ss2d import PDEGuidedSS2D


def _touching_instances(size: int = 96) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    first = (x - 38) ** 2 + (y - 48) ** 2 <= 16**2
    second = (x - 58) ** 2 + (y - 48) ** 2 <= 16**2
    instances = np.zeros((size, size), np.int32)
    instances[first] = 1
    instances[second] = 2
    return instances


def _irregular_guidance() -> torch.Tensor:
    field = torch.zeros(1, 1, 19, 23)
    field[:, :, 2:15, 3:7] = torch.linspace(0.1, 1.0, 13)[None, None, :, None]
    field[:, :, 11:16, 7:18] = 0.7
    field[:, :, 4:9, 15:21] = 0.4
    return field


def test_instancewise_pde_target_is_finite_bounded_and_separate() -> None:
    instances = _touching_instances()
    field = make_poisson_field(instances, iterations=64)
    assert np.isfinite(field).all()
    assert 0 <= float(field.min()) <= float(field.max()) <= 1.0
    for instance_id in (1, 2):
        mask = instances == instance_id
        interior = ndi.binary_erosion(mask)
        contour = mask & ~interior
        assert np.all(field[contour] == 0)
        assert np.all(field[interior] > 0)
        values = field[instances == instance_id]
        assert np.isclose(values.max(), 1.0)
        assert values.mean() > 0
    maxima = ndi.maximum_filter(field, size=9) == field
    assert sum(np.any(maxima & (field > 0) & (instances == item)) for item in (1, 2)) == 2


@pytest.mark.parametrize("shape", [(1, 1), (2, 5), (8, 9)])
def test_poisson_degenerate_and_image_edge_contours(shape):
    labels = np.ones(shape, np.int32)
    field = make_poisson_field(labels, iterations=8)
    interior = ndi.binary_erosion(labels > 0)
    assert np.all(field[~interior] == 0)
    assert np.isfinite(field).all()
    if interior.any():
        assert np.all(field[interior] > 0)
        assert field.max() == pytest.approx(1)
    else:
        assert not field.any()


def test_poisson_empty_and_invalid_iterations():
    assert not make_poisson_field(np.zeros((4, 4), np.int32)).any()
    with pytest.raises(ValueError):
        make_poisson_field(np.ones((4, 4), np.int32), iterations=0)


def test_mask_pde_watershed_recovers_two_touching_nuclei() -> None:
    instances = _touching_instances()
    field = make_poisson_field(instances, iterations=96)
    mask = (instances > 0).astype(np.float32) * 0.99
    predicted = mask_pde_watershed(
        mask, field, nucleus_threshold=0.5,
        marker_threshold=0.3, min_distance=8,
        min_size=10, smoothing_sigma=1.0,
    )
    assert predicted.max() == 2


@pytest.mark.parametrize("height,width", [(8, 8), (16, 16), (13, 17)])
def test_pde_permutations_are_bijections_and_restore_exactly(
    height: int, width: int
) -> None:
    guidance = _irregular_guidance().repeat(2, 1, 1, 1)
    permutations = build_pde_permutations(
        guidance, height, width, num_pde_bins=16, window_size=4
    )
    expected = torch.arange(height * width)
    for permutation, inverse in (
        (permutations.normal, permutations.normal_inverse),
        (permutations.tangent, permutations.tangent_inverse),
    ):
        assert torch.equal(torch.sort(permutation[0]).values, expected)
        assert torch.equal(inverse[0, permutation[0]], expected)
        values = torch.randn(2, 5, height * width)
        restored = restore_sequence(
            gather_sequence(values, permutation), inverse
        )
        assert torch.equal(values, restored)


def test_forward_reverse_orders_and_irregular_modes_differ() -> None:
    permutations = build_pde_permutations(
        _irregular_guidance(), 16, 16,
        num_pde_bins=16, window_size=4,
    )
    normal_reverse = torch.flip(permutations.normal, dims=[-1])
    tangent_reverse = torch.flip(permutations.tangent, dims=[-1])
    assert torch.equal(torch.flip(normal_reverse, dims=[-1]), permutations.normal)
    assert torch.equal(torch.flip(tangent_reverse, dims=[-1]), permutations.tangent)
    assert not torch.equal(permutations.normal, permutations.tangent)

    values = torch.arange(256).reshape(1, 1, 256).float()
    for forward, inverse in (
        (permutations.normal, permutations.normal_inverse),
        (permutations.tangent, permutations.tangent_inverse),
    ):
        reverse = torch.flip(forward, dims=[-1])
        reverse_inverse = inverse_permutation(reverse)
        assert torch.equal(
            restore_sequence(gather_sequence(values, reverse), reverse_inverse),
            values,
        )
        # This is the exact flip/restore path used after a reverse selective
        # scan in PDEGuidedSS2D (identity scan used here for isolation).
        reversed_sequence = torch.flip(
            gather_sequence(values, forward), dims=[-1]
        )
        assert torch.equal(
            restore_sequence(
                torch.flip(reversed_sequence, dims=[-1]), inverse
            ),
            values,
        )


def test_permutation_cache_reuses_each_resolution() -> None:
    cache = PDEPermutationCache(
        _irregular_guidance(), num_pde_bins=16, window_size=4
    )
    assert cache.get(16, 16) is cache.get(16, 16)
    assert cache.get(8, 8) is not cache.get(16, 16)


def test_normal_locality_and_rank_diagnostics():
    field = _irregular_guidance()
    base = build_pde_permutations(field, 16, 16)
    legacy = build_pde_permutations(field, 16, 16, normal_algorithm="legacy_global")
    noise = torch.randn(field.shape, generator=torch.Generator().manual_seed(42)) * 1e-3
    other = build_pde_permutations((field + noise).clamp(0, 1), 16, 16)
    stats = scan_diagnostics(base, other)
    old_stats = scan_diagnostics(legacy, legacy)
    assert stats["normal"]["mean_consecutive_spatial_jump_tokens"] < 0.6 * old_stats["normal"]["mean_consecutive_spatial_jump_tokens"]
    assert stats["normal"]["max_consecutive_spatial_jump_tokens"] < old_stats["normal"]["max_consecutive_spatial_jump_tokens"]
    noisy_legacy = build_pde_permutations((field + noise).clamp(0, 1), 16, 16, normal_algorithm="legacy_global")
    assert stats["normal"]["mean_absolute_rank_change_under_1e-3_noise"] < scan_diagnostics(legacy, noisy_legacy)["normal"]["mean_absolute_rank_change_under_1e-3_noise"]
    # Windows bound every within-group jump; window transitions are included
    # in the reported all-step jump, excluded from directional alignment.
    for name in ("normal", "tangential"):
        result = stats[name]
        assert result["permutation_exact"] and result["inverse_exact"] and result["gather_restore_exact"]
        assert result["valid_alignment_steps"] > 0
        assert -1 <= result["mean_signed_directional_cosine"] <= 1
        assert 0 <= result["mean_absolute_directional_cosine"] <= 1
        assert all(np.isfinite(value) for value in result.values())
    assert stats["normal"]["mean_signed_directional_cosine"] > 0.25


def test_directional_alignment_on_planar_field():
    x = torch.linspace(0.1, 0.9, 16)
    field = x[None, None, None].expand(1, 1, 16, 16)
    base = build_pde_permutations(field, 16, 16)
    stats = scan_diagnostics(base, base)
    assert stats["normal"]["mean_signed_directional_cosine"] > 0.99
    # Coarse bands can combine adjacent columns, so contour steps approximate
    # a tangent. Narrow bands isolate columns and recover exact vertical steps.
    assert stats["tangential"]["mean_absolute_directional_cosine"] > 0.8
    narrow = build_pde_permutations(field, 16, 16, num_pde_bins=64)
    assert scan_diagnostics(narrow, narrow)["tangential"]["mean_absolute_directional_cosine"] > 0.99


def test_single_token_diagnostics_and_flat_fallback():
    base = build_pde_permutations(torch.zeros(1, 1, 1, 1), 1, 1)
    stats = scan_diagnostics(base, base)
    assert stats["normal"]["valid_alignment_steps"] == 0
    assert stats["normal"]["mean_consecutive_spatial_jump_tokens"] == 0
    flat = torch.zeros(1, 1, 16, 16)
    noisy = flat + 1e-3 * torch.rand(flat.shape, generator=torch.Generator().manual_seed(42))
    assert torch.equal(build_pde_permutations(flat, 16, 16).normal_group, build_pde_permutations(noisy, 16, 16).normal_group)


@pytest.mark.parametrize("present_ids", [[], [1], [4], [1, 3]])
def test_foreground_type_loss_present_classes_and_gradients(present_ids):
    config = PDEGeometricConfig(type_classes=("background", "a", "b", "c", "d"), type_loss_foreground_only=True)
    criterion = PDEGeometricLoss(torch.ones(2), torch.ones(5), config)
    logits = torch.randn(1, 5, 4, 4, generator=torch.Generator().manual_seed(42), requires_grad=True)
    instances = torch.zeros(1, 4, 4, dtype=torch.long)
    types = torch.zeros_like(instances)
    for i, class_id in enumerate(present_ids):
        instances[:, i, :] = i + 1
        types[:, i, :] = class_id
    ce, dice = criterion._type_losses(logits, types, (instances > 0).long())
    assert torch.isfinite(ce + dice)
    if present_ids:
        valid = instances > 0
        selected = logits.permute(0, 2, 3, 1)[valid]
        assert float(ce.detach()) == pytest.approx(float(torch.nn.functional.cross_entropy(selected, types[valid]).detach()))
        scores = []
        for class_id in present_ids:
            p = selected.softmax(1)[:, class_id]
            t = (types[valid] == class_id).float()
            scores.append((2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1))
        assert torch.allclose(dice, 1 - torch.stack(scores).mean())
    else:
        assert ce == 0 and dice == 0
    (ce + dice).backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad.permute(0, 2, 3, 1)[instances == 0] == 0)
    # Also exclude untyped pixels inside nuclei.
    types.zero_()
    ce, dice = criterion._type_losses(logits, types, torch.ones_like(instances))
    assert ce == 0 and dice == 0


def test_loss_subcomponents_sum_to_aggregates():
    config = PDEGeometricConfig(type_loss_foreground_only=True)
    loss = PDEGeometricLoss(torch.ones(2), torch.ones(4), config)
    output = {"np": torch.randn(1, 2, 8, 8), "tp": torch.randn(1, 4, 8, 8), "field": torch.randn(1, 1, 8, 8), "guide": torch.randn(1, 1, 4, 4)}
    instances = torch.zeros(1, 8, 8, dtype=torch.long)
    instances[:, 2:6, 2:6] = 1
    total, parts = loss(output, instances, instances, torch.zeros(1, 8, 8))
    assert parts["mask"] == pytest.approx(parts["mask_ce"] + parts["mask_dice"])
    assert parts["type"] == pytest.approx(parts["type_ce"] + parts["type_dice"])
    for name in ("field", "guide"):
        assert parts[name] == pytest.approx(parts[name + "_foreground"] + 0.1 * parts[name + "_background"])
    assert float(total) == pytest.approx(parts["mask"] + parts["type"] + parts["field"] + 0.5 * parts["guide"])


def test_legacy_type_objective_is_retained():
    from nuclei_seg.mamba_unet.loss import dice_loss
    criterion = PDEGeometricLoss(torch.ones(2), torch.tensor([0.1, 0.5, 1.0, 2.0]), PDEGeometricConfig())
    logits = torch.randn(1, 4, 4, 4)
    types = torch.arange(16).remainder(4).reshape(1, 4, 4)
    ce, dice = criterion._type_losses(logits, types, (types > 0).long())
    assert torch.equal(ce, torch.nn.functional.cross_entropy(logits, types, weight=criterion.type_weights))
    assert torch.equal(dice, dice_loss(logits, types, 4))


def test_class_weight_counts_are_instance_aware():
    instances = np.zeros((6, 8), np.int32)
    instances[:3, :4] = 1
    instances[3:, :4] = 2
    instances[0, 6] = 3
    types = np.where(instances == 3, 2, np.where(instances > 0, 1, 0)).astype(np.uint8)
    tiles = [(None, instances, types, "a")]
    assert type_class_counts(tiles, 5, "instance").tolist() == [0, 2, 1, 0, 0]
    assert type_class_counts(tiles, 5, "foreground_pixel").tolist() == [0, 24, 1, 0, 0]
    assert type_class_counts(tiles, 5, "legacy_pixel").tolist() == [23, 24, 1, 0, 0]
    config = PDEGeometricConfig(type_weight_basis="instance", type_loss_foreground_only=True, type_classes=("background", "a", "b", "c", "d"))
    _, weights, metadata = geometric_loss_weights(SimpleNamespace(tiles=tiles), torch.device("cpu"), config)
    assert weights[0] == 0 and weights[3] == weights[4] == 1
    assert weights[2] / weights[1] == pytest.approx(np.sqrt(2))
    assert metadata["counts"] == [0, 2, 1, 0, 0]
    assert torch.isfinite(weights).all()


@pytest.mark.parametrize("metrics", [
    [(0.9, 0.8), (0.7, 0.6), (1.0, 1.0), (0.1, 0.2)],
    [(0.1, 0.2), (0.3, 0.4), (0.6, 0.7), (0.9, 0.95)],
])
def test_loss_only_selects_epoch_15_regardless_of_pq_and_f1(metrics):
    tracker = LossCheckpointTracker()
    history = []
    selected_epoch = None
    for (epoch, loss), (pq, f1) in zip([(1, 3.0), (5, 2.7), (10, 2.9), (15, 2.5)], metrics):
        history.append({"epoch": epoch, "val_total": loss, "pq": pq, "f1": f1})
        if tracker.observe(history[-1]["val_total"], epoch - 1):
            selected_epoch = epoch
    assert selected_epoch == 15
    assert tracker.best_epoch == 14 and tracker.best_val_loss == 2.5
    assert tracker.bad_validations == 0


def test_better_pq_with_worse_loss_keeps_epoch_1_and_increments_patience():
    results = [{"epoch": 1, "total": 3.0, "pq": 0.20}, {"epoch": 5, "total": 3.5, "pq": 0.70}]
    tracker = LossCheckpointTracker()
    assert tracker.observe(results[0]["total"], results[0]["epoch"] - 1)
    assert not tracker.observe(results[1]["total"], results[1]["epoch"] - 1)
    assert tracker.best_epoch == 0 and tracker.best_val_loss == 3.0
    assert tracker.bad_validations == 1


def test_patience_resets_only_on_strictly_lower_total_and_resumes(tmp_path):
    tracker = LossCheckpointTracker()
    assert tracker.observe(3.0, 0)
    assert not tracker.observe(3.0, 4)  # ties do not improve
    assert not tracker.observe(3.5, 9)
    assert tracker.bad_validations == 2
    path = tmp_path / "loss_tracker.pth"
    torch.save({**tracker.state_dict(), "checkpoint_selection": CHECKPOINT_SELECTION}, path)
    restored = LossCheckpointTracker.restore(torch.load(path, weights_only=False))
    assert restored == tracker
    assert restored.observe(2.9, 14)
    assert restored.bad_validations == 0 and restored.best_epoch == 14
    for invalid in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(FloatingPointError):
            restored.observe(invalid, 19)
    assert restored.best_val_loss == 2.9 and restored.bad_validations == 0


def test_early_stopping_tracks_loss_even_when_metrics_increase():
    tracker = LossCheckpointTracker()
    tracker.observe(3.0, 0)
    for epoch, pq, loss in [(5, 0.7, 3.5), (10, 0.8, 3.6), (80, 0.9, 3.7)]:
        tracker.observe(loss, epoch - 1)
        assert pq > 0.2  # logged improvement never resets the loss counter
        assert should_early_stop(epoch, tracker.bad_validations, 3, 80) == (epoch == 80)
    tracker.observe(2.7, 84)
    assert not should_early_stop(85, tracker.bad_validations, 3, 80)


def test_early_stopping_min_epoch_and_legacy_resume():
    for epoch in range(1, 75):
        assert not should_early_stop(epoch, 99, 8, 75)
    assert should_early_stop(75, 8, 8, 75)
    assert not should_early_stop(80, 7, 8, 75)
    assert not should_early_stop(100, 99, None, 75)
    with pytest.warns(UserWarning):
        legacy = LossCheckpointTracker.restore({"best_val_loss": 3.0, "epoch": 4})
    assert legacy.best_val_loss == 3.0 and legacy.bad_validations == 0
    assert legacy.best_epoch == -1  # recovered from selected file by the runner


def test_incompatible_resume_preserves_existing_metadata(tmp_path):
    metadata = tmp_path / "geometric_run_config.json"
    metadata.write_text("existing completed metadata")
    resume = tmp_path / "checkpoint.pth"
    state = {"best_val_loss": 3.0, "best_metric_name": "multiclass_pq_global", "checkpoint_mode": "max"}
    torch.save(state, resume)
    config = PDEGeometricConfig(output_dir=tmp_path, resume_checkpoint=resume)
    with pytest.raises(ValueError, match="Metric-selected checkpoints"):
        run_experiment(config)
    assert metadata.read_text() == "existing completed metadata"


def test_validation_total_uses_actual_returned_loss_and_logs_all_parts():
    class FakeModel:
        def eval(self):
            pass
        def __call__(self, images):
            return {}
    keys = ["mask_ce", "mask_dice", "field_foreground", "field_background", "type_ce", "type_dice", "guide_foreground", "guide_background", "mask", "field", "type", "guide"]
    actual_totals = iter([3.0, 2.0])
    def criterion(*args):
        total = next(actual_totals)
        return torch.tensor(total), {**{key: total for key in keys}, "total": 999.0}
    batch = (torch.zeros(1, 3, 2, 2), torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), "fake")
    parts = validation_loss(FakeModel(), [batch, batch], criterion, torch.device("cpu"), False)
    assert parts["total"] == parts["loss"] == 2.5
    assert all(parts[key] == 2.5 for key in keys)


def test_config_rejects_early_stopping_before_decay_and_source_guidance():
    with pytest.raises(ValueError, match="StepLR"):
        PDEGeometricConfig(early_stopping_min_epoch=74).validate()
    with pytest.raises(ValueError, match="after guide_source_stage"):
        PDEGeometricConfig(guide_source_stage=0, pde_scan_stages=("encoder_0",)).validate()


@pytest.mark.parametrize("legacy", [False, True])
def test_experiment_checkpoint_roundtrip_restores_training_state(monkeypatch, tmp_path, legacy):
    """One tiny synthetic unit step, then resume without another training step."""
    from nuclei_seg.mamba_unet import geometric_experiment as experiment
    models, optimizers, schedulers = [], [], []
    restored_optimizer, restored_scheduler, restored_scaler = [], [], []
    class TinyModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(0.2))
            self.q = torch.nn.Parameter(torch.tensor(0.3))
            models.append(self)
        def forward(self, images):
            batch, _, height, width = images.shape
            return {"np": self.p.expand(batch, 2, height, width), "tp": self.q.expand(batch, 4, height, width), "field": self.p.expand(batch, 1, height, width)}
        def parameter_groups(self):
            return [self.p], [self.q]
        def parameter_report(self):
            return {"total": 2, "guided_ss2d_blocks": 0}
    original_adam = torch.optim.Adam
    original_scheduler = torch.optim.lr_scheduler.StepLR
    original_scaler = torch.amp.GradScaler
    class TrackingAdam(original_adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            optimizers.append(self)
        def load_state_dict(self, state):
            restored_optimizer.append(state)
            return super().load_state_dict(state)
    class TrackingStepLR(original_scheduler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            schedulers.append(self)
        def load_state_dict(self, state):
            restored_scheduler.append(state)
            return super().load_state_dict(state)
    class TrackingScaler(original_scaler):
        def load_state_dict(self, state):
            restored_scaler.append(state)
            return super().load_state_dict(state)
    monkeypatch.setattr(torch.optim, "Adam", TrackingAdam)
    monkeypatch.setattr(torch.optim.lr_scheduler, "StepLR", TrackingStepLR)
    monkeypatch.setattr(torch.amp, "GradScaler", TrackingScaler)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(experiment, "PDEGeometricMambaUNet", TinyModel)
    record = SimpleNamespace(stem="synthetic")
    split = SimpleNamespace(train=[record], val=[record], test=[record])
    instances = torch.zeros(1, 8, 8, dtype=torch.long)
    instances[:, 2:6, 2:6] = 1
    batch = (torch.zeros(1, 3, 8, 8), instances, instances.clone(), torch.zeros(1, 8, 8), ["synthetic"])
    tiles = [(None, instances[0].numpy(), instances[0].numpy(), "synthetic")]
    monkeypatch.setattr(experiment, "discover_records", lambda *args: [record])
    monkeypatch.setattr(experiment, "detect_type_encoding", lambda *args: "already_merged_3_class")
    monkeypatch.setattr(experiment, "build_split", lambda *args: split)
    monkeypatch.setattr(experiment, "create_pde_loaders", lambda *args: (SimpleNamespace(tiles=tiles), [batch], [batch]))
    monkeypatch.setattr(experiment, "ensure_pretrained_checkpoint", lambda *args: tmp_path / "unused.pth")
    monkeypatch.setattr(experiment, "load_official_pretraining", lambda *args: {"loaded": [], "missing_target": []})
    monkeypatch.setattr(experiment, "verify_geometric_output_contract", lambda *args: {})
    monkeypatch.setattr(experiment, "_overhead_report", lambda *args: {})
    # The runner must select the canonical total, even if an alias is wrong.
    monkeypatch.setattr(experiment, "validation_loss", lambda *args: {"total": 3.0, "loss": 999.0})
    monkeypatch.setattr(experiment, "evaluate", lambda *args: {"multiclass_pq_global": 0.70, "instance_macro_f1_detection_aware": 0.90})
    for name in ("save_validation_visualizations", "save_task_visualizations", "save_scan_debug", "save_synthetic_scan_debug", "save_scan_diagnostics", "_archive_results"):
        monkeypatch.setattr(experiment, name, lambda *args: tmp_path / "unused_artifact")
    first_dir = tmp_path / "first"
    config = PDEGeometricConfig(output_dir=first_dir, smoke_test=True, type_loss_foreground_only=True, type_weight_basis="instance")
    experiment.run_experiment(config)
    state = torch.load(first_dir / "geometric_latest_checkpoint.pth", weights_only=False)
    assert state["best_val_loss"] == 3.0 and state["best_epoch"] == state["epoch"] == 0
    assert state["bad_validations"] == 0 and state["checkpoint_selection"] == CHECKPOINT_SELECTION
    assert state["history"][0]["val_total"] == state["history"][0]["best_val_loss_so_far"] == 3.0
    assert state["history"][0]["val_metric_multiclass_pq_global"] == 0.70
    assert {"model", "optimizer", "scheduler", "scaler", "epoch", "history"} <= state.keys()
    assert state["optimizer"]["state"] and state["scheduler"]["last_epoch"] == 1
    assert not (first_dir / "geometric_best_loss_checkpoint.pth").exists()
    if legacy:
        for key in ("best_epoch", "bad_validations", "history", "scaler", "checkpoint_selection"):
            state.pop(key)
        torch.save(state, first_dir / "geometric_latest_checkpoint.pth")
        torch.save(state, first_dir / "geometric_best_checkpoint.pth")
    resumed_dir = tmp_path / "resumed"
    config.output_dir = resumed_dir
    config.resume_checkpoint = first_dir / "geometric_latest_checkpoint.pth"
    if legacy:
        with pytest.warns(UserWarning, match="no patience state"):
            experiment.run_experiment(config)
    else:
        experiment.run_experiment(config)
    assert len(restored_optimizer) == len(restored_scheduler) == 1
    assert len(restored_scaler) == (0 if legacy else 1)
    assert torch.equal(models[-1].p, state["model"]["p"])
    assert optimizers[-1].param_groups[0]["lr"] == state["optimizer"]["param_groups"][0]["lr"]
    assert schedulers[-1].state_dict() == state["scheduler"]
    selected = torch.load(resumed_dir / "geometric_best_checkpoint.pth", weights_only=False)
    assert torch.equal(selected["model"]["p"], state["model"]["p"])
    metadata = json.loads((resumed_dir / "geometric_run_config.json").read_text())
    assert metadata["checkpoint_selection"] == CHECKPOINT_SELECTION
    assert metadata["best_validation_loss"] == 3.0 and metadata["best_epoch"] == 1
    restored_history = (resumed_dir / "geometric_training_history.csv").read_text()
    assert "val_total" in restored_history and "best_val_loss_so_far" in restored_history
    if not legacy:
        assert selected["history"] == state["history"]


def _copy_shell_runner(tmp_path, name):
    script = tmp_path / name
    shutil.copy2(Path(__file__).resolve().parents[1] / name, script)
    return script


@pytest.mark.parametrize("name", ["run_geometric_mamba.sh", "run_geometric_mamba_monusac.sh", "run_geometric_ablations.sh"])
@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_shell_help_never_starts_docker(tmp_path, name, flag):
    script = _copy_shell_runner(tmp_path, name)
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    marker = tmp_path / "docker_started"
    docker = binary_dir / "docker"
    docker.write_text(f'#!/bin/bash\ntouch "{marker}"\nexit 99\n')
    docker.chmod(0o755)
    result = subprocess.run([str(script), flag], env={**os.environ, "PATH": f"{binary_dir}:{os.environ['PATH']}"}, capture_output=True, text=True)
    assert result.returncode == 0 and not marker.exists()
    assert "CONFIG" in result.stdout and "IMAGE_NAME" in result.stdout
    assert all(mode in result.stdout for mode in ("cartesian", "normal", "tangential", "hybrid"))


@pytest.mark.parametrize("dataset", ["monusac", "glysac"])
@pytest.mark.parametrize("custom_config", [False, True])
def test_shell_docker_environment_and_exact_argument_passthrough(tmp_path, dataset, custom_config):
    name = "run_geometric_mamba_monusac.sh" if dataset == "monusac" else "run_geometric_mamba.sh"
    script = _copy_shell_runner(tmp_path, name)
    if dataset == "monusac":
        split = tmp_path / "data/monusac_dataset/split.csv"
        split.parent.mkdir(parents=True)
        split.touch()
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    log = tmp_path / "docker.json"
    docker = binary_dir / "docker"
    docker.write_text('#!/usr/bin/env python3\nimport json, os, sys\nopen(os.environ["GEOMETRIC_TEST_LOG"], "w").write(json.dumps(sys.argv[1:]))\n')
    docker.chmod(0o755)
    env = {**os.environ, "PATH": f"{binary_dir}:{os.environ['PATH']}", "GEOMETRIC_TEST_LOG": str(log), "IMAGE_NAME": "custom-image"}
    env.pop("CONFIG", None)
    if custom_config:
        env["CONFIG"] = "configs/custom with spaces.json"
    args = ["--smoke-test", "--scan-mode", "normal", "--output-dir", "outputs/path with spaces", "--resume-checkpoint", "outputs/path with spaces/latest.pth"]
    result = subprocess.run([str(script), *args], env=env, capture_output=True, text=True)
    assert result.returncode == 0
    actual = json.loads(log.read_text())
    assert actual[:7] == ["run", "--rm", "--gpus", "all", "--shm-size=8g", "--network", "host"]
    assert actual[actual.index("-v") + 1] == f"{tmp_path}:/app"
    assert actual[actual.index("-w") + 1] == "/app"
    assert "custom-image" in actual
    command = actual[actual.index("-lc") + 1]
    assert 'CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src' in command and '"$@"' in command
    expected_config = env.get("CONFIG", f"configs/pde_geometric_mamba_{dataset}_v2.json")
    assert actual[-(len(args) + 3):] == ["bash", "--config", expected_config, *args]


def test_monusac_runner_requires_prepared_split(tmp_path):
    script = _copy_shell_runner(tmp_path, "run_geometric_mamba_monusac.sh")
    result = subprocess.run([str(script), "--smoke-test"], capture_output=True, text=True)
    assert result.returncode == 2 and "prepare_monusac.sh" in result.stdout


@pytest.mark.parametrize("dataset", ["monusac", "glysac"])
@pytest.mark.parametrize("kind", ["--smoke-test", "--full"])
def test_ablation_runner_is_explicit_sequential_and_stops_on_failure(tmp_path, dataset, kind):
    script = _copy_shell_runner(tmp_path, "run_geometric_ablations.sh")
    log = tmp_path / "ablations.jsonl"
    wrapper_name = "run_geometric_mamba_monusac.sh" if dataset == "monusac" else "run_geometric_mamba.sh"
    wrapper = tmp_path / wrapper_name
    wrapper.write_text('#!/usr/bin/env python3\nimport json, os, sys\nwith open(os.environ["GEOMETRIC_TEST_LOG"], "a") as f: f.write(json.dumps(sys.argv[1:]) + "\\n")\nsys.exit(7 if sys.argv[2] == os.environ.get("GEOMETRIC_FAIL_MODE") else 0)\n')
    wrapper.chmod(0o755)
    env = {**os.environ, "GEOMETRIC_TEST_LOG": str(log)}
    for invalid in ([], [dataset], [dataset, "--unknown"], ["invalid", kind]):
        assert subprocess.run([str(script), *invalid], env=env, capture_output=True).returncode == 2
        assert not log.exists()
    result = subprocess.run([str(script), dataset, kind], env=env, capture_output=True)
    assert result.returncode == 0
    suffix = ["--smoke-test"] if kind == "--smoke-test" else []
    assert [json.loads(line) for line in log.read_text().splitlines()] == [["--scan-mode", mode, *suffix] for mode in ("cartesian", "normal", "tangential", "hybrid")]
    log.unlink()
    result = subprocess.run([str(script), dataset, kind], env={**env, "GEOMETRIC_FAIL_MODE": "normal"}, capture_output=True)
    assert result.returncode == 7
    assert [json.loads(line)[1] for line in log.read_text().splitlines()] == ["cartesian", "normal"]


@pytest.mark.parametrize("dataset", ["monusac", "glysac"])
@pytest.mark.parametrize("mode", ["cartesian", "pde", "normal", "tangential", "hybrid"])
def test_python_cli_preserves_mode_specific_smoke_output(monkeypatch, dataset, mode):
    from nuclei_seg.mamba_unet import geometric_experiment as experiment
    received = []
    monkeypatch.setattr("sys.argv", ["geometric_experiment", "--config", f"configs/pde_geometric_mamba_{dataset}_v2.json", "--scan-mode", mode, "--smoke-test"])
    monkeypatch.setattr(experiment, "run_experiment", lambda config: received.append(config) or config.output_dir)
    experiment.main()
    assert received[0].output_dir == Path(f"outputs/pde_geometric_mamba_{dataset}_v2_{mode}_smoke")


def test_python_cli_preserves_explicit_output_and_resume_arguments(monkeypatch, tmp_path):
    from nuclei_seg.mamba_unet import geometric_experiment as experiment
    received = []
    output, resume = tmp_path / "chosen output", tmp_path / "checkpoint.pth"
    monkeypatch.setattr("sys.argv", ["geometric_experiment", "--config", "configs/pde_geometric_mamba_monusac_v2.json", "--scan-mode", "normal", "--smoke-test", "--output-dir", str(output), "--resume-checkpoint", str(resume)])
    monkeypatch.setattr(experiment, "run_experiment", lambda config: received.append(config) or config.output_dir)
    experiment.main()
    assert received[0].output_dir == output and received[0].resume_checkpoint == resume


def test_foreground_type_inference_ignores_background_logit():
    class FakeModel:
        def eval(self):
            return self
        def __call__(self, images):
            batch, _, height, width = images.shape
            np_logits = torch.zeros(batch, 2, height, width)
            np_logits[:, 0] = 10
            np_logits[:, 1, 64:192, 64:192] = 20
            tp_logits = torch.zeros(batch, 4, height, width)
            tp_logits[:, 0] = 1000  # also exercises all-channel softmax underflow
            tp_logits[:, 2] = 10
            return {"np": np_logits, "tp": tp_logits, "field": torch.ones(batch, 1, height, width)}
    config = PDEGeometricConfig(type_loss_foreground_only=True)
    instances, types, _, _ = predict_tile(FakeModel(), np.zeros((256, 256, 3), np.uint8), config, torch.device("cpu"))
    assert (instances > 0).any()
    assert np.all(types[instances > 0] == 2)
    assert np.all(types[instances == 0] == 0)


@pytest.mark.parametrize("dataset,type_channels", [("monusac", 5), ("glysac", 4)])
def test_revised_guide_source_contract_with_cpu_scan_stub(monkeypatch, tmp_path, dataset, type_channels):
    # Exercises all real shape/channel/gather/restore paths. The identity kernel
    # is only a shape substitute, not validation of CUDA selective-scan math.
    from nuclei_seg.mamba_unet.official import mamba_sys
    monkeypatch.setattr(mamba_sys, "selective_scan_fn", lambda values, *args, **kwargs: values)
    config = PDEGeometricConfig.from_json(Path(f"configs/pde_geometric_mamba_{dataset}_v2.json"))
    config.validate()
    model = PDEGeometricMambaUNet(config)
    assert model.guide_head.in_channels == 192
    shapes = verify_geometric_output_contract(model, torch.device("cpu"), 256)
    assert shapes["guide"] == (1, 1, 32, 32)
    assert shapes["tp"] == (1, type_channels, 256, 256)
    assert guided_stage_resolutions(config) == {"encoder_1": 32, "encoder_2": 16, "encoder_3": 8, "decoder_1": 16}
    assert set(model._last_scan_cache._cache) == {(32, 32), (16, 16), (8, 8)}
    metadata = run_metadata(config, {"basis": "instance"})
    assert metadata["guide_native_resolution"] == [32, 32]
    assert metadata["checkpoint_selection"] == "minimum_validation_total_loss"
    assert "checkpoint_metric" not in config.to_dict() and "checkpoint_mode" not in config.to_dict()
    if dataset == "glysac":
        assert config.type_encoding == "auto"
        assert list(config.type_classes) == ["background", "other", "lymphocyte", "epithelial"]
        assert not config.type_loss_foreground_only and config.type_weight_basis == "legacy_pixel"
    assert model.parameter_report()["guided_ss2d_blocks"] == 8
    config.output_dir = tmp_path
    report = json.loads(save_scan_diagnostics(model, config).read_text())
    assert report["resolution"] == [32, 32]
    assert set(report["by_resolution"]) == {"32x32", "16x16", "8x8"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="selective scan requires CUDA")
def test_revised_guide_source_cuda_contract():
    config = PDEGeometricConfig.from_json(Path("configs/pde_geometric_mamba_monusac_v2.json"))
    model = PDEGeometricMambaUNet(config).cuda()
    shapes = verify_geometric_output_contract(model, torch.device("cuda"), 256)
    assert shapes["guide"] == (1, 1, 32, 32)


def test_guided_model_api_cannot_accept_ground_truth_guidance() -> None:
    signature = inspect.signature(PDEGeometricMambaUNet.forward)
    assert list(signature.parameters) == ["self", "images"]


def test_original_cartesian_model_is_structurally_unchanged() -> None:
    original = MambaUNetNP_HV_Type()
    assert not any(
        isinstance(module, PDEGuidedSS2D) for module in original.modules()
    )
    assert original.parameter_report()["total"] == 19_122_056


@pytest.mark.skipif(not torch.cuda.is_available(), reason="selective scan requires CUDA")
@pytest.mark.parametrize(
    "scan_mode", ["cartesian", "pde", "hybrid", "normal", "tangential"]
)
def test_all_ablation_output_shapes_are_finite(scan_mode: str) -> None:
    config = PDEGeometricConfig(scan_mode=scan_mode, smoke_test=True)
    model = PDEGeometricMambaUNet(config).cuda()
    shapes = verify_geometric_output_contract(model, torch.device("cuda"), 256)
    assert shapes["np"] == (1, 2, 256, 256)
    assert shapes["field"] == (1, 1, 256, 256)
    assert shapes["tp"] == (1, 4, 256, 256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="selective scan requires CUDA")
def test_zero_gate_hybrid_preserves_cartesian_block_output() -> None:
    from nuclei_seg.mamba_unet.official.mamba_sys import SS2D

    torch.manual_seed(42)
    cartesian = SS2D(d_model=24, d_state=8).cuda().eval()
    hybrid = PDEGuidedSS2D.from_cartesian(
        cartesian, scan_mode="hybrid"
    ).cuda().eval()
    hybrid.set_scan_cache(
        PDEPermutationCache(
            torch.rand(2, 1, 8, 8, device="cuda"),
            num_pde_bins=8, window_size=4,
        )
    )
    values = torch.randn(2, 8, 8, 24, device="cuda")
    with torch.inference_mode():
        expected = cartesian(values)
        actual = hybrid(values)
    assert torch.equal(expected, actual)
