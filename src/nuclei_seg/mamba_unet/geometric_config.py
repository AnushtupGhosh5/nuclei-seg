"""Configuration for PDE-guided geometric Mamba ablations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import MambaUNetConfig
from .pde_ss2d import SCAN_MODES


@dataclass(slots=True)
class PDEGeometricConfig(MambaUNetConfig):
    output_dir: Path = Path("outputs/pde_geometric_mamba_hybrid")
    scan_mode: str = "hybrid"
    pde_scan_stages: tuple[str, ...] = (
        "encoder_2", "encoder_3", "decoder_1"
    )
    guide_source_stage: int = 1
    guide_detach_for_sort: bool = True
    num_pde_bins: int = 16
    pde_scan_window: int = 4
    pde_target_iterations: int = 48
    pde_normal_direction_bins: int = 8
    pde_normal_ray_width: float = 1.0
    pde_gradient_epsilon: float = 0.01
    pde_normal_scan_algorithm: str = "local_rays"

    early_stopping_min_epoch: int = 75
    type_loss_foreground_only: bool = False
    type_weight_basis: str = "legacy_pixel"

    mask_loss_weight: float = 1.0
    field_loss_weight: float = 1.0
    type_loss_weight: float = 1.0
    guide_loss_weight: float = 0.5
    field_background_weight: float = 0.1

    nucleus_threshold: float = 0.5
    marker_threshold: float = 0.22
    minimum_peak_distance: int = 9
    minimum_object_size: int = 10
    field_smoothing_sigma: float = 1.5
    visualization_count: int = 5

    def validate(self) -> None:
        # Explicit base call avoids zero-argument super() edge cases with
        # slotted dataclass inheritance on Python 3.12.
        MambaUNetConfig.validate(self)
        if self.scan_mode not in SCAN_MODES:
            raise ValueError(f"Unsupported scan_mode: {self.scan_mode}")
        allowed = {
            "encoder_0", "encoder_1", "encoder_2", "encoder_3",
            "decoder_1", "decoder_2", "decoder_3",
        }
        unknown = set(self.pde_scan_stages) - allowed
        if unknown:
            raise ValueError(f"Unknown PDE scan stages: {sorted(unknown)}")
        if self.scan_mode != "cartesian":
            encoder_stages = [
                int(stage.split("_")[1])
                for stage in self.pde_scan_stages
                if stage.startswith("encoder_")
            ]
            if encoder_stages and min(encoder_stages) <= self.guide_source_stage:
                raise ValueError(
                    "Guided encoder stages must occur after guide_source_stage"
                )
        if self.guide_source_stage not in {0, 1, 2}:
            raise ValueError("guide_source_stage must be 0, 1, or 2")
        if self.num_pde_bins < 2 or self.pde_scan_window < 1:
            raise ValueError("Invalid PDE permutation configuration")
        if self.pde_target_iterations < 1 or self.eval_every < 1:
            raise ValueError("Target iterations and evaluation interval must be positive")
        if self.pde_normal_direction_bins < 2 or self.pde_normal_ray_width <= 0:
            raise ValueError("Invalid normal direction/ray configuration")
        if self.pde_gradient_epsilon <= 0:
            raise ValueError("pde_gradient_epsilon must be positive")
        if self.pde_normal_scan_algorithm not in {"local_rays", "legacy_global"}:
            raise ValueError("Unknown normal scan algorithm")
        if self.early_stopping_min_epoch < 75:
            raise ValueError("Early stopping must wait for the StepLR decay at epoch 75")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive or null")
        if self.type_weight_basis not in {"legacy_pixel", "foreground_pixel", "instance"}:
            raise ValueError("Unknown type class-weight basis")
        if not 0 < self.nucleus_threshold < 1:
            raise ValueError("nucleus_threshold must lie in (0,1)")
        if not 0 <= self.marker_threshold <= 1:
            raise ValueError("marker_threshold must lie in [0,1]")
        if min(
            self.mask_loss_weight, self.field_loss_weight,
            self.type_loss_weight, self.guide_loss_weight,
            self.field_background_weight,
        ) < 0:
            raise ValueError("Loss weights must be non-negative")
