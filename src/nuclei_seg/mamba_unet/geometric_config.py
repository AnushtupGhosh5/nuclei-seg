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
