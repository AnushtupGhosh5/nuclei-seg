"""Discrete scan orders derived from a predicted per-instance PDE field.

The construction is deliberately an approximation.  A 2-D vector field does
not define a unique global 1-D traversal, and histology crops contain many
nuclei.  Normal order uses a stable lexicographic sort by potential and local
gradient angle.  Tangential order operates in small spatial windows and sorts
by quantized level set then angle around a window-local soft center.  This
avoids the especially bad single-global-center assumption, but it can still
jump between disconnected nuclei within a window and is not streamline
integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class PDEPermutations:
    """Forward permutations and exact inverses for one spatial resolution."""

    normal: Tensor
    normal_inverse: Tensor
    tangent: Tensor
    tangent_inverse: Tensor
    potential: Tensor
    gradient_x: Tensor
    gradient_y: Tensor


def inverse_permutation(permutation: Tensor) -> Tensor:
    """Return ``inverse`` satisfying ``inverse[permutation] == arange(L)``."""
    if permutation.ndim != 2:
        raise ValueError(f"Expected [B,L] permutation, got {permutation.shape}")
    batch, length = permutation.shape
    inverse = torch.empty_like(permutation)
    positions = torch.arange(length, device=permutation.device).expand(batch, -1)
    inverse.scatter_(1, permutation, positions)
    return inverse


def _stable_lexsort(keys: list[Tensor]) -> Tensor:
    """Stable batch lexsort; keys are ordered most-to-least significant."""
    if not keys or any(key.ndim != 2 for key in keys):
        raise ValueError("Lexicographic keys must be non-empty [B,L] tensors")
    batch, length = keys[0].shape
    if any(key.shape != (batch, length) for key in keys):
        raise ValueError("All lexicographic keys must have identical shapes")
    permutation = torch.arange(
        length, device=keys[0].device, dtype=torch.long
    ).expand(batch, -1)
    for key in reversed(keys):
        values = torch.gather(key, 1, permutation)
        order = torch.argsort(values, dim=1, stable=True)
        permutation = torch.gather(permutation, 1, order)
    return permutation


def _field_gradient(potential: Tensor) -> tuple[Tensor, Tensor]:
    """Centered finite differences for a ``[B,H,W]`` potential."""
    padded_x = F.pad(potential[:, None], (1, 1, 0, 0), mode="replicate")
    padded_y = F.pad(potential[:, None], (0, 0, 1, 1), mode="replicate")
    gradient_x = 0.5 * (padded_x[:, 0, :, 2:] - padded_x[:, 0, :, :-2])
    gradient_y = 0.5 * (padded_y[:, 0, 2:, :] - padded_y[:, 0, :-2, :])
    return gradient_x, gradient_y


def _window_geometry(
    potential: Tensor, window_size: int, epsilon: float
) -> tuple[Tensor, Tensor]:
    """Return serpentine window IDs and local polar angle for each token."""
    batch, height, width = potential.shape
    device = potential.device
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=potential.dtype),
        torch.arange(width, device=device, dtype=potential.dtype),
        indexing="ij",
    )
    number_columns = (width + window_size - 1) // window_size
    window_row = y.long() // window_size
    window_column = x.long() // window_size
    serpentine_column = torch.where(
        window_row.remainder(2) == 0,
        window_column,
        number_columns - 1 - window_column,
    )
    window_order = window_row * number_columns + serpentine_column
    number_windows = int(window_order.max()) + 1
    flat_window = window_order.flatten().long()

    weights = potential.flatten(1).clamp_min(0)
    # A tiny uniform term makes an all-zero window use its geometric center.
    weights = weights + epsilon
    index = flat_window[None].expand(batch, -1)
    sums = torch.zeros(batch, number_windows, device=device, dtype=potential.dtype)
    sums_x = torch.zeros_like(sums)
    sums_y = torch.zeros_like(sums)
    sums.scatter_add_(1, index, weights)
    sums_x.scatter_add_(1, index, weights * x.flatten()[None])
    sums_y.scatter_add_(1, index, weights * y.flatten()[None])
    center_x = torch.gather(sums_x / sums.clamp_min(epsilon), 1, index)
    center_y = torch.gather(sums_y / sums.clamp_min(epsilon), 1, index)
    angle = torch.atan2(
        y.flatten()[None] - center_y,
        x.flatten()[None] - center_x,
    )
    return flat_window[None].expand(batch, -1), angle


def build_pde_permutations(
    guidance: Tensor,
    height: int,
    width: int,
    *,
    num_pde_bins: int = 16,
    window_size: int = 4,
    epsilon: float = 1.0e-6,
) -> PDEPermutations:
    """Build deterministic normal and approximate level-set permutations.

    Args:
        guidance: Model-predicted guidance, shaped ``[B,1,Hg,Wg]``.  Callers
            should detach it before this non-differentiable ordering operation.
        height, width: Target feature-map resolution.
        num_pde_bins: Number of potential bands used by tangential ordering.
        window_size: Local window side length in target feature tokens.
    """
    if guidance.ndim != 4 or guidance.shape[1] != 1:
        raise ValueError(f"Expected guidance [B,1,H,W], got {guidance.shape}")
    if height < 1 or width < 1 or num_pde_bins < 2 or window_size < 1:
        raise ValueError("Invalid PDE scan dimensions or configuration")
    potential = F.interpolate(
        guidance.float(), size=(height, width), mode="bilinear",
        align_corners=False,
    )[:, 0].clamp(0, 1)
    gradient_x, gradient_y = _field_gradient(potential)
    gradient_angle = torch.atan2(gradient_y, gradient_x).flatten(1)
    flat_potential = potential.flatten(1)
    spatial = torch.arange(
        height * width, device=guidance.device, dtype=torch.long
    )[None].expand(guidance.shape[0], -1)

    normal = _stable_lexsort(
        [flat_potential, gradient_angle, spatial]
    )

    window_order, local_angle = _window_geometry(
        potential, window_size, epsilon
    )
    level_bin = torch.floor(flat_potential * num_pde_bins).long().clamp(
        0, num_pde_bins - 1
    )
    tangent = _stable_lexsort(
        [window_order, level_bin, local_angle, spatial]
    )
    return PDEPermutations(
        normal=normal,
        normal_inverse=inverse_permutation(normal),
        tangent=tangent,
        tangent_inverse=inverse_permutation(tangent),
        potential=potential,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
    )


class PDEPermutationCache:
    """Per-forward cache: each batch/resolution is sorted at most once."""

    def __init__(
        self, guidance: Tensor, *, num_pde_bins: int, window_size: int
    ) -> None:
        self.guidance = guidance.detach()
        self.num_pde_bins = int(num_pde_bins)
        self.window_size = int(window_size)
        self._cache: dict[tuple[int, int], PDEPermutations] = {}

    def get(self, height: int, width: int) -> PDEPermutations:
        key = (int(height), int(width))
        if key not in self._cache:
            self._cache[key] = build_pde_permutations(
                self.guidance, *key,
                num_pde_bins=self.num_pde_bins,
                window_size=min(self.window_size, *key),
            )
        return self._cache[key]


def gather_sequence(values: Tensor, permutation: Tensor) -> Tensor:
    """Gather ``[B,C,L]`` values into a batch-specific scan order."""
    return torch.gather(
        values, 2,
        permutation[:, None].expand(-1, values.shape[1], -1),
    )


def restore_sequence(values: Tensor, inverse: Tensor) -> Tensor:
    """Restore geometry-ordered ``[B,C,L]`` values to raster order."""
    return torch.gather(
        values, 2,
        inverse[:, None].expand(-1, values.shape[1], -1),
    )

