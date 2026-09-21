"""Experimental PDE-ordered SS2D modules.

The pinned official :class:`SS2D` remains untouched.  This subclass preserves
all pretrained parameter names and tensor shapes, changing only how the four
spatial sequences are assembled/restored in guided blocks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .official import mamba_sys
from .official.mamba_sys import SS2D
from .pde_scan import PDEPermutationCache, gather_sequence, restore_sequence


SCAN_MODES = {"cartesian", "pde", "hybrid", "normal", "tangential"}


class PDEGuidedSS2D(SS2D):
    """SS2D with normal/tangential scan orders from a predicted PDE field.

    ``hybrid`` computes the unchanged pretrained Cartesian branch plus a PDE
    branch gated per inner channel.  The gate is initialized at zero, so the
    module is exactly Cartesian at initialization.
    """

    def __init__(self, *args, scan_mode: str = "hybrid", **kwargs) -> None:
        if scan_mode not in SCAN_MODES - {"cartesian"}:
            raise ValueError(f"Unsupported guided scan mode: {scan_mode}")
        super().__init__(*args, **kwargs)
        self.scan_mode = scan_mode
        self.pde_gate = nn.Parameter(torch.zeros(self.d_inner))
        self._scan_cache: PDEPermutationCache | None = None

    @classmethod
    def from_cartesian(
        cls, module: SS2D, *, scan_mode: str
    ) -> "PDEGuidedSS2D":
        """Construct from an official module without changing pretrained keys."""
        converted = cls(
            d_model=module.d_model,
            d_state=module.d_state,
            d_conv=module.d_conv,
            expand=module.expand,
            dt_rank=module.dt_rank,
            dropout=(module.dropout.p if module.dropout is not None else 0.0),
            conv_bias=module.conv2d.bias is not None,
            bias=module.in_proj.bias is not None,
            device=module.in_proj.weight.device,
            dtype=module.in_proj.weight.dtype,
            scan_mode=scan_mode,
        )
        missing, unexpected = converted.load_state_dict(
            module.state_dict(), strict=False
        )
        if missing != ["pde_gate"] or unexpected:
            raise RuntimeError(
                f"Unexpected SS2D conversion mismatch: {missing}, {unexpected}"
            )
        return converted

    def set_scan_cache(self, cache: PDEPermutationCache | None) -> None:
        self._scan_cache = cache

    def _selective_scan_geometric(self, x: Tensor) -> Tensor:
        if self._scan_cache is None:
            raise RuntimeError(
                "PDE guidance cache was not set. Inference must use the "
                "model-predicted guidance path, never ground truth guidance."
            )
        if mamba_sys.selective_scan_fn is None:
            raise RuntimeError("mamba-ssm selective_scan_fn is unavailable")
        batch, channels, height, width = x.shape
        length = height * width
        permutations = self._scan_cache.get(height, width)
        raster = x.view(batch, channels, length)
        normal = gather_sequence(raster, permutations.normal)
        tangent = gather_sequence(raster, permutations.tangent)

        if self.scan_mode == "normal":
            sequences = [normal, torch.flip(normal, dims=[-1])]
            parameter_indices = [0, 2]
            restore = [
                (permutations.normal_inverse, False),
                (permutations.normal_inverse, True),
            ]
        elif self.scan_mode == "tangential":
            sequences = [tangent, torch.flip(tangent, dims=[-1])]
            parameter_indices = [1, 3]
            restore = [
                (permutations.tangent_inverse, False),
                (permutations.tangent_inverse, True),
            ]
        else:
            sequences = [
                normal, tangent,
                torch.flip(normal, dims=[-1]),
                torch.flip(tangent, dims=[-1]),
            ]
            parameter_indices = [0, 1, 2, 3]
            restore = [
                (permutations.normal_inverse, False),
                (permutations.tangent_inverse, False),
                (permutations.normal_inverse, True),
                (permutations.tangent_inverse, True),
            ]

        xs = torch.stack(sequences, dim=1)
        count = len(parameter_indices)
        indices = torch.tensor(
            parameter_indices, device=x.device, dtype=torch.long
        )
        x_projection = self.x_proj_weight.index_select(0, indices)
        dt_projection = self.dt_projs_weight.index_select(0, indices)
        projected = torch.einsum(
            "b k d l, k c d -> b k c l", xs, x_projection
        )
        dts, state_b, state_c = torch.split(
            projected,
            [self.dt_rank, self.d_state, self.d_state], dim=2,
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts, dt_projection
        )

        all_a = self.A_logs.view(4, self.d_inner, self.d_state)
        all_d = self.Ds.view(4, self.d_inner)
        all_bias = self.dt_projs_bias.view(4, self.d_inner)
        state_a = -torch.exp(
            all_a.index_select(0, indices).float()
        ).reshape(-1, self.d_state)
        skip_d = all_d.index_select(0, indices).float().reshape(-1)
        dt_bias = all_bias.index_select(0, indices).float().reshape(-1)
        scanned = mamba_sys.selective_scan_fn(
            xs.float().reshape(batch, count * self.d_inner, length),
            dts.contiguous().float().reshape(
                batch, count * self.d_inner, length
            ),
            state_a,
            state_b.float(),
            state_c.float(),
            skip_d,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(batch, count, self.d_inner, length)

        restored = []
        for direction, (inverse, is_reverse) in enumerate(restore):
            values = scanned[:, direction]
            if is_reverse:
                values = torch.flip(values, dims=[-1])
            restored.append(restore_sequence(values, inverse))
        combined = torch.stack(restored).sum(0)
        if count == 2:
            # Preserve the four-direction activation scale in normal/tangent
            # ablations while still executing only two selective scans.
            combined = combined * 2.0
        combined = combined.transpose(1, 2).contiguous().view(
            batch, height, width, self.d_inner
        )
        return self.out_norm(combined).to(x.dtype)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        projected, gate = self.in_proj(x).chunk(2, dim=-1)
        projected = projected.permute(0, 3, 1, 2).contiguous()
        projected = self.act(self.conv2d(projected))

        geometric = self._selective_scan_geometric(projected)
        if self.scan_mode == "hybrid":
            cartesian = super().forward_corev0(projected)
            scale = self.pde_gate.view(1, 1, 1, -1)
            output = cartesian + scale * geometric
        else:
            output = geometric
        output = output * F.silu(gate)
        output = self.out_proj(output)
        return self.dropout(output) if self.dropout is not None else output


def replace_layer_scans(layer: nn.Module, scan_mode: str) -> int:
    """Replace every official SS2D in one layer and return the block count."""
    replaced = 0
    for block in getattr(layer, "blocks", []):
        original = block.self_attention
        if isinstance(original, PDEGuidedSS2D):
            original.scan_mode = scan_mode
            continue
        if not isinstance(original, SS2D):
            raise TypeError(f"Expected official SS2D, got {type(original)}")
        block.self_attention = PDEGuidedSS2D.from_cartesian(
            original, scan_mode=scan_mode
        )
        replaced += 1
    return replaced


def set_layer_scan_cache(
    layer: nn.Module, cache: PDEPermutationCache | None
) -> None:
    for block in getattr(layer, "blocks", []):
        if isinstance(block.self_attention, PDEGuidedSS2D):
            block.self_attention.set_scan_cache(cache)
