"""Mamba-UNet with staged model-predicted PDE scan guidance."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor, nn

from .geometric_config import PDEGeometricConfig
from .official import VSSM, mamba_sys
from .pde_scan import PDEPermutationCache
from .pde_ss2d import (
    PDEGuidedSS2D,
    replace_layer_scans,
    set_layer_scan_cache,
)


class PDEGeometricMambaUNet(nn.Module):
    """Native Mamba-UNet plus NP/PDE/type heads and staged scan guidance."""

    feature_channels = 96

    def __init__(self, config: PDEGeometricConfig) -> None:
        super().__init__()
        self.scan_mode = config.scan_mode
        self.num_type_classes = len(config.type_classes)
        self.guided_stages = tuple(config.pde_scan_stages)
        self.guide_source_stage = int(config.guide_source_stage)
        self.guide_detach_for_sort = bool(config.guide_detach_for_sort)
        self.num_pde_bins = int(config.num_pde_bins)
        self.pde_scan_window = int(config.pde_scan_window)
        self.core = VSSM(
            patch_size=4,
            in_chans=3,
            num_classes=1,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
            use_checkpoint=False,
        )
        self.core.output = nn.Identity()

        self.guidance_enabled = self.scan_mode != "cartesian"
        if self.guidance_enabled:
            guide_channels = 96 * 2 ** (self.guide_source_stage + 1)
            self.guide_head: nn.Module | None = nn.Conv2d(
                guide_channels, 1, kernel_size=1, bias=True
            )
            nn.init.kaiming_normal_(
                self.guide_head.weight, mode="fan_out", nonlinearity="relu"
            )
            nn.init.zeros_(self.guide_head.bias)
            self._install_guided_scans()
        else:
            self.guide_head = None

        self.np_head = nn.Conv2d(self.feature_channels, 2, 1, bias=True)
        self.field_head = nn.Conv2d(self.feature_channels, 1, 1, bias=True)
        self.tp_head = nn.Conv2d(
            self.feature_channels, self.num_type_classes, 1, bias=True
        )
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        for head in (self.np_head, self.field_head, self.tp_head):
            nn.init.kaiming_normal_(
                head.weight, mode="fan_out", nonlinearity="relu"
            )
            nn.init.zeros_(head.bias)
        self._last_scan_cache: PDEPermutationCache | None = None
        self._last_guide: Tensor | None = None

    def _install_guided_scans(self) -> None:
        replaced = 0
        for stage in self.guided_stages:
            family, raw_index = stage.split("_")
            index = int(raw_index)
            layer = (
                self.core.layers[index]
                if family == "encoder" else self.core.layers_up[index]
            )
            replaced += replace_layer_scans(layer, self.scan_mode)
        if replaced == 0:
            raise ValueError("No SS2D blocks selected for PDE guidance")

    def _set_stage_cache(
        self, stage: str, cache: PDEPermutationCache | None
    ) -> None:
        if stage not in self.guided_stages:
            return
        family, raw_index = stage.split("_")
        index = int(raw_index)
        layer = (
            self.core.layers[index]
            if family == "encoder" else self.core.layers_up[index]
        )
        set_layer_scan_cache(layer, cache)

    def _make_cache(self, guide_logits: Tensor) -> PDEPermutationCache:
        guide = guide_logits.sigmoid()
        scan_guide = guide.detach() if self.guide_detach_for_sort else guide
        # Sorting is discrete regardless; the explicit detach makes that fact
        # unambiguous and prevents accidental straight-through assumptions.
        cache = PDEPermutationCache(
            scan_guide,
            num_pde_bins=self.num_pde_bins,
            window_size=self.pde_scan_window,
        )
        self._last_scan_cache = cache
        self._last_guide = guide.detach()
        return cache

    def decoder_features(
        self, images: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        if mamba_sys.selective_scan_fn is None:
            raise RuntimeError(
                "mamba-ssm selective_scan_fn is unavailable in Docker; "
                "run ./build.sh"
            )
        normalized = (images / 255.0 - self.image_mean) / self.image_std
        x = self.core.patch_embed(normalized)
        downsampled: list[Tensor] = []
        guide_logits: Tensor | None = None
        cache: PDEPermutationCache | None = None
        for index, layer in enumerate(self.core.layers):
            stage = f"encoder_{index}"
            downsampled.append(x)
            self._set_stage_cache(stage, cache)
            x = layer(x)
            if self.guidance_enabled and index == self.guide_source_stage:
                assert self.guide_head is not None
                guide_logits = self.guide_head(
                    x.permute(0, 3, 1, 2).contiguous()
                )
                cache = self._make_cache(guide_logits)
        x = self.core.norm(x)

        for index, layer_up in enumerate(self.core.layers_up):
            stage = f"decoder_{index}"
            self._set_stage_cache(stage, cache)
            if index == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, downsampled[3 - index]], dim=-1)
                x = self.core.concat_back_dim[index](x)
                x = layer_up(x)
        x = self.core.norm_up(x)
        feature = self.core.up_x4(x)
        return feature, guide_logits

    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        feature, guide = self.decoder_features(images)
        output = OrderedDict(
            (
                ("np", self.np_head(feature)),
                ("field", self.field_head(feature)),
                ("tp", self.tp_head(feature)),
            )
        )
        if guide is not None:
            output["guide"] = guide
        return output

    def parameter_groups(
        self,
    ) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        new_parameter_ids = {
            id(parameter)
            for module in (
                self.np_head, self.field_head, self.tp_head, self.guide_head
            )
            if module is not None
            for parameter in module.parameters()
        }
        for module in self.core.modules():
            if isinstance(module, PDEGuidedSS2D):
                new_parameter_ids.add(id(module.pde_gate))
        pretrained, new = [], []
        for parameter in self.parameters():
            (new if id(parameter) in new_parameter_ids else pretrained).append(
                parameter
            )
        if {id(item) for item in pretrained} & {id(item) for item in new}:
            raise RuntimeError("Pretrained and new parameter groups overlap")
        return pretrained, new

    def parameter_report(self) -> dict[str, int | str]:
        pretrained, new = self.parameter_groups()
        gates = [
            module.pde_gate
            for module in self.core.modules()
            if isinstance(module, PDEGuidedSS2D)
        ]
        count = lambda values: sum(item.numel() for item in values)
        return {
            "scan_mode": self.scan_mode,
            "total": count(self.parameters()),
            "trainable": count(
                item for item in self.parameters() if item.requires_grad
            ),
            "pretrained_group": count(pretrained),
            "new_group": count(new),
            "pde_gate_parameters": count(gates),
            "guided_ss2d_blocks": len(gates),
        }

    def last_scan_debug(self, height: int, width: int):
        if self._last_scan_cache is None:
            raise RuntimeError("No guided forward pass has been executed")
        return self._last_scan_cache.get(height, width)


@torch.inference_mode()
def verify_geometric_output_contract(
    model: PDEGeometricMambaUNet,
    device: torch.device,
    patch_size: int,
) -> dict[str, tuple[int, ...]]:
    model.eval()
    output = model(
        torch.zeros(1, 3, patch_size, patch_size, device=device)
    )
    expected = {
        "np": (1, 2, patch_size, patch_size),
        "field": (1, 1, patch_size, patch_size),
        "tp": (1, model.num_type_classes, patch_size, patch_size),
    }
    if model.guidance_enabled:
        side = patch_size // (8 * 2 ** model.guide_source_stage)
        expected["guide"] = (1, 1, side, side)
    actual = {key: tuple(value.shape) for key, value in output.items()}
    if actual != expected:
        raise RuntimeError(f"Unexpected output contract: {actual} != {expected}")
    if any(not torch.isfinite(value).all() for value in output.values()):
        raise FloatingPointError("Non-finite geometric model output")
    return actual
