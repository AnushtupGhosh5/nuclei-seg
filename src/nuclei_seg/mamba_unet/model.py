from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor, nn

from .official import VSSM, mamba_sys


class MambaUNetNP_HV_Type(nn.Module):
    """Official Mamba-UNet with only its final classifier adapted to NP/HV/type."""

    feature_channels = 96

    def __init__(self) -> None:
        super().__init__()
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
        # Keep the complete native encoder/decoder and final x4 patch expansion.
        # Only the original single-task output convolution is replaced.
        self.core.output = nn.Identity()
        self.np_head = nn.Conv2d(self.feature_channels, 2, 1, bias=True)
        self.hv_head = nn.Conv2d(self.feature_channels, 2, 1, bias=True)
        self.tp_head = nn.Conv2d(self.feature_channels, 4, 1, bias=True)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for head in (self.np_head, self.hv_head, self.tp_head):
            nn.init.kaiming_normal_(head.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(head.bias)

    def decoder_features(self, images: Tensor) -> Tensor:
        if mamba_sys.selective_scan_fn is None:
            raise RuntimeError(
                "mamba-ssm selective_scan_fn is unavailable in the Docker image; "
                "run ./build.sh and retry"
            )
        normalized = (images / 255.0 - self.image_mean) / self.image_std
        return self.core(normalized)

    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        feature = self.decoder_features(images)
        return OrderedDict(
            (("np", self.np_head(feature)), ("hv", self.hv_head(feature)), ("tp", self.tp_head(feature)))
        )

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        pretrained = list(self.core.parameters())
        heads = list(self.np_head.parameters()) + list(self.hv_head.parameters()) + list(self.tp_head.parameters())
        if {id(item) for item in pretrained} & {id(item) for item in heads}:
            raise RuntimeError("Core and head parameter groups overlap")
        return pretrained, heads

    def parameter_report(self) -> dict[str, int]:
        encoder = list(self.core.patch_embed.parameters()) + list(self.core.layers.parameters()) + list(self.core.norm.parameters())
        decoder = (
            list(self.core.layers_up.parameters())
            + list(self.core.concat_back_dim.parameters())
            + list(self.core.norm_up.parameters())
            + list(self.core.up.parameters())
        )
        heads = list(self.np_head.parameters()) + list(self.hv_head.parameters()) + list(self.tp_head.parameters())
        count = lambda values: sum(parameter.numel() for parameter in values)
        return {
            "total": count(self.parameters()),
            "trainable": count(parameter for parameter in self.parameters() if parameter.requires_grad),
            "encoder_and_bottleneck": count(encoder),
            "native_decoder": count(decoder),
            "task_heads": count(heads),
        }


def load_official_pretraining(
    model: MambaUNetNP_HV_Type, checkpoint_path: Path, output_dir: Path, dataset: str
) -> dict[str, Any]:
    """Apply the official direct + reversed encoder-to-decoder loading rule."""
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    target_state = model.core.state_dict()
    compatible: dict[str, Tensor] = {}
    loaded: list[dict[str, str]] = []
    unmatched: list[dict[str, str]] = []
    used_source: set[str] = set()

    def consider(source_key: str, target_key: str, route: str) -> None:
        tensor = source[source_key]
        if target_key not in target_state:
            unmatched.append(
                {"source": source_key, "target": target_key, "reason": "target_missing", "route": route}
            )
            return
        if tuple(tensor.shape) != tuple(target_state[target_key].shape):
            unmatched.append(
                {
                    "source": source_key,
                    "target": target_key,
                    "reason": f"shape {tuple(tensor.shape)} != {tuple(target_state[target_key].shape)}",
                    "route": route,
                }
            )
            return
        compatible[target_key] = tensor
        used_source.add(source_key)
        loaded.append(
            {"source": source_key, "target": target_key, "route": route, "shape": str(tuple(tensor.shape))}
        )

    for key in source:
        consider(key, key, "direct")
        match = re.match(r"^layers\.(\d+)(.*)$", key)
        if match:
            consider(key, f"layers_up.{3 - int(match.group(1))}{match.group(2)}", "encoder_to_decoder")

    result = model.core.load_state_dict(compatible, strict=False)
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "pretraining_dataset": dataset,
        "loaded": loaded,
        "unmatched_candidate_assignments": unmatched,
        "completely_unused_source": sorted(set(source) - used_source),
        "missing_target": list(result.missing_keys),
        "unexpected_target": list(result.unexpected_keys),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pretrained_load_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(loaded).to_csv(output_dir / "pretrained_tensors_loaded.csv", index=False)
    pd.DataFrame(unmatched).to_csv(output_dir / "pretrained_assignments_unmatched.csv", index=False)
    print(f"Loaded {len(loaded)} source-to-target assignments ({len(compatible)} unique tensors)")
    return report


@torch.inference_mode()
def verify_output_contract(model: MambaUNetNP_HV_Type, device: torch.device, patch_size: int) -> None:
    model.eval()
    probe = torch.zeros(1, 3, patch_size, patch_size, device=device)
    feature = model.decoder_features(probe)
    output = model(probe)
    expected = {
        "np": (1, 2, patch_size, patch_size),
        "hv": (1, 2, patch_size, patch_size),
        "tp": (1, 4, patch_size, patch_size),
    }
    if tuple(feature.shape) != (1, 96, patch_size, patch_size):
        raise RuntimeError(f"Unexpected native decoder feature shape: {tuple(feature.shape)}")
    actual = {key: tuple(value.shape) for key, value in output.items()}
    if actual != expected:
        raise RuntimeError(f"Unexpected output contract: {actual}; expected {expected}")
