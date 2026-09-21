from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor, nn

from .official import VSSM, mamba_sys


class MambaUNetPoissonType(nn.Module):
    """Mamba-UNet decoder with a scalar Poisson field head plus type head."""

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
        self.core.output = nn.Identity()
        self.field_head = nn.Conv2d(self.feature_channels, 1, 1, bias=True)
        self.tp_head = nn.Conv2d(self.feature_channels, 4, 1, bias=True)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        for head in (self.field_head, self.tp_head):
            nn.init.kaiming_normal_(head.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(head.bias)

    def decoder_features(self, images: Tensor) -> Tensor:
        if mamba_sys.selective_scan_fn is None:
            raise RuntimeError("mamba-ssm selective_scan_fn is unavailable")
        normalized = (images / 255.0 - self.image_mean) / self.image_std
        return self.core(normalized)
    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        feature = self.decoder_features(images)
        return OrderedDict(
            (
                ("field", self.field_head(feature)),
                ("tp", self.tp_head(feature)),
            )
        )

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        pretrained = list(self.core.parameters())
        heads = list(self.field_head.parameters()) + list(self.tp_head.parameters())
        if {id(x) for x in pretrained} & {id(x) for x in heads}:
            raise RuntimeError("Core and head parameter groups overlap")
        return pretrained, heads

    def parameter_report(self) -> dict[str, int]:
        count = lambda values: sum(parameter.numel() for parameter in values)
        return {
            "total": count(self.parameters()),
            "trainable": count(
                parameter for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "task_heads": count(
                list(self.field_head.parameters()) + list(self.tp_head.parameters())
            ),
        }
