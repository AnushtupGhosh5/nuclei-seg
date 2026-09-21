"""Minimal mask/PDE/type/guidance objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .geometric_config import PDEGeometricConfig
from .loss import dice_loss


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class PDEGeometricLoss:
    def __init__(
        self,
        binary_weights: Tensor,
        type_weights: Tensor,
        config: PDEGeometricConfig,
    ) -> None:
        self.binary_weights = binary_weights
        self.type_weights = type_weights
        self.config = config

    def _field_regression(
        self, prediction: Tensor, target: Tensor, foreground: Tensor
    ) -> Tensor:
        residual = F.smooth_l1_loss(
            prediction.sigmoid(), target, reduction="none", beta=0.1
        )
        foreground_loss = _masked_mean(residual, foreground)
        background_loss = _masked_mean(residual, 1.0 - foreground)
        return (
            foreground_loss
            + self.config.field_background_weight * background_loss
        )

    def __call__(
        self,
        output: dict[str, Tensor],
        instances: Tensor,
        types: Tensor,
        field_target: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        binary = (instances > 0).long()
        foreground = binary[:, None].float()
        mask_loss = F.cross_entropy(
            output["np"].float(), binary, weight=self.binary_weights
        ) + dice_loss(output["np"].float(), binary, 2)
        field_loss = self._field_regression(
            output["field"].float(), field_target[:, None].float(), foreground
        )
        type_loss = F.cross_entropy(
            output["tp"].float(), types, weight=self.type_weights
        ) + dice_loss(
            output["tp"].float(), types, len(self.config.type_classes)
        )

        if "guide" in output:
            guide_size = output["guide"].shape[-2:]
            guide_target = F.interpolate(
                field_target[:, None].float(), size=guide_size,
                mode="bilinear", align_corners=False,
            )
            guide_foreground = F.interpolate(
                foreground, size=guide_size, mode="nearest"
            )
            guide_loss = self._field_regression(
                output["guide"].float(), guide_target, guide_foreground
            )
        else:
            guide_loss = field_loss.new_zeros(())

        total = (
            self.config.mask_loss_weight * mask_loss
            + self.config.field_loss_weight * field_loss
            + self.config.type_loss_weight * type_loss
            + self.config.guide_loss_weight * guide_loss
        )
        return total, {
            "mask": float(mask_loss.detach()),
            "field": float(field_loss.detach()),
            "type": float(type_loss.detach()),
            "guide": float(guide_loss.detach()),
            "total": float(total.detach()),
        }

