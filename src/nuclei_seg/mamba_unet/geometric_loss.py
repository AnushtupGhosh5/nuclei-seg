"""Minimal mask/PDE/type/guidance objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .geometric_config import PDEGeometricConfig
from .loss import dice_loss


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def foreground_type_weights(counts, device: torch.device, foreground_only: bool) -> Tensor:
    """Inverse sqrt counts, mean one over foreground classes present in training.

    Absent classes get neutral weight one rather than an unbounded inverse
    frequency. Background is excluded from the normalization.
    """
    foreground = torch.as_tensor(counts[1:], dtype=torch.float32, device=device)
    present = foreground > 0
    weights = torch.ones(len(counts), dtype=torch.float32, device=device)
    if present.any():
        raw = foreground[present].rsqrt()
        weights[1:][present] = raw / raw.mean()
    weights[0] = 0.0 if foreground_only else 1.0
    return weights


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
    ) -> tuple[Tensor, Tensor, Tensor]:
        residual = F.smooth_l1_loss(
            prediction.sigmoid(), target, reduction="none", beta=0.1
        )
        foreground_loss = _masked_mean(residual, foreground)
        background_loss = _masked_mean(residual, 1.0 - foreground)
        return (
            foreground_loss
            + self.config.field_background_weight * background_loss,
            foreground_loss, background_loss,
        )

    def _type_losses(self, logits: Tensor, types: Tensor, binary: Tensor):
        if not self.config.type_loss_foreground_only:
            return (
                F.cross_entropy(logits, types, weight=self.type_weights),
                dice_loss(logits, types, len(self.config.type_classes)),
            )
        # Untyped foreground pixels are excluded as well: class zero is never
        # a target. All channels remain in the softmax denominator.
        valid = (binary > 0) & (types > 0)
        if not valid.any():
            zero = logits.sum() * 0.0
            return zero, zero
        selected_logits = logits.permute(0, 2, 3, 1)[valid]
        selected_types = types[valid]
        ce = F.cross_entropy(selected_logits, selected_types, weight=self.type_weights)
        probabilities = selected_logits.softmax(-1)[:, 1:]
        targets = F.one_hot(selected_types, logits.shape[1]).float()[:, 1:]
        present = targets.sum(0) > 0
        score = (2 * (probabilities * targets).sum(0) + 1.0) / (
            probabilities.sum(0) + targets.sum(0) + 1.0
        )
        return ce, 1.0 - score[present].mean()

    def __call__(
        self,
        output: dict[str, Tensor],
        instances: Tensor,
        types: Tensor,
        field_target: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        binary = (instances > 0).long()
        foreground = binary[:, None].float()
        mask_ce = F.cross_entropy(
            output["np"].float(), binary, weight=self.binary_weights
        )
        mask_dice = dice_loss(output["np"].float(), binary, 2)
        mask_loss = mask_ce + mask_dice
        field_loss, field_foreground, field_background = self._field_regression(
            output["field"].float(), field_target[:, None].float(), foreground
        )
        type_ce, type_dice = self._type_losses(
            output["tp"].float(), types, binary
        )
        type_loss = type_ce + type_dice

        if "guide" in output:
            guide_size = output["guide"].shape[-2:]
            guide_target = F.interpolate(
                field_target[:, None].float(), size=guide_size,
                mode="bilinear", align_corners=False,
            )
            guide_foreground = F.interpolate(
                foreground, size=guide_size, mode="nearest"
            )
            guide_loss, guide_foreground_loss, guide_background_loss = self._field_regression(
                output["guide"].float(), guide_target, guide_foreground
            )
        else:
            guide_loss = field_loss.new_zeros(())
            guide_foreground_loss = guide_background_loss = guide_loss

        total = (
            self.config.mask_loss_weight * mask_loss
            + self.config.field_loss_weight * field_loss
            + self.config.type_loss_weight * type_loss
            + self.config.guide_loss_weight * guide_loss
        )
        return total, {
            "mask_ce": float(mask_ce.detach()),
            "mask_dice": float(mask_dice.detach()),
            "field_foreground": float(field_foreground.detach()),
            "field_background": float(field_background.detach()),
            "type_ce": float(type_ce.detach()),
            "type_dice": float(type_dice.detach()),
            "guide_foreground": float(guide_foreground_loss.detach()),
            "guide_background": float(guide_background_loss.detach()),
            "mask": float(mask_loss.detach()),
            "field": float(field_loss.detach()),
            "type": float(type_loss.detach()),
            "guide": float(guide_loss.detach()),
            "total": float(total.detach()),
        }
