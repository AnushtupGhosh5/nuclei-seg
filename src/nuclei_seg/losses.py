from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import NucleiOutput


def soft_dice_loss(logits: Tensor, target: Tensor, include_background: bool = True) -> Tensor:
    probabilities = logits.softmax(dim=1)
    valid = target >= 0
    safe_target = target.clamp_min(0)
    one_hot = F.one_hot(safe_target, logits.shape[1]).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1)
    probabilities = probabilities * valid
    one_hot = one_hot * valid
    if not include_background:
        probabilities = probabilities[:, 1:]
        one_hot = one_hot[:, 1:]
    dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dims)
    denominator = probabilities.sum(dims) + one_hot.sum(dims)
    # The reference HoVer-Net objective sums the per-channel Dice losses.
    return (1.0 - (2.0 * intersection + 1.0e-3) / (denominator + 1.0e-3)).sum()


def _hover_gradients(hv: Tensor) -> Tensor:
    """Return the two directional gradients used by official HoVer-Net MSGE."""
    coordinates = torch.arange(-2, 3, dtype=hv.dtype, device=hv.device)
    horizontal, vertical = torch.meshgrid(coordinates, coordinates, indexing="ij")
    denominator = horizontal.square() + vertical.square() + 1.0e-15
    kernel_h = (horizontal / denominator).view(1, 1, 5, 5)
    kernel_v = (vertical / denominator).view(1, 1, 5, 5)
    grad_h = F.conv2d(hv[:, 0:1], kernel_h, padding=2)
    grad_v = F.conv2d(hv[:, 1:2], kernel_v, padding=2)
    return torch.cat((grad_h, grad_v), dim=1)


@dataclass(frozen=True)
class LossWeights:
    np_ce: float = 1.0
    np_dice: float = 1.0
    hv_mse: float = 1.0
    hv_gradient: float = 1.0
    type_ce: float = 1.0
    type_dice: float = 1.0


class HoVerNetLoss(nn.Module):
    def __init__(self, weights: LossWeights = LossWeights()) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, output: NucleiOutput, target: dict[str, Tensor]) -> dict[str, Tensor]:
        np_target = target["np_map"].long()
        type_target = target["type_map"].long()
        hv_target = target["hv_map"].float()
        nucleus = np_target.bool().unsqueeze(1)

        np_ce = F.cross_entropy(output.np_logits, np_target)
        np_dice = soft_dice_loss(output.np_logits, np_target)

        # Match the reference losses: MSE over the full map and MSGE only in
        # nuclear pixels. Crucially, MSGE supervises one direction per channel.
        hv_mse = F.mse_loss(output.hv_map, hv_target)
        focus = nucleus.expand(-1, 2, -1, -1).float()
        gradient_error = (_hover_gradients(output.hv_map) - _hover_gradients(hv_target)).square()
        hv_gradient = (gradient_error * focus).sum() / focus.sum().clamp_min(1.0)

        # Class-aware crop sampling already exposes rare classes more often.
        # Applying another 4x CE weight here caused the type branch to overfit
        # and destabilized validation through the shared U-Net features.
        type_ce = F.cross_entropy(output.type_logits, type_target, ignore_index=-1)
        type_dice = soft_dice_loss(output.type_logits, type_target)

        parts = {
            "np_ce": np_ce,
            "np_dice": np_dice,
            "hv_mse": hv_mse,
            "hv_gradient": hv_gradient,
            "type_ce": type_ce,
            "type_dice": type_dice,
        }
        total = sum(getattr(self.weights, name) * value for name, value in parts.items())
        return {"loss": total, **parts}
