from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
SOBEL_Y = SOBEL_X.transpose(-1, -2)


def dice_loss(logits: Tensor, target: Tensor, classes: int) -> Tensor:
    probabilities = logits.softmax(1)
    one_hot = F.one_hot(target, classes).permute(0, 3, 1, 2).float()
    dimensions = (0, 2, 3)
    score = (2 * (probabilities * one_hot).sum(dimensions) + 1.0) / (
        (probabilities + one_hot).sum(dimensions) + 1.0
    )
    return 1 - score[1:].mean()


def class_weights(counts: np.ndarray, device: torch.device) -> Tensor:
    frequency = counts / counts.sum()
    weights = 1 / np.sqrt(np.maximum(frequency, 1.0e-12))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)


class SmileStyleLoss:
    def __init__(self, binary_weights: Tensor, type_weights: Tensor) -> None:
        self.binary_weights = binary_weights
        self.type_weights = type_weights

    def __call__(
        self, output: dict[str, Tensor], instances: Tensor, types: Tensor, hv_target: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        np_logits = output["np"].float()
        type_logits = output["tp"].float()
        hv_output = output["hv"].float()
        binary = (instances > 0).long()
        binary_loss = F.cross_entropy(np_logits, binary, weight=self.binary_weights) + dice_loss(
            np_logits, binary, 2
        )
        type_loss = F.cross_entropy(type_logits, types, weight=self.type_weights) + dice_loss(
            type_logits, types, 4
        )
        mask = binary[:, None].float()
        denominator = mask.sum().clamp_min(1) * 2
        hv_loss = F.mse_loss(hv_output * mask, hv_target.float() * mask, reduction="sum") / denominator
        sobel_x, sobel_y = SOBEL_X.to(hv_target), SOBEL_Y.to(hv_target)
        predicted_gradient = torch.cat(
            [
                F.conv2d(hv_output[:, 0:1], sobel_x, padding=1),
                F.conv2d(hv_output[:, 1:2], sobel_y, padding=1),
            ],
            1,
        )
        true_gradient = torch.cat(
            [
                F.conv2d(hv_target[:, 0:1], sobel_x, padding=1),
                F.conv2d(hv_target[:, 1:2], sobel_y, padding=1),
            ],
            1,
        )
        gradient_loss = F.l1_loss(
            predicted_gradient * mask, true_gradient * mask, reduction="sum"
        ) / denominator
        total = binary_loss + type_loss + hv_loss + gradient_loss
        parts = {
            "binary": binary_loss.item(),
            "type": type_loss.item(),
            "hv": hv_loss.item(),
            "gradient": gradient_loss.item(),
        }
        return total, parts

