from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class DoubleConv(nn.Module):
    """The original U-Net's two unpadded 3x3 convolutions and ReLUs."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.block(image)


class UNetEncoder(nn.Module):
    """Contracting path from Ronneberger et al. (2015)."""

    def __init__(self, channels: tuple[int, ...] = (64, 128, 256, 512, 1024)) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("U-Net requires at least two resolution levels")
        self.blocks = nn.ModuleList()
        in_channels = 3
        for out_channels in channels:
            self.blocks.append(DoubleConv(in_channels, out_channels))
            in_channels = out_channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, image: Tensor) -> list[Tensor]:
        features = []
        for index, block in enumerate(self.blocks):
            image = block(image)
            features.append(image)
            if index < len(self.blocks) - 1:
                image = self.pool(image)
        return features


def _center_crop(feature: Tensor, height: int, width: int) -> Tensor:
    difference_y = feature.shape[-2] - height
    difference_x = feature.shape[-1] - width
    if difference_y < 0 or difference_x < 0:
        raise ValueError("Skip feature is smaller than the decoder feature")
    top = difference_y // 2
    left = difference_x // 2
    return feature[..., top : top + height, left : left + width]


class UpBlock(nn.Module):
    """Original 2x2 up-convolution, cropped skip connection, and double conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = DoubleConv(out_channels * 2, out_channels)

    def forward(self, feature: Tensor, skip: Tensor) -> Tensor:
        feature = self.up(feature)
        skip = _center_crop(skip, feature.shape[-2], feature.shape[-1])
        return self.fuse(torch.cat((skip, feature), dim=1))


class UNetDecoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        reversed_channels = tuple(reversed(channels))
        self.up_blocks = nn.ModuleList(
            UpBlock(reversed_channels[index], reversed_channels[index + 1])
            for index in range(len(reversed_channels) - 1)
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        feature = features[-1]
        for block, skip in zip(self.up_blocks, reversed(features[:-1])):
            feature = block(feature, skip)
        return feature


@dataclass(frozen=True)
class NucleiOutput:
    np_logits: Tensor
    hv_map: Tensor
    type_logits: Tensor


class OriginalUNet(nn.Module):
    """Canonical 2015 U-Net adapted only at its final task output layer.

    The original shared 64-channel decoder feature feeds the three outputs
    required by this study. Replicating the complete decoder three times would
    no longer be the approximately 31M-parameter original U-Net baseline.
    """

    def __init__(
        self,
        num_types: int,
        encoder_channels: tuple[int, ...] = (64, 128, 256, 512, 1024),
        bound_hv: bool = False,
    ) -> None:
        super().__init__()
        if num_types < 2:
            raise ValueError("num_types includes background and must be at least 2")
        self.num_types = num_types
        self.bound_hv = bound_hv
        self.encoder = UNetEncoder(encoder_channels)
        self.decoder = UNetDecoder(encoder_channels)
        final_channels = encoder_channels[0]
        self.np_head = nn.Conv2d(final_channels, 2, kernel_size=1)
        self.hv_head = nn.Conv2d(final_channels, 2, kernel_size=1)
        self.type_head = nn.Conv2d(final_channels, num_types, kernel_size=1)

    def forward(self, image: Tensor) -> NucleiOutput:
        feature = self.decoder(self.encoder(image))
        hv_map = self.hv_head(feature)
        if self.bound_hv:
            hv_map = hv_map.tanh()
        return NucleiOutput(
            np_logits=self.np_head(feature),
            hv_map=hv_map,
            type_logits=self.type_head(feature),
        )


# Retain the old import name for callers while making the implemented model
# unambiguously the original U-Net baseline.
MultiBranchUNet = OriginalUNet
