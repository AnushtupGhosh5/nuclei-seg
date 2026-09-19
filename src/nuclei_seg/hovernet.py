"""HoVer-Net fast-mode architecture.

Adapted from the official MIT-licensed implementation at
https://github.com/vqdang/hover_net. Module names intentionally match the
reference checkpoint so released weights can be loaded directly.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import NucleiOutput


def _crop(x: Tensor, amount: tuple[int, int]) -> Tensor:
    top, left = amount[0] // 2, amount[1] // 2
    bottom, right = amount[0] - top, amount[1] - left
    return x[:, :, top : x.shape[2] - bottom, left : x.shape[3] - right]


def _crop_like(x: Tensor, target: Tensor) -> Tensor:
    return _crop(x, (x.shape[2] - target.shape[2], x.shape[3] - target.shape[3]))


class TFSamepaddingLayer(nn.Module):
    def __init__(self, ksize: int, stride: int) -> None:
        super().__init__()
        self.ksize, self.stride = ksize, stride

    def forward(self, x: Tensor) -> Tensor:
        remainder = x.shape[2] % self.stride
        pad = max(self.ksize - (self.stride if remainder == 0 else remainder), 0)
        start, end = pad // 2, pad - pad // 2
        return F.pad(x, (start, end, start, end))


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        unit_ksize: list[int],
        unit_ch: list[int],
        unit_count: int,
        split: int = 1,
    ) -> None:
        super().__init__()
        self.units = nn.ModuleList()
        unit_in = in_ch
        for _ in range(unit_count):
            self.units.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("preact_bna/bn", nn.BatchNorm2d(unit_in, eps=1e-5)),
                            ("preact_bna/relu", nn.ReLU(inplace=True)),
                            ("conv1", nn.Conv2d(unit_in, unit_ch[0], unit_ksize[0], bias=False)),
                            ("conv1/bn", nn.BatchNorm2d(unit_ch[0], eps=1e-5)),
                            ("conv1/relu", nn.ReLU(inplace=True)),
                            (
                                "conv2",
                                nn.Conv2d(
                                    unit_ch[0],
                                    unit_ch[1],
                                    unit_ksize[1],
                                    groups=split,
                                    bias=False,
                                ),
                            ),
                        ]
                    )
                )
            )
            unit_in += unit_ch[1]
        self.blk_bna = nn.Sequential(
            OrderedDict(
                [("bn", nn.BatchNorm2d(unit_in, eps=1e-5)), ("relu", nn.ReLU(inplace=True))]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        for unit in self.units:
            new = unit(x)
            x = torch.cat((_crop_like(x, new), new), dim=1)
        return self.blk_bna(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        unit_ksize: list[int],
        unit_ch: list[int],
        unit_count: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.units = nn.ModuleList()
        unit_in = in_ch
        for index in range(unit_count):
            unit = [
                ("preact/bn", nn.BatchNorm2d(unit_in, eps=1e-5)),
                ("preact/relu", nn.ReLU(inplace=True)),
                ("conv1", nn.Conv2d(unit_in, unit_ch[0], unit_ksize[0], bias=False)),
                ("conv1/bn", nn.BatchNorm2d(unit_ch[0], eps=1e-5)),
                ("conv1/relu", nn.ReLU(inplace=True)),
                (
                    "conv2/pad",
                    TFSamepaddingLayer(unit_ksize[1], stride if index == 0 else 1),
                ),
                (
                    "conv2",
                    nn.Conv2d(
                        unit_ch[0],
                        unit_ch[1],
                        unit_ksize[1],
                        stride=stride if index == 0 else 1,
                        bias=False,
                    ),
                ),
                ("conv2/bn", nn.BatchNorm2d(unit_ch[1], eps=1e-5)),
                ("conv2/relu", nn.ReLU(inplace=True)),
                ("conv3", nn.Conv2d(unit_ch[1], unit_ch[2], unit_ksize[2], bias=False)),
            ]
            self.units.append(nn.Sequential(OrderedDict(unit if index else unit[2:])))
            unit_in = unit_ch[-1]
        self.shortcut = (
            nn.Conv2d(in_ch, unit_ch[-1], 1, stride=stride, bias=False)
            if in_ch != unit_ch[-1] or stride != 1
            else None
        )
        self.blk_bna = nn.Sequential(
            OrderedDict(
                [("bn", nn.BatchNorm2d(unit_in, eps=1e-5)), ("relu", nn.ReLU(inplace=True))]
            )
        )

    def forward(self, x: Tensor, freeze: bool = False) -> Tensor:
        shortcut = x if self.shortcut is None else self.shortcut(x)
        for unit in self.units:
            with torch.set_grad_enabled(torch.is_grad_enabled() and not freeze):
                new = unit(x)
            x = new + shortcut
            shortcut = x
        return self.blk_bna(x)


class UpSample2x(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Kept for state-dict compatibility with the official implementation.
        self.register_buffer("unpool_mat", torch.ones((2, 2), dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        expanded = torch.tensordot(x.unsqueeze(-1), self.unpool_mat.unsqueeze(0), dims=1)
        return expanded.permute(0, 1, 2, 4, 3, 5).reshape(
            shape[0], shape[1], shape[2] * 2, shape[3] * 2
        )


class HoVerNetFast(nn.Module):
    """Reference-compatible 256 input -> 164 output HoVer-Net."""

    def __init__(self, num_types: int = 5, freeze_encoder: bool = False) -> None:
        super().__init__()
        self.num_types = num_types
        self.freeze = freeze_encoder
        self.conv0 = nn.Sequential(
            OrderedDict(
                [
                    ("pad", TFSamepaddingLayer(7, 1)),
                    ("/", nn.Conv2d(3, 64, 7, bias=False)),
                    ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3)
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 4, stride=2)
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 6, stride=2)
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2)
        self.conv_bot = nn.Conv2d(2048, 1024, 1, bias=False)
        self.decoder = nn.ModuleDict(
            OrderedDict(
                [
                    ("tp", self._decoder(num_types)),
                    ("np", self._decoder(2)),
                    ("hv", self._decoder(2)),
                ]
            )
        )
        self.upsample2x = UpSample2x()
        self._initialize()

    @staticmethod
    def _decoder(out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        "u3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("conva", nn.Conv2d(1024, 256, 3, bias=False)),
                                    ("dense", DenseBlock(256, [1, 3], [128, 32], 8, split=4)),
                                    ("convf", nn.Conv2d(512, 512, 1, bias=False)),
                                ]
                            )
                        ),
                    ),
                    (
                        "u2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("conva", nn.Conv2d(512, 128, 3, bias=False)),
                                    ("dense", DenseBlock(128, [1, 3], [128, 32], 4, split=4)),
                                    ("convf", nn.Conv2d(256, 256, 1, bias=False)),
                                ]
                            )
                        ),
                    ),
                    (
                        "u1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("conva/pad", TFSamepaddingLayer(3, 1)),
                                    ("conva", nn.Conv2d(256, 64, 3, bias=False)),
                                ]
                            )
                        ),
                    ),
                    (
                        "u0",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                                    ("relu", nn.ReLU(inplace=True)),
                                    ("conv", nn.Conv2d(64, out_channels, 1)),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, image: Tensor) -> NucleiOutput:
        d0 = self.conv0(image)
        d0 = self.d0(d0, self.freeze)
        d1 = self.d1(d0, self.freeze)
        d2 = self.d2(d1, self.freeze)
        d3 = self.conv_bot(self.d3(d2, self.freeze))
        d0, d1 = _crop(d0, (92, 92)), _crop(d1, (36, 36))
        outputs: dict[str, Tensor] = {}
        for name, branch in self.decoder.items():
            u3 = branch.u3(self.upsample2x(d3) + d2)
            u2 = branch.u2(self.upsample2x(u3) + d1)
            u1 = branch.u1(self.upsample2x(u2) + d0)
            outputs[name] = branch.u0(u1)
        return NucleiOutput(
            np_logits=outputs["np"], hv_map=outputs["hv"], type_logits=outputs["tp"]
        )
