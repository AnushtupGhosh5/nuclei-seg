"""PDE-target dataset adapters shared by geometric experiments."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .data import RandomPatchDataset
from .pde_field import make_poisson_field


class PDETargetDataset(Dataset):
    """Reuse the exact GLySAC crop/augmentation pipeline, replacing HV by PDE."""

    def __init__(
        self, base: RandomPatchDataset, *, iterations: int = 48
    ) -> None:
        self.base = base
        self.iterations = int(iterations)

    @property
    def rng(self):
        return self.base.rng

    @rng.setter
    def rng(self, value):
        self.base.rng = value

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, instances, types, _, stem = self.base[index]
        field = make_poisson_field(
            instances.numpy(), iterations=self.iterations
        )
        if not torch.from_numpy(field).isfinite().all():
            raise FloatingPointError(f"Non-finite PDE target in {stem}")
        return image, instances, types, torch.from_numpy(field), stem

