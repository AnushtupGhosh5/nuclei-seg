from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MambaUNetConfig:
    """The executed Kaggle experiment expressed as an explicit configuration."""

    data_root: Path = Path("data/glysac_dataset")
    output_dir: Path = Path("outputs/mamba_unet_glysac")
    pretrained_checkpoint: Path = Path("outputs/pretrained/vmamba_tiny_e292.pth")
    split_csv: Path | None = None
    type_encoding: str = "auto"
    type_classes: tuple[str, ...] = ("background", "other", "lymphocyte", "epithelial")
    strict_official_split: bool = True
    seed: int = 42
    patch_size: int = 256
    infer_overlap: int = 64
    batch_size: int = 2
    infer_batch_size: int = 2
    patches_per_tile: int = 12
    epochs: int = 200
    learning_rate: float = 1.0e-4
    pretrained_learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    num_workers: int = 2
    val_fraction: float = 0.15
    eval_every: int = 5
    magnification: int = 40
    match_iou: float = 0.5
    min_foreground_fraction: float = 0.01
    early_stopping_patience: int | None = 4
    amp: bool = False
    deterministic: bool = True
    smoke_test: bool = False
    resume_checkpoint: Path | None = None

    checkpoint_url: str = (
        "https://github.com/MzeroMiko/VMamba/releases/download/"
        "%23v0cls/vssmtiny_dp01_ckpt_epoch_292.pth"
    )
    checkpoint_sha256: str = (
        "dbc0cc4f5ec0e45db5fba7c939d2c7d9b617e891ac10766912a8d604c37c5e47"
    )
    checkpoint_dataset: str = "ImageNet-1K classification, 224x224, VMamba-T epoch 292"
    upstream_commit: str = "2eeec299581934e05b2af0322cc3107e2605867a"

    @classmethod
    def from_json(cls, path: Path) -> "MambaUNetConfig":
        values: dict[str, Any] = json.loads(path.read_text())
        for name in ("data_root", "output_dir", "pretrained_checkpoint", "split_csv", "resume_checkpoint"):
            if values.get(name) is not None:
                values[name] = Path(values[name])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}

    def validate(self) -> None:
        if self.seed != 42:
            raise ValueError("The reference reproduction requires seed=42")
        if self.patch_size != 256:
            raise ValueError("The reference Mamba-UNet experiment requires 256-pixel patches")
        if self.type_encoding not in {"auto", "original_10_class", "already_merged_3_class", "monusac_4class"}:
            raise ValueError(f"Unsupported type encoding: {self.type_encoding}")
        if len(self.type_classes) < 2 or self.type_classes[0].lower() != "background":
            raise ValueError("type_classes must start with background and include foreground classes")
        if self.infer_overlap >= self.patch_size:
            raise ValueError("infer_overlap must be smaller than patch_size")
        if self.batch_size < 1 or self.infer_batch_size < 1:
            raise ValueError("Batch sizes must be positive")

