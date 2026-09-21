from __future__ import annotations

import hashlib
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from scipy.ndimage import center_of_mass
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import MambaUNetConfig


IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
CLASS_NAMES = {0: "background", 1: "other", 2: "lymphocyte", 3: "epithelial"}


@dataclass(frozen=True, slots=True)
class Record:
    stem: str
    image: Path
    label: Path
    split: str


@dataclass(frozen=True, slots=True)
class DataSplit:
    train: list[Record]
    val: list[Record]
    test: list[Record]
    identity_sha256: str


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _mat_has_maps(path: Path) -> bool:
    try:
        names = {item[0] for item in sio.whosmat(path)}
        return {"inst_map", "type_map"}.issubset(names)
    except Exception:
        return False


def _infer_split(path: Path) -> str:
    parts = [part.lower() for part in path.parts[-5:]]
    if any(re.search(r"(^|[^a-z])test", part) for part in parts):
        return "test"
    if any(re.search(r"(^|[^a-z])val(id|idation)?", part) for part in parts):
        return "val"
    if any(re.search(r"(^|[^a-z])train", part) for part in parts):
        return "train"
    return ""


def discover_records(root: Path, split_csv: Path | None = None) -> list[Record]:
    images: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images[path.stem.lower()].append(path)

    split_lookup: dict[str, str] = {}
    if split_csv is not None:
        frame = pd.read_csv(split_csv)
        if not {"stem", "split"}.issubset(frame.columns):
            raise ValueError("split_csv must contain stem and split columns")
        split_lookup = dict(
            zip(frame.stem.astype(str).str.lower(), frame.split.astype(str).str.lower())
        )

    records: list[Record] = []
    for label in sorted(root.rglob("*.mat")):
        if not _mat_has_maps(label):
            continue
        candidates = images.get(label.stem.lower(), [])
        if not candidates:
            continue
        image = max(
            candidates,
            key=lambda candidate: len(os.path.commonpath([candidate, label]).split(os.sep)),
        )
        split = split_lookup.get(label.stem.lower()) or _infer_split(label) or _infer_split(image)
        records.append(Record(label.stem, image, label, split))

    if not records:
        raise RuntimeError(f"No GLySAC image/annotation pairs found below {root}")
    unknown = [record for record in records if not record.split]
    if unknown:
        examples = "\n".join(f"  {item.image} | {item.label}" for item in unknown[:8])
        raise RuntimeError(f"Cannot determine split for {len(unknown)} pairs:\n{examples}")
    invalid = sorted({record.split for record in records} - {"train", "val", "test"})
    if invalid:
        raise ValueError(f"Unknown split names: {invalid}")
    return records


def detect_type_encoding(records: list[Record], requested: str) -> str:
    ids: set[int] = set()
    for record in tqdm(records, desc="Checking label encoding"):
        values = sio.loadmat(record.label, variable_names=["type_map"])["type_map"]
        ids.update(int(value) for value in np.unique(values))
    if requested == "monusac_4class":
        unknown = ids - set(range(5))
        if unknown:
            raise ValueError(f"Unexpected MoNuSAC type IDs: {sorted(unknown)}")
        print("Detected type encoding:", requested, "| IDs:", sorted(ids))
        return requested
    unknown = ids - set(range(11))
    if unknown:
        raise ValueError(f"Unexpected GLySAC type IDs: {sorted(unknown)}")
    if requested == "auto":
        requested = "original_10_class" if any(value > 3 for value in ids) else "already_merged_3_class"
    print("Detected type encoding:", requested, "| IDs:", sorted(ids))
    return requested


def merge_glysac_types(type_map: np.ndarray, encoding: str) -> np.ndarray:
    values = np.asarray(type_map, dtype=np.int32)
    if encoding in {"already_merged_3_class", "monusac_4class"}:
        return values.astype(np.uint8)
    output = np.zeros_like(values, dtype=np.uint8)
    for old in (1, 2, 9, 10):
        output[values == old] = 1
    for old in (4, 5, 6, 7):
        output[values == old] = 2
    for old in (3, 8):
        output[values == old] = 3
    return output


def read_annotation(path: Path, encoding: str) -> tuple[np.ndarray, np.ndarray]:
    annotation = sio.loadmat(path, variable_names=["inst_map", "type_map"])
    instances = np.asarray(annotation["inst_map"], dtype=np.int32).squeeze()
    types = merge_glysac_types(np.asarray(annotation["type_map"]).squeeze(), encoding)
    if instances.shape != types.shape:
        raise ValueError(f"Shape mismatch in {path}: {instances.shape} vs {types.shape}")
    types = types.copy()
    types[instances == 0] = 0
    return instances, types


def read_ignore_map(path: Path) -> np.ndarray:
    """Return optional evaluation-only ignore mask stored in a MAT label."""
    names = {item[0] for item in sio.whosmat(path)}
    if "ignore_map" not in names:
        shape = np.asarray(
            sio.loadmat(path, variable_names=["inst_map"])["inst_map"]
        ).squeeze().shape
        return np.zeros(shape, dtype=bool)
    ignore = np.asarray(
        sio.loadmat(path, variable_names=["ignore_map"])["ignore_map"]
    ).squeeze()
    return ignore.astype(bool)


def build_split(records: list[Record], config: MambaUNetConfig, output_dir: Path) -> DataSplit:
    train = [record for record in records if record.split == "train"]
    val = [record for record in records if record.split == "val"]
    test = [record for record in records if record.split == "test"]
    if not train or not test:
        raise RuntimeError("Both official train and test records are required")
    if config.strict_official_split and not val and (len(train), len(test)) != (34, 25):
        raise RuntimeError(f"Expected the official 34/25 split, found {len(train)}/{len(test)}")
    if not val and config.val_fraction > 0:
        order = np.random.default_rng(config.seed).permutation(len(train))
        number_val = max(1, round(len(train) * config.val_fraction))
        val_indices = set(order[:number_val].tolist())
        val = [record for index, record in enumerate(train) if index in val_indices]
        train = [record for index, record in enumerate(train) if index not in val_indices]
    if not val:
        raise RuntimeError("Validation records are required for best-checkpoint selection")

    if config.smoke_test:
        train, val, test = train[:2], val[:1], test[:1]

    rows = [
        {"stem": record.stem, "split": split, "image": str(record.image), "label": str(record.label)}
        for split, subset in (("train", train), ("val", val), ("test", test))
        for record in subset
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "split_manifest.csv", index=False)
    identity = "\n".join(f"{row['split']}:{row['stem']}" for row in rows)
    digest = hashlib.sha256(identity.encode()).hexdigest()
    if {item.stem for item in train} & {item.stem for item in test}:
        raise RuntimeError("Train/test leakage detected")
    return DataSplit(train, val, test, digest)


def _bounding_box(mask: np.ndarray) -> list[int] | None:
    rows, columns = np.where(mask)
    if not len(rows):
        return None
    return [rows.min(), rows.max() + 1, columns.min(), columns.max() + 1]


def make_hv_map(instance_map: np.ndarray) -> np.ndarray:
    """Exact NP/HV target generation used by the executed notebook."""
    horizontal = np.zeros(instance_map.shape, np.float32)
    vertical = np.zeros(instance_map.shape, np.float32)
    for instance_id in np.unique(instance_map):
        if instance_id == 0:
            continue
        full_mask = instance_map == instance_id
        box = _bounding_box(full_mask)
        if box is None:
            continue
        row0, row1, column0, column1 = box
        row0, column0 = max(0, row0 - 2), max(0, column0 - 2)
        row1 = min(instance_map.shape[0], row1 + 2)
        column1 = min(instance_map.shape[1], column1 + 2)
        mask = full_mask[row0:row1, column0:column1]
        if min(mask.shape) < 2:
            continue
        center_y, center_x = center_of_mass(mask)
        yy, xx = np.meshgrid(
            np.arange(mask.shape[0]) - round(center_y),
            np.arange(mask.shape[1]) - round(center_x),
            indexing="ij",
        )
        xx, yy = xx.astype(np.float32), yy.astype(np.float32)
        xx[~mask] = 0
        yy[~mask] = 0
        negative, positive = xx < 0, xx > 0
        if negative.any():
            xx[negative] /= -xx[negative].min()
        if positive.any():
            xx[positive] /= xx[positive].max()
        negative, positive = yy < 0, yy > 0
        if negative.any():
            yy[negative] /= -yy[negative].min()
        if positive.any():
            yy[positive] /= yy[positive].max()
        horizontal[row0:row1, column0:column1][mask] = xx[mask]
        vertical[row0:row1, column0:column1][mask] = yy[mask]
    return np.stack([horizontal, vertical])


def _pad_to_minimum(
    image: np.ndarray, instances: np.ndarray, types: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pad_h, pad_w = max(0, size - image.shape[0]), max(0, size - image.shape[1])
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        instances = np.pad(instances, ((0, pad_h), (0, pad_w)), mode="constant")
        types = np.pad(types, ((0, pad_h), (0, pad_w)), mode="constant")
    return image, instances, types


class RandomPatchDataset(Dataset):
    def __init__(
        self,
        records: list[Record],
        patch_size: int,
        patches_per_tile: int,
        seed: int,
        encoding: str,
        min_foreground_fraction: float,
        augment: bool,
    ) -> None:
        self.records = list(records)
        self.size = int(patch_size)
        self.length = len(records) * int(patches_per_tile)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.min_foreground_fraction = float(min_foreground_fraction)
        self.rng = np.random.default_rng(seed)
        self.tiles: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
        for record in tqdm(self.records, desc="Caching train tiles" if augment else "Caching val tiles"):
            image = read_rgb(record.image)
            instances, types = read_annotation(record.label, encoding)
            image, instances, types = _pad_to_minimum(image, instances, types, self.size)
            self.tiles.append((image, instances, types, record.stem))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        image, instances_full, types_full, stem = self.tiles[index % len(self.tiles)]
        height, width = instances_full.shape
        rng = self.rng if self.augment else np.random.default_rng(self.seed + index)
        for _ in range(12):
            y = rng.integers(0, height - self.size + 1)
            x = rng.integers(0, width - self.size + 1)
            instances = instances_full[y : y + self.size, x : x + self.size]
            if (instances > 0).mean() >= self.min_foreground_fraction:
                break
        patch = image[y : y + self.size, x : x + self.size].copy()
        types = types_full[y : y + self.size, x : x + self.size].copy()
        if self.augment:
            turns = int(rng.integers(0, 4))
            patch, instances, types = (
                np.rot90(patch, turns),
                np.rot90(instances, turns),
                np.rot90(types, turns),
            )
            if rng.random() < 0.5:
                patch, instances, types = patch[:, ::-1], instances[:, ::-1], types[:, ::-1]
            if rng.random() < 0.5:
                patch, instances, types = patch[::-1], instances[::-1], types[::-1]
        patch = np.ascontiguousarray(patch)
        instances = np.ascontiguousarray(instances)
        types = np.ascontiguousarray(types)
        if self.augment and rng.random() < 0.5:
            gain = rng.uniform(0.85, 1.15, size=(1, 1, 3))
            bias = rng.uniform(-12, 12, size=(1, 1, 3))
            patch = np.clip(patch.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        hv_map = make_hv_map(instances)
        return (
            torch.from_numpy(patch.transpose(2, 0, 1)).float(),
            torch.from_numpy(instances.copy()).long(),
            torch.from_numpy(types.copy()).long(),
            torch.from_numpy(hv_map).float(),
            stem,
        )


def _seed_worker(worker_id: int, seed: int) -> None:
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker = torch.utils.data.get_worker_info()
    if worker is not None:
        worker.dataset.rng = np.random.default_rng(worker_seed)


def create_loaders(
    split: DataSplit, config: MambaUNetConfig, encoding: str
) -> tuple[RandomPatchDataset, DataLoader, DataLoader]:
    patches = 2 if config.smoke_test else config.patches_per_tile
    train_dataset = RandomPatchDataset(
        split.train,
        config.patch_size,
        patches,
        config.seed,
        encoding,
        config.min_foreground_fraction,
        augment=True,
    )
    val_dataset = RandomPatchDataset(
        split.val,
        config.patch_size,
        patches,
        config.seed,
        encoding,
        config.min_foreground_fraction,
        augment=False,
    )
    generator = torch.Generator().manual_seed(config.seed)
    worker_init = partial(_seed_worker, seed=config.seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": True,
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": worker_init,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, generator=generator, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_dataset, train_loader, val_loader
