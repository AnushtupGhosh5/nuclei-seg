from __future__ import annotations

import json
import random
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

from .targets import generate_hv_map, remap_instances


DATASET_INFO = {
    "monusac": {
        "num_types": 5,
        "type_names": ["background", "epithelial", "lymphocyte", "macrophage", "neutrophil"],
    },
    "glysac": {
        "num_types": 4,
        "type_names": ["background", "miscellaneous", "lymphocyte", "epithelial"],
    },
}

MONUSAC_TYPES = {
    "Epithelial": 1,
    "Lymphocyte": 2,
    "Macrophage": 3,
    "Neutrophil": 4,
}
MONUSAC_IGNORED_TYPES = {"Ambiguous"}


def _read_rgb(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    except Exception:
        # Aperio SVS files can use codecs unsupported by Pillow.
        import openslide

        slide = openslide.OpenSlide(str(path))
        image = slide.read_region((0, 0), 0, slide.dimensions).convert("RGB")
        slide.close()
        return np.asarray(image)


def _rasterize_monusac(xml_path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    instance_map = np.zeros(shape, dtype=np.int32)
    type_map = np.zeros(shape, dtype=np.int16)
    next_id = 1
    root = ET.parse(xml_path).getroot()
    for annotation in root.findall(".//Annotation"):
        attribute = annotation.find("./Attributes/Attribute")
        if attribute is None:
            continue
        class_name = attribute.attrib.get("Name", "")
        if class_name in MONUSAC_IGNORED_TYPES:
            continue
        if class_name not in MONUSAC_TYPES:
            continue
        class_id = MONUSAC_TYPES[class_name]
        for region in annotation.findall("./Regions/Region"):
            points = [
                (round(float(vertex.attrib["X"])), round(float(vertex.attrib["Y"])))
                for vertex in region.findall("./Vertices/Vertex")
            ]
            if len(points) < 3:
                continue
            polygon = np.asarray(points, dtype=np.int32)
            mask = np.zeros(shape, dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 1)
            instance_map[mask == 1] = next_id
            type_map[mask == 1] = class_id
            next_id += 1
    return remap_instances(instance_map), type_map


def _collapse_glysac_types(type_map: np.ndarray) -> np.ndarray:
    """Collapse the raw GlySAC taxonomy to three foreground classes.

    Raw IDs 8-10 only occur in the test taxonomy, although ID 8 is also
    present in the copy of the training set distributed in this workspace.
    Applying one mapping to both splits keeps their output label semantics
    identical: 1=miscellaneous, 2=lymphocyte, 3=epithelial.
    """
    collapsed = np.zeros(type_map.shape, dtype=np.int16)
    collapsed[np.isin(type_map, (1, 2, 9, 10))] = 1
    collapsed[np.isin(type_map, (4, 5, 6, 7))] = 2
    collapsed[np.isin(type_map, (3, 8))] = 3
    unknown = np.setdiff1d(np.unique(type_map), np.arange(11))
    if len(unknown):
        raise ValueError(f"Unexpected GlySAC type IDs: {unknown.tolist()}")
    return collapsed


def _write_sample(
    output_path: Path,
    image: np.ndarray,
    instance_map: np.ndarray,
    type_map: np.ndarray,
) -> None:
    if image.shape[:2] != instance_map.shape or instance_map.shape != type_map.shape:
        raise ValueError(f"Mismatched shapes for {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        image=image.astype(np.uint8),
        instance_map=instance_map.astype(np.int32),
        type_map=type_map.astype(np.int16),
        hv_map=generate_hv_map(instance_map).astype(np.float16),
    )


def prepare_glysac(data_dir: Path, processed_dir: Path, overwrite: bool = False) -> list[dict]:
    root = data_dir / "glysac_dataset"
    manifest = []
    for split_name in ("Train", "Test"):
        for image_path in sorted((root / split_name / "Images").glob("*.png")):
            label_path = root / split_name / "Labels" / f"{image_path.stem}.mat"
            if not label_path.exists():
                raise FileNotFoundError(label_path)
            output_path = processed_dir / "glysac" / split_name.lower() / f"{image_path.stem}.npz"
            if overwrite or not output_path.exists():
                annotation = loadmat(label_path)
                _write_sample(
                    output_path,
                    _read_rgb(image_path),
                    remap_instances(annotation["inst_map"]),
                    _collapse_glysac_types(annotation["type_map"]),
                )
            group = re.sub(r"_(normal|tumor|mixed|new_cancer|new_normal|new_mixed).*", "", image_path.stem)
            group = re.sub(r"_\d+$", "", group)
            manifest.append({"path": str(output_path), "split": split_name.lower(), "group": group})
    return manifest


def _monusac_roots(data_dir: Path) -> dict[str, Path]:
    base = data_dir / "MONUSAC 2020 segmentation classification"
    return {
        "train": base / "Training" / "MoNuSAC_images_and_annotations" / "MoNuSAC_images_and_annotations",
        "test": base / "Testing" / "MoNuSAC Testing Data and Annotations" / "MoNuSAC Testing Data and Annotations",
    }


def prepare_monusac(data_dir: Path, processed_dir: Path, overwrite: bool = False) -> list[dict]:
    manifest = []
    for split_name, root in _monusac_roots(data_dir).items():
        for xml_path in sorted(root.glob("*/*.xml")):
            image_path = xml_path.with_suffix(".tif")
            if not image_path.exists():
                image_path = xml_path.with_suffix(".svs")
            if not image_path.exists():
                raise FileNotFoundError(f"No image for {xml_path}")
            output_path = processed_dir / "monusac" / split_name / f"{xml_path.stem}.npz"
            if overwrite or not output_path.exists():
                image = _read_rgb(image_path)
                instance_map, type_map = _rasterize_monusac(xml_path, image.shape[:2])
                _write_sample(output_path, image, instance_map, type_map)
            manifest.append({"path": str(output_path), "split": split_name, "group": xml_path.parent.name})
    return manifest


def prepare_dataset(dataset: str, data_dir: Path, overwrite: bool = False) -> Path:
    processed_dir = data_dir / "processed"
    if dataset == "monusac":
        records = prepare_monusac(data_dir, processed_dir, overwrite)
    elif dataset == "glysac":
        records = prepare_glysac(data_dir, processed_dir, overwrite)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    manifest_path = processed_dir / dataset / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"dataset": dataset, "records": records}, indent=2))
    return manifest_path


def load_manifest(data_dir: Path, dataset: str) -> list[dict]:
    path = data_dir / "processed" / dataset / "manifest.json"
    if not path.exists():
        path = prepare_dataset(dataset, data_dir)
    records = json.loads(path.read_text())["records"]
    # Store relative paths where possible so manifests remain portable.
    for record in records:
        candidate = Path(record["path"])
        if not candidate.exists():
            record["path"] = str(data_dir / "processed" / dataset / record["split"] / candidate.name)
    return records


def split_records(records: list[dict], seed: int, val_fraction: float) -> tuple[list[dict], list[dict], list[dict]]:
    train_pool = [record for record in records if record["split"] == "train"]
    test = [record for record in records if record["split"] == "test"]
    groups = sorted({record["group"] for record in train_pool})
    random.Random(seed).shuffle(groups)
    num_val = max(1, round(len(groups) * val_fraction))
    val_groups = set(groups[:num_val])
    train = [record for record in train_pool if record["group"] not in val_groups]
    val = [record for record in train_pool if record["group"] in val_groups]
    return train, val, test


@lru_cache(maxsize=12)
def _load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _pad_to_patch(array: np.ndarray, patch_size: int, image: bool = False) -> np.ndarray:
    height, width = array.shape[:2]
    pad_h, pad_w = max(0, patch_size - height), max(0, patch_size - width)
    if not (pad_h or pad_w):
        return array
    pads = ((0, pad_h), (0, pad_w)) + (((0, 0),) if array.ndim == 3 else ())
    mode = "reflect" if image and min(height, width) > 1 else "constant"
    return np.pad(array, pads, mode=mode)


def _augment(image: np.ndarray, instance: np.ndarray, type_map: np.ndarray, hv: np.ndarray):
    geometry_changed = False
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        instance = np.fliplr(instance)
        type_map = np.fliplr(type_map)
        geometry_changed = True
    if np.random.rand() > 0.5:
        image = np.flipud(image)
        instance = np.flipud(instance)
        type_map = np.flipud(type_map)
        geometry_changed = True
    if np.random.rand() > 0.5:
        k = int(np.random.choice([1, 2, 3]))
        image = np.rot90(image, k)
        instance = np.rot90(instance, k)
        type_map = np.rot90(type_map, k)
        geometry_changed = True

    # Flips and rotations can create negative-stride views, which OpenCV does
    # not consistently accept. Blurring is intentionally image-only.
    image = np.ascontiguousarray(image, dtype=np.uint8)
    instance = np.ascontiguousarray(instance)
    type_map = np.ascontiguousarray(type_map)
    if np.random.rand() < 0.2:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    if np.random.rand() < 0.2:
        image = cv2.medianBlur(image, 5)

    # HV values encode direction, so a spatially transformed instance map
    # needs a newly generated vector field rather than a plain array transform.
    if geometry_changed:
        hv = generate_hv_map(instance)
    return image, instance, type_map, hv


class NucleiPatchDataset(Dataset):
    def __init__(
        self,
        records: Iterable[dict],
        patch_size: int,
        patches_per_image: int,
        training: bool,
        foreground_probability: float = 0.75,
    ) -> None:
        if not 0.0 <= foreground_probability <= 1.0:
            raise ValueError("foreground_probability must be between 0 and 1")
        self.records = list(records)
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.training = training
        self.foreground_probability = foreground_probability
        self.class_records: dict[int, list[int]] = {}
        if training:
            for record_index, record in enumerate(self.records):
                type_map = _load_npz(record["path"])["type_map"]
                for class_id in np.unique(type_map):
                    class_id = int(class_id)
                    if class_id > 0:
                        self.class_records.setdefault(class_id, []).append(record_index)

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_image

    def __getitem__(self, index: int) -> dict:
        anchor = None
        record_index = index // self.patches_per_image
        if self.training and self.class_records and random.random() < self.foreground_probability:
            class_id = random.choice(sorted(self.class_records))
            record_index = random.choice(self.class_records[class_id])
            anchor = class_id

        record = self.records[record_index]
        sample = _load_npz(record["path"])
        image = _pad_to_patch(sample["image"], self.patch_size, image=True)
        instance = _pad_to_patch(sample["instance_map"], self.patch_size)
        type_map = _pad_to_patch(sample["type_map"], self.patch_size)
        hv = _pad_to_patch(sample["hv_map"].astype(np.float32), self.patch_size)
        height, width = instance.shape

        if anchor is not None:
            ys, xs = np.nonzero(type_map == anchor)
            chosen = random.randrange(len(xs))
            y0 = int(np.clip(ys[chosen] - random.randrange(self.patch_size), 0, height - self.patch_size))
            x0 = int(np.clip(xs[chosen] - random.randrange(self.patch_size), 0, width - self.patch_size))
        elif self.training:
            y0 = random.randint(0, height - self.patch_size)
            x0 = random.randint(0, width - self.patch_size)
        else:
            slot = index % self.patches_per_image
            rng = random.Random(f"{record['path']}:{slot}")
            y0 = rng.randint(0, height - self.patch_size)
            x0 = rng.randint(0, width - self.patch_size)

        crop = np.s_[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        image, instance, type_map, hv = image[crop], instance[crop], type_map[crop], hv[crop]
        if self.training:
            image, instance, type_map, hv = _augment(image, instance, type_map, hv)
        return {
            "image": np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0,
            "instance_map": np.ascontiguousarray(instance, dtype=np.int64),
            "np_map": np.ascontiguousarray(instance > 0, dtype=np.int64),
            "type_map": np.ascontiguousarray(type_map, dtype=np.int64),
            "hv_map": np.ascontiguousarray(hv.transpose(2, 0, 1), dtype=np.float32),
            "name": Path(record["path"]).stem,
        }
