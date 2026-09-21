from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.io import savemat


CLASS_TO_ID = {
    "epithelial": 1,
    "lymphocyte": 2,
    "macrophage": 3,
    "neutrophil": 4,
}
IGNORED_CLASSES = {"ambiguous", "ambigous"}


def _annotation_class(annotation: ET.Element) -> str:
    attribute = annotation.find("./Attributes/Attribute")
    if attribute is None:
        return ""
    return (attribute.attrib.get("Name") or "").strip().lower()


def rasterize(xml_path: Path, shape: tuple[int, int]):
    height, width = shape
    inst_map = np.zeros((height, width), dtype=np.int32)
    type_map = np.zeros((height, width), dtype=np.uint8)
    counts: Counter[str] = Counter()
    instance_id = 0

    root = ET.parse(xml_path).getroot()
    for annotation in root.findall("Annotation"):
        class_name = _annotation_class(annotation)
        if class_name in IGNORED_CLASSES or not class_name:
            counts[class_name or "missing"] += 1
            continue
        if class_name not in CLASS_TO_ID:
            raise ValueError(
                f"Unknown MoNuSAC class {class_name!r} in {xml_path}"
            )
        class_id = CLASS_TO_ID[class_name]
        for region in annotation.findall("./Regions/Region"):
            vertices = []
            for vertex in region.findall("./Vertices/Vertex"):
                x = int(round(float(vertex.attrib["X"])))
                y = int(round(float(vertex.attrib["Y"])))
                vertices.append((x, y))
            if len(vertices) < 3:
                continue
            polygon = np.asarray(vertices, dtype=np.int32)
            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 1)
            if not mask.any():
                continue
            instance_id += 1
            inst_map[mask > 0] = instance_id
            type_map[mask > 0] = class_id
            counts[class_name] += 1

    return inst_map, type_map, counts


def convert_split(source_root: Path, output_root: Path, split_name: str):
    image_dir = output_root / split_name / "Images"
    label_dir = output_root / split_name / "Labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    totals: Counter[str] = Counter()
    converted = 0
    rows: list[tuple[str, str]] = []
    for image_path in sorted(source_root.rglob("*.tif")):
        xml_path = image_path.with_suffix(".xml")
        if not xml_path.exists():
            continue
        if image_path.stem in seen:
            raise RuntimeError(f"Duplicate MoNuSAC stem: {image_path.stem}")
        seen.add(image_path.stem)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Cannot read image: {image_path}")
        inst_map, type_map, counts = rasterize(xml_path, image.shape[:2])
        shutil.copy2(image_path, image_dir / image_path.name)
        savemat(
            label_dir / f"{image_path.stem}.mat",
            {"inst_map": inst_map, "type_map": type_map},
            do_compression=True,
        )
        totals.update(counts)
        rows.append((image_path.stem, image_path.parent.name))
        converted += 1

    if converted == 0:
        raise RuntimeError(f"No TIFF/XML pairs found under {source_root}")
    print(
        f"{split_name}: converted {converted} images | "
        f"class/ignored annotation counts: {dict(totals)}"
    )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw MoNuSAC 2020 TIFF/XML annotations to MAT maps."
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    train_root = args.raw_root / "Training"
    test_root = args.raw_root / "Testing"
    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(
            "Expected Training/ and Testing/ inside the raw MoNuSAC root"
        )

    train_rows = convert_split(train_root, args.output_root, "Train")
    test_rows = convert_split(test_root, args.output_root, "Test")

    patients = sorted({patient for _, patient in train_rows})
    rng = np.random.default_rng(42)
    shuffled = np.asarray(patients, dtype=object)[rng.permutation(len(patients))]
    number_val = max(1, round(0.15 * len(patients)))
    val_patients = set(shuffled[:number_val].tolist())

    split_path = args.output_root / "split.csv"
    with split_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stem", "split"])
        for stem, patient in train_rows:
            writer.writerow([stem, "val" if patient in val_patients else "train"])
        for stem, _ in test_rows:
            writer.writerow([stem, "test"])

    print("Prepared MoNuSAC dataset at:", args.output_root)
    print("Patient-level split written to:", split_path)
    print("Validation patients:", sorted(val_patients))
    print("Class IDs:", {"background": 0, **CLASS_TO_ID})


if __name__ == "__main__":
    main()
