from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

CLASS_TO_ID = {
    "Epithelial": 1,
    "Lymphocyte": 2,
    "Macrophage": 3,
    "Neutrophil": 4,
}
IGNORE_CLASS = "Ambiguous"


def _find_raw_roots(source_root: Path) -> tuple[Path, Path]:
    train = source_root / "Training" / "MoNuSAC_images_and_annotations"
    nested = train / "MoNuSAC_images_and_annotations"
    if nested.exists():
        train = nested
    test = source_root / "Testing" / "MoNuSAC Testing Data and Annotations"
    nested = test / "MoNuSAC Testing Data and Annotations"
    if nested.exists():
        test = nested
    if not train.exists() or not test.exists():
        raise FileNotFoundError(
            f"Could not locate MoNuSAC train/test roots below {source_root}"
        )
    return train, test


def _read_image(xml_path: Path) -> np.ndarray:
    tif = xml_path.with_suffix(".tif")
    if tif.exists():
        image = cv2.imread(str(tif), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Cannot read {tif}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    svs = xml_path.with_suffix(".svs")
    if not svs.exists():
        raise FileNotFoundError(f"No .tif or .svs paired with {xml_path}")
    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError(
            "SVS-only MoNuSAC samples require openslide-python"
        ) from exc
    slide = openslide.OpenSlide(str(svs))
    width, height = slide.dimensions
    image = np.asarray(
        slide.read_region((0, 0), 0, (width, height)).convert("RGB")
    )
    slide.close()
    return image


def _annotation_name(annotation: ET.Element) -> str:
    attributes = annotation.find("Attributes")
    if attributes is None:
        return ""
    for item in attributes.findall("Attribute"):
        name = item.attrib.get("Name", "").strip()
        if name:
            return name
    return ""


def _polygon(region: ET.Element) -> np.ndarray | None:
    vertices = region.find("Vertices")
    if vertices is None:
        return None
    points = []
    for vertex in vertices.findall("Vertex"):
        points.append(
            [round(float(vertex.attrib["X"])), round(float(vertex.attrib["Y"]))]
        )
    if len(points) < 3:
        return None
    polygon = np.asarray(points, dtype=np.int32)
    return polygon.reshape(-1, 1, 2)


def parse_xml(xml_path: Path, shape: tuple[int, int]):
    height, width = shape
    inst_map = np.zeros((height, width), dtype=np.int32)
    type_map = np.zeros((height, width), dtype=np.uint8)
    ignore_map = np.zeros((height, width), dtype=np.uint8)
    next_instance = 1
    seen_names: set[str] = set()

    root = ET.parse(xml_path).getroot()
    for annotation in root.findall("Annotation"):
        class_name = _annotation_name(annotation)
        if not class_name:
            continue
        seen_names.add(class_name)
        regions = annotation.find("Regions")
        if regions is None:
            continue

        if class_name not in CLASS_TO_ID and class_name != IGNORE_CLASS:
            if class_name.lower() == "description":
                continue
            raise ValueError(
                f"Unknown MoNuSAC class '{class_name}' in {xml_path}"
            )
        for region in regions.findall("Region"):
            polygon = _polygon(region)
            if polygon is None:
                continue
            polygon[:, 0, 0] = np.clip(polygon[:, 0, 0], 0, width - 1)
            polygon[:, 0, 1] = np.clip(polygon[:, 0, 1], 0, height - 1)

            if class_name == IGNORE_CLASS:
                cv2.fillPoly(ignore_map, [polygon], 1)
                continue

            region_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(region_mask, [polygon], 1)
            pixels = region_mask.astype(bool)
            if not pixels.any():
                continue
            inst_map[pixels] = next_instance
            type_map[pixels] = CLASS_TO_ID[class_name]
            next_instance += 1

    return inst_map, type_map, ignore_map, seen_names


def _prepare_one(
    xml_path: Path,
    split: str,
    output_root: Path,
) -> dict:
    image = _read_image(xml_path)
    inst_map, type_map, ignore_map, names = parse_xml(
        xml_path, image.shape[:2]
    )
    if split == "train" and ignore_map.any():
        raise ValueError(f"Unexpected Ambiguous nuclei in training: {xml_path}")

    image_dir = output_root / split.capitalize() / "Images"
    label_dir = output_root / split.capitalize() / "Labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / f"{xml_path.stem}.png"
    label_path = label_dir / f"{xml_path.stem}.mat"
    cv2.imwrite(
        str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )
    sio.savemat(
        label_path,
        {
            "inst_map": inst_map,
            "type_map": type_map,
            "ignore_map": ignore_map,
        },
        do_compression=True,
    )
    return {
        "stem": xml_path.stem,
        "patient": xml_path.parent.name,
        "source_xml": str(xml_path),
        "split": split,
        "instances": int(inst_map.max()),
        "ignored_pixels": int(ignore_map.sum()),
        "classes_seen": ",".join(sorted(names)),
    }


def _patient_split(
    rows: list[dict],
    val_fraction: float,
    seed: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    train_mask = frame["split"].eq("train")
    patients = sorted(frame.loc[train_mask, "patient"].unique())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patients))
    number_val = max(1, round(len(patients) * val_fraction))
    val_patients = {patients[index] for index in order[:number_val]}
    frame.loc[
        train_mask & frame["patient"].isin(val_patients), "split"
    ] = "val"
    return frame
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare raw MoNuSAC XML/TIF/SVS data for Mamba training"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/MONUSAC 2020 segmentation classification"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/monusac_dataset"),
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_root} exists; use --overwrite to rebuild"
            )
        shutil.rmtree(args.output_root)
    train_root, test_root = _find_raw_roots(args.source_root)
    rows: list[dict] = []
    for split, root in (("train", train_root), ("test", test_root)):
        xml_files = sorted(root.rglob("*.xml"))
        print(f"Preparing {split}: {len(xml_files)} XML files")
        for index, xml_path in enumerate(xml_files, 1):
            rows.append(_prepare_one(xml_path, split, args.output_root))
            if index % 25 == 0 or index == len(xml_files):
                print(f"  {index}/{len(xml_files)}")

    manifest = _patient_split(rows, args.val_fraction, args.seed)
    split_csv = args.output_root / "split.csv"
    manifest[["stem", "split"]].to_csv(split_csv, index=False)
    manifest.to_csv(args.output_root / "preparation_manifest.csv", index=False)

    summary = {
        "class_to_id": CLASS_TO_ID,
        "ignored_test_class": IGNORE_CLASS,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "tiles": manifest.groupby("split").size().to_dict(),
        "patients": manifest.groupby("split")["patient"].nunique().to_dict(),
        "ignored_test_pixels": int(
            manifest.loc[manifest.split.eq("test"), "ignored_pixels"].sum()
        ),
    }
    (args.output_root / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))
    print("Prepared dataset:", args.output_root)
    print("Split manifest:", split_csv)


if __name__ == "__main__":
    main()
