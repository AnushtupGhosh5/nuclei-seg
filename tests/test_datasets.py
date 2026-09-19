import xml.etree.ElementTree as ET

import numpy as np

from nuclei_seg.datasets import (
    NucleiPatchDataset,
    _augment,
    _collapse_glysac_types,
    _rasterize_monusac,
)
from nuclei_seg.targets import generate_hv_map


def _add_annotation(root, name, points):
    annotation = ET.SubElement(root, "Annotation")
    attributes = ET.SubElement(annotation, "Attributes")
    ET.SubElement(attributes, "Attribute", Name=name)
    regions = ET.SubElement(annotation, "Regions")
    region = ET.SubElement(regions, "Region")
    vertices = ET.SubElement(region, "Vertices")
    for x, y in points:
        ET.SubElement(vertices, "Vertex", X=str(x), Y=str(y))


def test_monusac_ambiguous_regions_are_discarded(tmp_path) -> None:
    root = ET.Element("Annotations")
    _add_annotation(root, "Epithelial", [(1, 1), (4, 1), (4, 4), (1, 4)])
    _add_annotation(root, "Ambiguous", [(7, 7), (10, 7), (10, 10), (7, 10)])
    xml_path = tmp_path / "sample.xml"
    ET.ElementTree(root).write(xml_path)

    instance_map, type_map = _rasterize_monusac(xml_path, (12, 12))

    assert instance_map.max() == 1
    assert set(np.unique(type_map)) == {0, 1}
    assert instance_map[8, 8] == 0
    assert type_map[8, 8] == 0


def test_glysac_types_are_collapsed_to_three_foreground_classes() -> None:
    raw = np.arange(11, dtype=np.int16)

    collapsed = _collapse_glysac_types(raw)

    assert collapsed.tolist() == [0, 1, 1, 3, 2, 2, 2, 2, 3, 1, 1]


def test_augmentation_keeps_targets_aligned_and_rebuilds_hv() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    instance_map = np.zeros((16, 16), dtype=np.int32)
    instance_map[2:7, 3:9] = 1
    type_map = np.zeros((16, 16), dtype=np.int16)
    type_map[instance_map == 1] = 3
    hv_map = generate_hv_map(instance_map)
    np.random.seed(0)

    _, augmented_instances, augmented_types, augmented_hv = _augment(
        image, instance_map, type_map, hv_map
    )

    assert np.array_equal(augmented_instances > 0, augmented_types == 3)
    assert np.allclose(augmented_hv, generate_hv_map(augmented_instances))


def test_training_dataset_indexes_records_by_foreground_class(tmp_path) -> None:
    records = []
    for index, class_id in enumerate((1, 1, 1, 4)):
        instance_map = np.zeros((16, 16), dtype=np.int32)
        instance_map[4:12, 4:12] = 1
        type_map = np.zeros((16, 16), dtype=np.int16)
        type_map[instance_map == 1] = class_id
        path = tmp_path / f"sample_{index}.npz"
        np.savez_compressed(
            path,
            image=np.zeros((16, 16, 3), dtype=np.uint8),
            instance_map=instance_map,
            type_map=type_map,
            hv_map=generate_hv_map(instance_map),
        )
        records.append({"path": str(path), "split": "train", "group": str(index)})

    dataset = NucleiPatchDataset(records, patch_size=8, patches_per_image=1, training=True)

    assert dataset.class_records == {1: [0, 1, 2], 4: [3]}
