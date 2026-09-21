from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nuclei_seg.mamba_unet.config import MambaUNetConfig
from nuclei_seg.mamba_unet.data import build_split, discover_records, make_hv_map, merge_glysac_types
from nuclei_seg.mamba_unet.metrics import MetricAccumulator
from nuclei_seg.mamba_unet.model import MambaUNetNP_HV_Type
from nuclei_seg.mamba_unet.pde_field import poisson_watershed
from nuclei_seg.mamba_unet.postprocess import instance_type_map, smile_watershed


def test_local_glysac_split_matches_executed_notebook(tmp_path) -> None:
    data_root = Path("data/glysac_dataset")
    if not data_root.exists():
        pytest.skip("Local GLySAC data is not mounted")
    config = MambaUNetConfig(data_root=data_root, output_dir=tmp_path)
    split = build_split(discover_records(data_root), config, tmp_path)
    assert (len(split.train), len(split.val), len(split.test)) == (29, 5, 25)
    assert split.identity_sha256 == "edf16c64430947fcc94df06147d0798a373cae1f9fa2b3b4da16125ed5098d2d"


def test_mamba_unet_architecture_and_parameter_partition() -> None:
    model = MambaUNetNP_HV_Type()
    report = model.parameter_report()
    assert report == {
        "total": 19_122_056,
        "trainable": 19_122_056,
        "encoder_and_bottleneck": 13_754_016,
        "native_decoder": 5_367_264,
        "task_heads": 776,
    }
    assert model.core.output.__class__.__name__ == "Identity"
    assert model.np_head.out_channels == 2
    assert model.hv_head.out_channels == 2
    assert model.tp_head.out_channels == 4


def test_reference_type_mapping_and_hv_targets() -> None:
    raw = np.arange(11, dtype=np.int32)
    assert merge_glysac_types(raw, "original_10_class").tolist() == [0, 1, 1, 3, 2, 2, 2, 2, 3, 1, 1]
    instances = np.zeros((32, 32), dtype=np.int32)
    instances[5:15, 4:12] = 1
    instances[18:29, 20:30] = 2
    hv_map = make_hv_map(instances)
    assert hv_map.shape == (2, 32, 32)
    assert np.all(hv_map[:, instances == 0] == 0)
    assert -1.0001 <= hv_map.min() <= hv_map.max() <= 1.0001


def test_reference_watershed_and_instance_types() -> None:
    y, x = np.mgrid[:64, :64]
    first = (x - 25) ** 2 + (y - 32) ** 2 <= 12**2
    second = (x - 39) ** 2 + (y - 32) ** 2 <= 12**2
    truth = np.zeros((64, 64), np.int32)
    truth[first] = 1
    truth[second] = 2
    hv_map = make_hv_map(truth).transpose(1, 2, 0)
    predicted = smile_watershed(((first | second) * 0.99).astype(np.float32), hv_map)
    assert predicted.max() >= 2
    pixel_types = np.zeros((64, 64), np.uint8)
    pixel_types[first] = 2
    pixel_types[second] = 3
    typed = instance_type_map(predicted, pixel_types)
    assert set(np.unique(typed)) >= {0, 2, 3}


def test_metric_accumulator_is_perfect_for_perfect_prediction() -> None:
    instances = np.zeros((32, 32), np.int32)
    instances[2:10, 3:11] = 1
    instances[18:28, 20:30] = 2
    types = np.zeros_like(instances, dtype=np.uint8)
    types[instances == 1] = 2
    types[instances == 2] = 3
    accumulator = MetricAccumulator(match_iou=0.5, magnification=40)
    accumulator.update("perfect", instances, types, instances, types, 1.0)
    summary, *_ = accumulator.finalize()
    for key in (
        "binary_dice_image_mean",
        "binary_iou_image_mean",
        "binary_dq_global",
        "binary_sq_global",
        "binary_pq_global",
        "multiclass_dq_global",
        "multiclass_sq_global",
        "multiclass_pq_global",
        "centroid_detection_f1_global",
    ):
        assert np.isclose(summary[key], 1.0)


def test_poisson_watershed_preserves_fragmented_nuclear_signal() -> None:
    y, x = np.mgrid[:96, :96]
    field = np.zeros((96, 96), dtype=np.float32)
    for center_x in (30, 66):
        nucleus = (x - center_x) ** 2 + (y - 48) ** 2 <= 11**2
        fragmented = nucleus & ((x + y) % 3 == 0)
        field[fragmented] = 1.0
    instances = poisson_watershed(field)
    assert instances.max() >= 2
    assert np.count_nonzero(instances) > 100
