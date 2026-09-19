import numpy as np
import torch

from nuclei_seg.losses import HoVerNetLoss
from nuclei_seg.engine import profile_model
from nuclei_seg.metrics import evaluate_instances, instance_types_from_map
from nuclei_seg.model import OriginalUNet
from nuclei_seg.postprocess import post_process_instances
from nuclei_seg.targets import generate_hv_map
from nuclei_seg.visualization import (
    TYPE_COLORS,
    instance_label_image,
    type_label_image,
)


def test_model_and_loss_shapes() -> None:
    model = OriginalUNet(5, encoder_channels=(8, 16, 32))
    image = torch.rand(2, 3, 64, 64)
    output = model(image)
    assert output.np_logits.shape == (2, 2, 24, 24)
    assert output.hv_map.shape == (2, 2, 24, 24)
    assert output.type_logits.shape == (2, 5, 24, 24)
    target = {
        "np_map": torch.zeros(2, 24, 24, dtype=torch.long),
        "type_map": torch.zeros(2, 24, 24, dtype=torch.long),
        "hv_map": torch.zeros(2, 2, 24, 24),
    }
    target["np_map"][:, 5:19, 5:19] = 1
    target["type_map"][:, 5:19, 5:19] = 1
    losses = HoVerNetLoss()(output, target)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


def test_model_profile_reports_parameters_flops_and_latency() -> None:
    model = OriginalUNet(5, encoder_channels=(8, 16, 32))
    profile = profile_model(model, torch.device("cpu"), input_size=64)
    assert profile["total_parameters"] == sum(parameter.numel() for parameter in model.parameters())
    assert profile["gflops"] > 0
    assert profile["tile_inference_mean_ms"] > 0
    assert profile["np_output_shape"] == [1, 2, 24, 24]


def test_original_unet_parameter_count_and_layers() -> None:
    model = OriginalUNet(5)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 30_000_000 <= parameters <= 32_000_000
    assert not any(isinstance(module, torch.nn.modules.normalization.GroupNorm) for module in model.modules())
    assert sum(isinstance(module, torch.nn.ConvTranspose2d) for module in model.modules()) == 4


def test_hv_targets_are_normalized_per_instance() -> None:
    instances = np.zeros((20, 24), np.int32)
    instances[2:8, 3:11] = 1
    instances[10:19, 14:22] = 2
    hv = generate_hv_map(instances)
    assert hv.shape == (20, 24, 2)
    assert -1.0001 <= hv.min() <= hv.max() <= 1.0001
    assert np.all(hv[instances == 0] == 0)
    for instance_id in (1, 2):
        assert abs(float(hv[..., 0][instances == instance_id].mean())) < 0.15
        assert abs(float(hv[..., 1][instances == instance_id].mean())) < 0.15


def test_perfect_instance_metrics() -> None:
    instances = np.zeros((32, 32), np.int32)
    instances[2:10, 3:11] = 1
    instances[18:28, 20:30] = 2
    types = np.zeros_like(instances, dtype=np.int16)
    types[instances == 1] = 1
    types[instances == 2] = 2
    instance_types = instance_types_from_map(instances, types)
    metrics = evaluate_instances(
        instances, instances, instance_types, instance_types, ["background", "a", "b"]
    )
    for name in ("dice", "iou", "aji", "aji_plus", "dq", "sq", "pq", "detection_f1", "type_macro_f1"):
        assert np.isclose(metrics[name], 1.0)


def test_ambiguous_type_is_ignored() -> None:
    instances = np.zeros((16, 16), np.int32)
    instances[2:8, 2:8] = 1
    true_types = np.array([0, -1], dtype=np.int16)
    pred_types = np.array([0, 2], dtype=np.int16)
    metrics = evaluate_instances(
        instances, instances, true_types, pred_types, ["background", "a", "b"]
    )
    assert metrics["f1_a"] == 0.0
    assert metrics["f1_b"] == 0.0


def test_watershed_returns_instances() -> None:
    yy, xx = np.mgrid[:64, :64]
    first = (xx - 25) ** 2 + (yy - 32) ** 2 <= 12**2
    second = (xx - 39) ** 2 + (yy - 32) ** 2 <= 12**2
    instances = np.zeros((64, 64), np.int32)
    instances[first] = 1
    instances[second] = 2
    hv = generate_hv_map(instances)
    probability = ((first | second) * 0.99).astype(np.float32)
    predicted, _ = post_process_instances(probability, hv, min_size=10)
    assert predicted.max() >= 2


def test_instance_and_type_visualization_colors_are_separate() -> None:
    instances = np.zeros((16, 16), dtype=np.int32)
    instances[2:7, 2:7] = 1
    instances[9:14, 9:14] = 2
    types = np.array([0, 1, 2], dtype=np.int16)
    neutral = instance_label_image(instances)
    typed = type_label_image(instances, types)
    assert np.all(neutral[instances == 0] == 0)
    assert np.all(typed[instances == 0] == 0)
    assert np.any(np.all(neutral == 255, axis=-1))
    assert np.any(np.all(typed == TYPE_COLORS[1], axis=-1))
    assert np.any(np.all(typed == TYPE_COLORS[2], axis=-1))
