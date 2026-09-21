from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

from nuclei_seg.mamba_unet.geometric_config import PDEGeometricConfig
from nuclei_seg.mamba_unet.geometric_model import (
    PDEGeometricMambaUNet,
    verify_geometric_output_contract,
)
from nuclei_seg.mamba_unet.geometric_postprocess import mask_pde_watershed
from nuclei_seg.mamba_unet.model import MambaUNetNP_HV_Type
from nuclei_seg.mamba_unet.pde_field import make_poisson_field
from nuclei_seg.mamba_unet.pde_scan import (
    PDEPermutationCache,
    build_pde_permutations,
    gather_sequence,
    inverse_permutation,
    restore_sequence,
)
from nuclei_seg.mamba_unet.pde_ss2d import PDEGuidedSS2D


def _touching_instances(size: int = 96) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    first = (x - 38) ** 2 + (y - 48) ** 2 <= 16**2
    second = (x - 58) ** 2 + (y - 48) ** 2 <= 16**2
    instances = np.zeros((size, size), np.int32)
    instances[first] = 1
    instances[second] = 2
    return instances


def _irregular_guidance() -> torch.Tensor:
    field = torch.zeros(1, 1, 19, 23)
    field[:, :, 2:15, 3:7] = torch.linspace(0.1, 1.0, 13)[None, None, :, None]
    field[:, :, 11:16, 7:18] = 0.7
    field[:, :, 4:9, 15:21] = 0.4
    return field


def test_instancewise_pde_target_is_finite_bounded_and_separate() -> None:
    instances = _touching_instances()
    field = make_poisson_field(instances, iterations=64)
    assert np.isfinite(field).all()
    assert 0 <= float(field.min()) <= float(field.max()) <= 1.0
    for instance_id in (1, 2):
        values = field[instances == instance_id]
        assert np.isclose(values.max(), 1.0)
        assert values.mean() > 0
    maxima = ndi.maximum_filter(field, size=9) == field
    assert sum(np.any(maxima & (instances == item)) for item in (1, 2)) == 2


def test_mask_pde_watershed_recovers_two_touching_nuclei() -> None:
    instances = _touching_instances()
    field = make_poisson_field(instances, iterations=96)
    mask = (instances > 0).astype(np.float32) * 0.99
    predicted = mask_pde_watershed(
        mask, field, nucleus_threshold=0.5,
        marker_threshold=0.3, min_distance=8,
        min_size=10, smoothing_sigma=1.0,
    )
    assert predicted.max() == 2


@pytest.mark.parametrize("height,width", [(8, 8), (16, 16), (13, 17)])
def test_pde_permutations_are_bijections_and_restore_exactly(
    height: int, width: int
) -> None:
    guidance = _irregular_guidance().repeat(2, 1, 1, 1)
    permutations = build_pde_permutations(
        guidance, height, width, num_pde_bins=16, window_size=4
    )
    expected = torch.arange(height * width)
    for permutation, inverse in (
        (permutations.normal, permutations.normal_inverse),
        (permutations.tangent, permutations.tangent_inverse),
    ):
        assert torch.equal(torch.sort(permutation[0]).values, expected)
        assert torch.equal(inverse[0, permutation[0]], expected)
        values = torch.randn(2, 5, height * width)
        restored = restore_sequence(
            gather_sequence(values, permutation), inverse
        )
        assert torch.equal(values, restored)


def test_forward_reverse_orders_and_irregular_modes_differ() -> None:
    permutations = build_pde_permutations(
        _irregular_guidance(), 16, 16,
        num_pde_bins=16, window_size=4,
    )
    normal_reverse = torch.flip(permutations.normal, dims=[-1])
    tangent_reverse = torch.flip(permutations.tangent, dims=[-1])
    assert torch.equal(torch.flip(normal_reverse, dims=[-1]), permutations.normal)
    assert torch.equal(torch.flip(tangent_reverse, dims=[-1]), permutations.tangent)
    assert not torch.equal(permutations.normal, permutations.tangent)

    values = torch.arange(256).reshape(1, 1, 256).float()
    for forward, inverse in (
        (permutations.normal, permutations.normal_inverse),
        (permutations.tangent, permutations.tangent_inverse),
    ):
        reverse = torch.flip(forward, dims=[-1])
        reverse_inverse = inverse_permutation(reverse)
        assert torch.equal(
            restore_sequence(gather_sequence(values, reverse), reverse_inverse),
            values,
        )
        # This is the exact flip/restore path used after a reverse selective
        # scan in PDEGuidedSS2D (identity scan used here for isolation).
        reversed_sequence = torch.flip(
            gather_sequence(values, forward), dims=[-1]
        )
        assert torch.equal(
            restore_sequence(
                torch.flip(reversed_sequence, dims=[-1]), inverse
            ),
            values,
        )


def test_permutation_cache_reuses_each_resolution() -> None:
    cache = PDEPermutationCache(
        _irregular_guidance(), num_pde_bins=16, window_size=4
    )
    assert cache.get(16, 16) is cache.get(16, 16)
    assert cache.get(8, 8) is not cache.get(16, 16)


def test_guided_model_api_cannot_accept_ground_truth_guidance() -> None:
    signature = inspect.signature(PDEGeometricMambaUNet.forward)
    assert list(signature.parameters) == ["self", "images"]


def test_original_cartesian_model_is_structurally_unchanged() -> None:
    original = MambaUNetNP_HV_Type()
    assert not any(
        isinstance(module, PDEGuidedSS2D) for module in original.modules()
    )
    assert original.parameter_report()["total"] == 19_122_056


@pytest.mark.skipif(not torch.cuda.is_available(), reason="selective scan requires CUDA")
@pytest.mark.parametrize(
    "scan_mode", ["cartesian", "pde", "hybrid", "normal", "tangential"]
)
def test_all_ablation_output_shapes_are_finite(scan_mode: str) -> None:
    config = PDEGeometricConfig(scan_mode=scan_mode, smoke_test=True)
    model = PDEGeometricMambaUNet(config).cuda()
    shapes = verify_geometric_output_contract(model, torch.device("cuda"), 256)
    assert shapes["np"] == (1, 2, 256, 256)
    assert shapes["field"] == (1, 1, 256, 256)
    assert shapes["tp"] == (1, 4, 256, 256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="selective scan requires CUDA")
def test_zero_gate_hybrid_preserves_cartesian_block_output() -> None:
    from nuclei_seg.mamba_unet.official.mamba_sys import SS2D

    torch.manual_seed(42)
    cartesian = SS2D(d_model=24, d_state=8).cuda().eval()
    hybrid = PDEGuidedSS2D.from_cartesian(
        cartesian, scan_mode="hybrid"
    ).cuda().eval()
    hybrid.set_scan_cache(
        PDEPermutationCache(
            torch.rand(2, 1, 8, 8, device="cuda"),
            num_pde_bins=8, window_size=4,
        )
    )
    values = torch.randn(2, 8, 8, 24, device="cuda")
    with torch.inference_mode():
        expected = cartesian(values)
        actual = hybrid(values)
    assert torch.equal(expected, actual)
