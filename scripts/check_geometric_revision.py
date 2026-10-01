"""CPU-only parameter and synthetic ordering audit; never starts training.

Run with PYTHONPATH=src in the project runtime. Generated reports are separate
from training metadata and can coexist with an as-yet untrained v2 directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from nuclei_seg.mamba_unet.data import build_split, detect_type_encoding, discover_records, read_annotation, type_class_counts
from nuclei_seg.mamba_unet.geometric_config import PDEGeometricConfig
from nuclei_seg.mamba_unet.geometric_experiment import guided_stage_resolutions, normal_scan_options
from nuclei_seg.mamba_unet.geometric_loss import foreground_type_weights
from nuclei_seg.mamba_unet.geometric_model import PDEGeometricMambaUNet
from nuclei_seg.mamba_unet.loss import class_weights
from nuclei_seg.mamba_unet.pde_scan import build_pde_permutations, scan_diagnostics


def synthetic_fields():
    # Same block-shaped field as the existing permutation/locality regression.
    regression = torch.zeros(1, 1, 19, 23)
    regression[:, :, 2:15, 3:7] = torch.linspace(0.1, 1.0, 13)[None, None, :, None]
    regression[:, :, 11:16, 7:18] = 0.7
    regression[:, :, 4:9, 15:21] = 0.4
    y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    first = torch.exp(-(((x - 4.5) / 3.2) ** 2 + ((y - 6.0) / 5.0) ** 2))
    second = torch.exp(-(((x - 11.5) / 2.1) ** 2 + ((y - 10.5) / 3.0) ** 2))
    notch = torch.where((x > 5) & (y < 7), 0.35, 1.0)
    smooth = torch.maximum(first * notch, second)[None, None]
    return {"regression_irregular": regression, "smooth_two_peak": smooth}


def revision_report(config):
    dataset = "monusac" if config.type_encoding == "monusac_4class" else "glysac"
    baseline = PDEGeometricConfig.from_json(Path(f"configs/pde_geometric_mamba_{dataset}.json"))
    old = PDEGeometricMambaUNet(baseline).parameter_report()
    new = PDEGeometricMambaUNet(config).parameter_report()
    scans = {}
    for name, field in synthetic_fields().items():
        options = dict(num_pde_bins=config.num_pde_bins, window_size=config.pde_scan_window, **normal_scan_options(config))
        base = build_pde_permutations(field, 16, 16, **options)
        noise = torch.randn(field.shape, generator=torch.Generator().manual_seed(config.seed)) * 1e-3
        alternate = build_pde_permutations((field + noise).clamp(0, 1), 16, 16, **options)
        options["normal_algorithm"] = "legacy_global"
        legacy = build_pde_permutations(field, 16, 16, **options)
        legacy_alternate = build_pde_permutations((field + noise).clamp(0, 1), 16, 16, **options)
        scans[name] = {"revised": scan_diagnostics(base, alternate), "legacy_global": scan_diagnostics(legacy, legacy_alternate)}
    return {
        "config": config.to_dict(),
        "baseline_parameters": old,
        "revised_parameters": new,
        "parameter_delta": int(new["total"]) - int(old["total"]),
        "guide_native_resolution": [config.patch_size // (8 * 2 ** config.guide_source_stage)] * 2,
        "guided_stage_resolutions": guided_stage_resolutions(config),
        "synthetic_scans": scans,
        "runtime": {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()},
        "limitations": [
            "CPU structure/ordering audit only; no CUDA forward or training.",
            "Local ray and level-band orders approximate geometry, not numerical streamline integration.",
            "All-step jumps include group/window transitions; alignment excludes them and unresolved gradients.",
            "Signed normal cosine measures ascent; absolute tangent cosine allows either contour orientation.",
            "Rank sensitivity remains, particularly on exact potential ties and quantization thresholds.",
            "Legacy global alignment uses local groups as a filter and is not a global streamline score.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pde_geometric_mamba_monusac_v2.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count-training-labels", action="store_true")
    args = parser.parse_args()
    config = PDEGeometricConfig.from_json(args.config)
    config.validate()
    report = revision_report(config)
    if args.count_training_labels:
        records = discover_records(config.data_root, config.split_csv)
        encoding = detect_type_encoding(records, config.type_encoding)
        # The standard split routine writes its manifest only in this temporary
        # directory. Images are never loaded; one annotation is held at a time.
        with TemporaryDirectory(prefix="geometric-label-audit-") as directory:
            split = build_split(records, config, Path(directory))
        counts = np.zeros(len(config.type_classes), dtype=np.int64)
        for record in split.train:
            instances, types = read_annotation(record.label, encoding)
            padding = [(0, max(0, config.patch_size - side)) for side in instances.shape]
            instances, types = np.pad(instances, padding), np.pad(types, padding)
            counts += type_class_counts([(None, instances, types, record.stem)], len(counts), config.type_weight_basis)
        weights = (
            class_weights(counts, torch.device("cpu")) if config.type_weight_basis == "legacy_pixel"
            else foreground_type_weights(counts, torch.device("cpu"), config.type_loss_foreground_only)
        )
        report["training_type_weight_audit"] = {
            "basis": config.type_weight_basis,
            "encoding": encoding,
            "class_names": list(config.type_classes),
            "counts": counts.tolist(),
            "weights": weights.tolist(),
            "training_tiles": len(split.train),
            "split_identity_sha256": split.identity_sha256,
            "scope": "training labels after validation holdout, padded as cached tiles; configured count basis; validation/test excluded from counts",
        }
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Audit output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
