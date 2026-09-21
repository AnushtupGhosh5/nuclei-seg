"""Fail fast when the Docker image lacks the compiled Mamba runtime."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        import torch
        import transformers
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
    except Exception as error:
        raise SystemExit(
            "The Docker image does not contain a working Mamba runtime. "
            "Run ./build.sh once, then retry the launcher.\n"
            f"Import error: {error!r}"
        ) from error

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not visible inside Docker. Check the NVIDIA Container "
            "Toolkit and verify `docker run --rm --gpus all "
            "nuclei-segmentation nvidia-smi`."
        )

    print(
        "Mamba Docker runtime: PASS | "
        f"Python {sys.version_info.major}.{sys.version_info.minor} | "
        f"Torch {torch.__version__} | CUDA {torch.version.cuda} | "
        f"transformers {transformers.__version__}"
    )


if __name__ == "__main__":
    main()

