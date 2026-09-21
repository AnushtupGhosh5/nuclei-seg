#!/bin/bash
set -Eeuo pipefail

IMAGE_NAME="${IMAGE_NAME:-nuclei-segmentation}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${CONFIG:-configs/mamba_unet_glysac.json}"

docker run --rm --gpus all --shm-size=8g --network host \
    -v "$PROJECT_DIR:/app" \
    -w /app \
    "$IMAGE_NAME" \
    bash -lc 'python3 scripts/check_mamba_runtime.py && exec env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src python3 -m nuclei_seg.mamba_unet "$@"' \
    bash --config "$CONFIG" "$@"
