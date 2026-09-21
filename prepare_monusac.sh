#!/bin/bash
set -Eeuo pipefail

IMAGE_NAME="${IMAGE_NAME:-nuclei-segmentation}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

docker run --rm --shm-size=8g \
    -v "$PROJECT_DIR:/app" \
    -w /app \
    "$IMAGE_NAME" \
    python3 scripts/prepare_monusac.py "$@"
