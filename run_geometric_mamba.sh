#!/bin/bash
set -Eeuo pipefail

IMAGE_NAME="${IMAGE_NAME:-nuclei-segmentation}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_CONFIG="configs/pde_geometric_mamba_glysac_v2.json"
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
Run a GLySAC geometric Mamba experiment.
Default config: $DEFAULT_CONFIG
Scan modes: cartesian, pde, normal, tangential, hybrid

Examples:
  ./run_geometric_mamba.sh
  ./run_geometric_mamba.sh --smoke-test
  ./run_geometric_mamba.sh --scan-mode normal
  ./run_geometric_mamba.sh --scan-mode hybrid --output-dir outputs/custom_glysac
  ./run_geometric_mamba.sh --resume-checkpoint outputs/custom_glysac/geometric_latest_checkpoint.pth --output-dir outputs/custom_glysac

Overrides:
  CONFIG=configs/custom.json ./run_geometric_mamba.sh
  IMAGE_NAME=custom-image ./run_geometric_mamba.sh

All arguments pass through to geometric_experiment. Without --output-dir,
--scan-mode selects its own output directory; --smoke-test appends _smoke.
EOF
    exit 0
fi

docker run --rm --gpus all --shm-size=8g --network host \
    -v "$PROJECT_DIR:/app" \
    -w /app \
    "$IMAGE_NAME" \
    bash -lc 'python3 scripts/check_mamba_runtime.py && exec env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src python3 -m nuclei_seg.mamba_unet.geometric_experiment "$@"' \
    bash --config "$CONFIG" "$@"
