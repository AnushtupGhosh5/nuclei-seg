#!/bin/bash
set -Eeuo pipefail

IMAGE_NAME="${IMAGE_NAME:-nuclei-segmentation}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_CONFIG="configs/pde_geometric_mamba_monusac_v2.json"
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
Run a MoNuSAC geometric Mamba experiment.
Default config: $DEFAULT_CONFIG
Scan modes: cartesian, pde, normal, tangential, hybrid

Examples:
  ./run_geometric_mamba_monusac.sh
  ./run_geometric_mamba_monusac.sh --smoke-test
  ./run_geometric_mamba_monusac.sh --scan-mode tangential
  ./run_geometric_mamba_monusac.sh --scan-mode hybrid --output-dir outputs/custom_monusac
  ./run_geometric_mamba_monusac.sh --resume-checkpoint outputs/custom_monusac/geometric_latest_checkpoint.pth --output-dir outputs/custom_monusac

Overrides:
  CONFIG=configs/custom.json ./run_geometric_mamba_monusac.sh
  IMAGE_NAME=custom-image ./run_geometric_mamba_monusac.sh

Requires data/monusac_dataset/split.csv (run ./prepare_monusac.sh first).
All arguments pass through to geometric_experiment. Without --output-dir,
--scan-mode selects its own output directory; --smoke-test appends _smoke.
EOF
    exit 0
fi

if [[ ! -f "$PROJECT_DIR/data/monusac_dataset/split.csv" ]]; then
    echo "MoNuSAC is not prepared yet."
    echo "Run: ./prepare_monusac.sh"
    exit 2
fi

docker run --rm --gpus all --shm-size=8g --network host \
    -v "$PROJECT_DIR:/app" \
    -w /app \
    "$IMAGE_NAME" \
    bash -lc 'python3 scripts/check_mamba_runtime.py && exec env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src python3 -m nuclei_seg.mamba_unet.geometric_experiment "$@"' \
    bash --config "$CONFIG" "$@"
