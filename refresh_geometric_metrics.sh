#!/bin/bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-nuclei-segmentation}"
DEFAULT_OUTPUT="outputs/pde_geometric_mamba_glysac_v2_hybrid"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Refresh classification and evaluation reports from saved geometric predictions.
Runs on CPU in the existing Docker image; no training or model inference.
Checkpoints and saved predictions are preserved.

Usage:
  ./refresh_geometric_metrics.sh
  ./refresh_geometric_metrics.sh outputs/pde_geometric_mamba_glysac_v2_hybrid
  ./refresh_geometric_metrics.sh outputs/pde_geometric_mamba_monusac_v2_hybrid

Default: outputs/pde_geometric_mamba_glysac_v2_hybrid
Requires a completed full-image run with saved validation/test predictions.
Updates CSV/JSON reports in that directory; existing ZIP archives are unchanged.
Override the Docker image with IMAGE_NAME=custom-image.
EOF
    exit 0
fi

if (( $# > 1 )); then
    echo "Usage: $0 [output_directory]" >&2
    exit 2
fi

OUTPUT_DIR="${1:-$DEFAULT_OUTPUT}"
if [[ "$OUTPUT_DIR" == "$PROJECT_DIR/"* ]]; then
    OUTPUT_DIR="${OUTPUT_DIR#"$PROJECT_DIR/"}"
elif [[ "$OUTPUT_DIR" == /* ]]; then
    echo "The output directory must be inside $PROJECT_DIR" >&2
    exit 2
fi

if [[ ! -f "$PROJECT_DIR/$OUTPUT_DIR/geometric_run_config.json" ]]; then
    echo "Missing geometric_run_config.json in $OUTPUT_DIR" >&2
    exit 1
fi

echo "Refreshing validation/test reports in $OUTPUT_DIR from saved predictions."
exec docker run --rm --network none \
    -v "$PROJECT_DIR:/app" -w /app \
    -e PYTHONPATH=src -e OMP_NUM_THREADS=2 \
    "$IMAGE_NAME" \
    python3 -u scripts/refresh_geometric_metrics.py "$OUTPUT_DIR"
