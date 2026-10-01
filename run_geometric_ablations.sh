#!/bin/bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    cat <<'EOF'
Run geometric ablations sequentially: cartesian, normal, tangential, hybrid.
Usage: ./run_geometric_ablations.sh <monusac|glysac> <--smoke-test|--full>

Examples:
  ./run_geometric_ablations.sh monusac --smoke-test
  ./run_geometric_ablations.sh glysac --smoke-test
  ./run_geometric_ablations.sh monusac --full

An explicit dataset and run mode are required. --full starts four full runs.
Stops immediately on any failure. Each mode uses its own output directory.
CONFIG and IMAGE_NAME environment overrides are passed to the dataset runner.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 2 ]]; then
    usage >&2
    exit 2
fi

case "$1" in
    monusac) RUNNER="$PROJECT_DIR/run_geometric_mamba_monusac.sh" ;;
    glysac) RUNNER="$PROJECT_DIR/run_geometric_mamba.sh" ;;
    *) echo "Dataset must be monusac or glysac." >&2; exit 2 ;;
esac

RUN_ARGS=()
case "$2" in
    --smoke-test) RUN_ARGS+=(--smoke-test) ;;
    --full) ;;
    *) echo "Choose --smoke-test or --full explicitly." >&2; exit 2 ;;
esac

for SCAN_MODE in cartesian normal tangential hybrid; do
    echo "Running $1 $SCAN_MODE ($2)"
    "$RUNNER" --scan-mode "$SCAN_MODE" "${RUN_ARGS[@]}"
done
