#!/bin/bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Run a GLySAC geometric Mamba experiment.
Default config: configs/pde_geometric_mamba_glysac_v2.json
Scan modes: cartesian, pde, normal, tangential, hybrid

Examples:
  ./run_geometric_mamba_glysac.sh --smoke-test
  ./run_geometric_mamba_glysac.sh --scan-mode hybrid
  ./run_geometric_mamba_glysac.sh --scan-mode normal --output-dir outputs/custom_glysac

CONFIG and IMAGE_NAME environment overrides and all experiment arguments
are forwarded to run_geometric_mamba.sh.
EOF
    exit 0
fi

exec "$PROJECT_DIR/run_geometric_mamba.sh" "$@"
