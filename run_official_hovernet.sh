#!/bin/bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOVERNET_ROOT="$(cd "$PROJECT_DIR/../hover_net" && pwd)"
ACTION="${1:-all}"

if [[ "${OFFICIAL_HOVERNET_CONTAINER:-0}" != "1" ]]; then
    IMAGE_NAME="${IMAGE_NAME:-nuclei-segmentation}"
    TORCH_CACHE_DIR="$PROJECT_DIR/.torch-cache"
    PATCH_STORAGE_DIR="${PATCH_STORAGE_DIR:-/mnt/ai-storage/nuclei-seg/official_hovernet/glysac}"
    MODEL_STORAGE_DIR="${MODEL_STORAGE_DIR:-/mnt/ai-storage/nuclei-seg/official_hovernet/models}"
    mkdir -p "$TORCH_CACHE_DIR" "$PATCH_STORAGE_DIR" "$MODEL_STORAGE_DIR"
    DOCKER_GPU_ARGS=()
    if [[ "$ACTION" != "prepare" ]]; then
        DOCKER_GPU_ARGS=(--gpus all)
    fi
    docker run --rm "${DOCKER_GPU_ARGS[@]}" --shm-size=8g --network host \
        -e OFFICIAL_HOVERNET_CONTAINER=1 \
        -e GPU_IDS -e RUN_NAME \
        -e VAL_FRACTION -e SPLIT_SEED \
        -e HOVERNET_PRETRAINED \
        -e HOVERNET_FROZEN_BATCH_SIZE -e HOVERNET_BATCH_SIZE -e HOVERNET_VALID_BATCH_SIZE \
        -v "$PROJECT_DIR:/workspace/nuclei-seg" \
        -v "$HOVERNET_ROOT:/workspace/hover_net:ro" \
        -v "$PATCH_STORAGE_DIR:/workspace/official-hovernet-patches" \
        -v "$MODEL_STORAGE_DIR:/workspace/official-hovernet-models" \
        -v "$TORCH_CACHE_DIR:/root/.cache/torch" \
        "$IMAGE_NAME" \
        bash /workspace/nuclei-seg/run_official_hovernet.sh "$ACTION"
    exit $?
fi

GLYSAC_ROOT="$PROJECT_DIR/data/glysac_dataset"
# Large extracted patches live on the AI-storage partition, mounted here by the host wrapper.
PATCH_ROOT="/workspace/official-hovernet-patches"
PATCH_LAYOUT="540x540_164x164"
RUN_NAME="${RUN_NAME:-official_hovernet_fast_glysac_3class}"

export HOVERNET_DATASET_NAME="glysac"
export HOVERNET_MODEL_MODE="fast"
export HOVERNET_NR_TYPE="4"
export HOVERNET_TRAIN_DIR="$PATCH_ROOT/train/$PATCH_LAYOUT"
export HOVERNET_VALID_DIR="$PATCH_ROOT/valid/$PATCH_LAYOUT"
export HOVERNET_LOG_DIR="/workspace/official-hovernet-models/$RUN_NAME"
export HOVERNET_PRETRAINED="${HOVERNET_PRETRAINED:-$PROJECT_DIR/outputs/pretrained/imagenet_resnet50_preact.tar}"
export HOVERNET_FROZEN_BATCH_SIZE="${HOVERNET_FROZEN_BATCH_SIZE:-2}"
export HOVERNET_BATCH_SIZE="${HOVERNET_BATCH_SIZE:-1}"
export HOVERNET_VALID_BATCH_SIZE="${HOVERNET_VALID_BATCH_SIZE:-2}"

prepare() {
    [[ -d "$GLYSAC_ROOT/Train/Images" && -d "$GLYSAC_ROOT/Train/Labels" ]] || {
        echo "ERROR: GlySAC was not found at $GLYSAC_ROOT" >&2
        exit 1
    }
    cd "$HOVERNET_ROOT"
    python3 extract_patches.py \
        --dataset glysac \
        --data-root "$GLYSAC_ROOT" \
        --output-root "$PATCH_ROOT" \
        --window-size 540 \
        --step-size 164 \
        --extract-type mirror \
        --validation-from-train \
        --val-fraction "${VAL_FRACTION:-0.2}" \
        --seed "${SPLIT_SEED:-42}"
}

train() {
    compgen -G "$HOVERNET_TRAIN_DIR/*.npy" >/dev/null || {
        echo "ERROR: no training patches found; run '$0 prepare' first" >&2
        exit 1
    }
    compgen -G "$HOVERNET_VALID_DIR/*.npy" >/dev/null || {
        echo "ERROR: no validation patches found; run '$0 prepare' first" >&2
        exit 1
    }
    [[ -f "$HOVERNET_PRETRAINED" ]] || {
        echo "ERROR: pretrained weights not found at $HOVERNET_PRETRAINED" >&2
        exit 1
    }

    echo "Training official HoVer-Net fast on GlySAC (3 foreground classes + background)"
    echo "Source data: $GLYSAC_ROOT"
    echo "Patches:     $PATCH_ROOT"
    echo "Checkpoints: $HOVERNET_LOG_DIR"
    cd "$HOVERNET_ROOT"
    python3 run_train.py --gpu="${GPU_IDS:-0}"
}

case "$ACTION" in
    prepare) prepare ;;
    train) train ;;
    all) prepare; train ;;
    *) echo "Usage: $0 [prepare|train|all]" >&2; exit 2 ;;
esac
