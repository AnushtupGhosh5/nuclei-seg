#!/bin/bash
set -e

IMAGE_NAME="nuclei-segmentation"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATA_DIR="$PROJECT_DIR/data"
SRC_DIR="$PROJECT_DIR/src"
OUTPUT_DIR="$PROJECT_DIR/outputs"
SCRIPT="$PROJECT_DIR/runScript.sh"
TORCH_CACHE_DIR="$PROJECT_DIR/.torch-cache"

mkdir -p "$DATA_DIR" "$SRC_DIR" \
    "$OUTPUT_DIR/models" "$OUTPUT_DIR/results" "$TORCH_CACHE_DIR"

docker run --rm --gpus all --shm-size=8g --network host \
    -e DATASET -e RUN_NAME -e PRETRAINED_CHECKPOINT -e PRETRAINED_URL \
    -e EPOCHS -e FREEZE_ENCODER_EPOCHS -e TRAIN_PATCHES_PER_IMAGE \
    -e FOREGROUND_PROBABILITY -e VISUALIZATIONS \
    -e FROZEN_BATCH_SIZE -e BATCH_SIZE -e VAL_BATCH_SIZE -e EVAL_BATCH_SIZE \
    -e FROZEN_ACCUMULATION_STEPS -e ACCUMULATION_STEPS \
    -e EARLY_STOPPING_PATIENCE \
    -v "$DATA_DIR:/app/data" \
    -v "$SRC_DIR:/app/src" \
    -v "$OUTPUT_DIR:/app/outputs" \
    -v "$SCRIPT:/app/runScript.sh" \
    -v "$TORCH_CACHE_DIR:/root/.cache/torch" \
    "$IMAGE_NAME" \
    bash /app/runScript.sh
