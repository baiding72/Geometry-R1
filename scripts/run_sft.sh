#!/bin/bash
# =============================================================================
# Geometry-R1 Phase 1: SFT Training Script
# =============================================================================
#
# IMPORTANT: Run this script in tmux or with nohup for long training jobs!
#
# Using tmux (recommended):
#   tmux new -s sft_train
#   ./scripts/run_sft.sh
#   # Press Ctrl+B, then D to detach
#   # tmux attach -t sft_train  # to reattach
#
# Using nohup:
#   nohup ./scripts/run_sft.sh > logs/sft_train.log 2>&1 &
#   tail -f logs/sft_train.log
#
# =============================================================================

set -e

echo "=============================================="
echo "Geometry-R1 Phase 1: SFT Training"
echo "=============================================="

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
# Source environment variables if not set
if [ -z "$HF_ENDPOINT" ]; then
    export HF_ENDPOINT=https://hf-mirror.com
fi

# Default paths (override with environment variables)
DATA_DIR=${DATA_DIR:-"data"}
OUTPUT_DIR=${OUTPUT_DIR:-"output/sft_phase1"}

# Model configuration
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-VL-3B-Instruct"}
CONFIG_FILE=${CONFIG_FILE:-"configs/sft_config.yaml"}

# Training configuration
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-"2e-5"}

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Data dir: $DATA_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Config: $CONFIG_FILE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# -----------------------------------------------------------------------------
# Training Command
# -----------------------------------------------------------------------------
echo "Starting SFT training..."
echo ""

python -m src.train_sft \
    --config $CONFIG_FILE \
    --model_name_or_path $MODEL_NAME \
    --train_file $DATA_DIR/sft_geometry3k.jsonl \
    --output_dir $OUTPUT_DIR

echo ""
echo "=============================================="
echo "SFT Training Complete!"
echo "=============================================="
echo "Checkpoints saved to: $OUTPUT_DIR"
