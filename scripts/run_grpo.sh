#!/bin/bash
# =============================================================================
# Geometry-R1 Phase 2: GRPO Training Script
# =============================================================================
#
# IMPORTANT: Run this script in tmux or with nohup for long training jobs!
#
# Using tmux (recommended):
#   tmux new -s grpo_train
#   ./scripts/run_grpo.sh
#   # Press Ctrl+B, then D to detach
#   # tmux attach -t grpo_train  # to reattach
#
# Using nohup:
#   nohup ./scripts/run_grpo.sh > logs/grpo_train.log 2>&1 &
#   tail -f logs/grpo_train.log
#
# =============================================================================

set -e

echo "=============================================="
echo "Geometry-R1 Phase 2: GRPO Training"
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
OUTPUT_DIR=${OUTPUT_DIR:-"output/grpo_phase2"}

# Model configuration - use SFT checkpoint as starting point
SFT_CHECKPOINT=${SFT_CHECKPOINT:-"output/sft_phase1"}
MODEL_NAME=${MODEL_NAME:-$SFT_CHECKPOINT}
CONFIG_FILE=${CONFIG_FILE:-"configs/grpo_config.yaml"}

# GRPO configuration
NUM_EPOCHS=${NUM_EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-2}
LEARNING_RATE=${LEARNING_RATE:-"1e-6"}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
TEMPERATURE=${TEMPERATURE:-"0.7"}

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  SFT checkpoint: $SFT_CHECKPOINT"
echo "  Data dir: $DATA_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Config: $CONFIG_FILE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Num generations: $NUM_GENERATIONS"
echo "  Temperature: $TEMPERATURE"
echo ""

# Verify SFT checkpoint exists
if [ ! -d "$SFT_CHECKPOINT" ]; then
    echo "Warning: SFT checkpoint not found at $SFT_CHECKPOINT"
    echo "Please run Phase 1 (SFT) training first."
    # Fall back to base model
    MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
    echo "Using base model: $MODEL_NAME"
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# -----------------------------------------------------------------------------
# Training Command
# -----------------------------------------------------------------------------
# TODO: Replace with actual training script when ready
# Current placeholder uses TRL GRPO API

echo "Starting GRPO training..."
echo ""

# Option 1: Using TRL GRPO Trainer (placeholder)
# trl grpo \
#     --model_name_or_path $MODEL_NAME \
#     --train_file $DATA_DIR/rl_geometry3k.jsonl \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs $NUM_EPOCHS \
#     --per_device_train_batch_size $BATCH_SIZE \
#     --learning_rate $LEARNING_RATE \
#     --num_generations $NUM_GENERATIONS \
#     --temperature $TEMPERATURE \
#     --bf16

# Option 2: Using accelerate launch (recommended for multi-GPU)
# accelerate launch \
#     --config_file configs/accelerate_config.yaml \
#     src/train_grpo.py \
#     --model_name_or_path $MODEL_NAME \
#     --train_file $DATA_DIR/rl_geometry3k.jsonl \
#     --output_dir $OUTPUT_DIR \
#     --reward_module src.reward \
#     --num_train_epochs $NUM_EPOCHS \
#     --per_device_train_batch_size $BATCH_SIZE \
#     --learning_rate $LEARNING_RATE

# Option 3: Direct Python script (placeholder)
python -c "
print('GRPO Training Placeholder')
print('==========================')
print('Implement your training logic in src/train_grpo.py')
print('Or use TRL GRPO Trainer as shown in run_grpo.sh')
print('')
print('Reward function: src/reward/reward.py')
print('  - accuracy_reward(): SymPy-based mathematical equivalence')
"

echo ""
echo "=============================================="
echo "GRPO Training Complete!"
echo "=============================================="
echo "Checkpoints saved to: $OUTPUT_DIR"
