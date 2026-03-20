#!/bin/bash
# =============================================================================
# Geometry-R1 AutoDL Environment Setup Script
# =============================================================================
#
# This script initializes the environment for training on AutoDL platform.
# Run this script after connecting to your AutoDL instance.
#
# Usage:
#   chmod +x scripts/setup_autodl.sh
#   ./scripts/setup_autodl.sh
#
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "Geometry-R1 AutoDL Environment Setup"
echo "=============================================="

# -----------------------------------------------------------------------------
# 1. Set up environment variables for Chinese mirrors
# -----------------------------------------------------------------------------
echo "[1/5] Setting up environment variables..."

# Hugging Face mirror (critical for AutoDL in China)
export HF_ENDPOINT=https://hf-mirror.com
echo "  HF_ENDPOINT: $HF_ENDPOINT"

# Data and output directories (AutoDL data disk)
export DATA_DIR=/root/autodl-tmp/data
export OUTPUT_DIR=/root/autodl-tmp/output
export HF_HOME=/root/autodl-tmp/hf_cache

echo "  DATA_DIR: $DATA_DIR"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  HF_HOME: $HF_HOME"

# Add to bashrc for persistence
if ! grep -q "HF_ENDPOINT" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Geometry-R1 environment variables" >> ~/.bashrc
    echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc
    echo "export DATA_DIR=/root/autodl-tmp/data" >> ~/.bashrc
    echo "export OUTPUT_DIR=/root/autodl-tmp/output" >> ~/.bashrc
    echo "export HF_HOME=/root/autodl-tmp/hf_cache" >> ~/.bashrc
    echo "  Added environment variables to ~/.bashrc"
fi

# -----------------------------------------------------------------------------
# 2. Create directories
# -----------------------------------------------------------------------------
echo "[2/5] Creating directories..."

mkdir -p $DATA_DIR/images/sft
mkdir -p $DATA_DIR/images/rl
mkdir -p $OUTPUT_DIR
mkdir -p $HF_HOME

echo "  Created: $DATA_DIR"
echo "  Created: $OUTPUT_DIR"
echo "  Created: $HF_HOME"

# -----------------------------------------------------------------------------
# 3. Install system dependencies
# -----------------------------------------------------------------------------
echo "[3/5] Installing system dependencies..."

# Update pip
pip install --upgrade pip -q

# Install additional system packages if needed
# apt-get update && apt-get install -y <package>

# -----------------------------------------------------------------------------
# 4. Install Python dependencies
# -----------------------------------------------------------------------------
echo "[4/5] Installing Python dependencies..."

# Check if using uv or pip
if command -v uv &> /dev/null; then
    echo "  Using uv for package management..."
    uv sync
else
    echo "  Using pip for package management..."
    pip install -e .
fi

# Install antlr4 for SymPy LaTeX parsing
pip install antlr4-python3-runtime==4.11.1 -q

# -----------------------------------------------------------------------------
# 5. Verify installation
# -----------------------------------------------------------------------------
echo "[5/5] Verifying installation..."

python -c "
import torch
import transformers
import trl
from src.config import get_data_dir, get_output_dir

print(f'  PyTorch version: {torch.__version__}')
print(f'  Transformers version: {transformers.__version__}')
print(f'  TRL version: {trl.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'  Data directory: {get_data_dir()}')
print(f'  Output directory: {get_output_dir()}')
"

echo ""
echo "=============================================="
echo "Environment setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Prepare data: python scripts/prepare_geometry3k.py --use-api"
echo "  2. Run SFT: ./scripts/run_sft.sh"
echo "  3. Run GRPO: ./scripts/run_grpo.sh"
echo ""
