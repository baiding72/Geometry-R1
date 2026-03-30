# Geometry-R1

**Geometry-R1** is a two-stage training pipeline for teaching vision-language models to solve geometry problems. Using Qwen2.5-VL-3B-Instruct as the base model and Hugging Face TRL framework, it combines Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO) for mathematical reasoning.

The repository now includes:
- End-to-end SFT training entrypoint: `python -m src.train_sft`
- End-to-end GRPO training entrypoint: `python -m src.train_grpo`
- Offline evaluation entrypoint: `python -m src.eval`
- Unit tests for reward logic: `python -m unittest tests/test_reward.py`

## Architecture Overview

### Phase 1: Cold-Start SFT

The first phase uses **distillation from DashScope teacher models** to create training data:
- Dataset: First 500 samples from `hiyouga/geometry3k`
- Base recommendation: `qwen3-vl-plus`
- Stronger but more expensive option: `qwen3-vl-235b-a22b-instruct`
- Method: Call DashScope multimodal API to generate pseudo chain-of-thought
- Format: `½...½ → <answer>...</answer>` structured reasoning

### Phase 2: GRPO-based Reinforcement Learning

The second phase uses **rule-based rewards** for self-improvement:
- Dataset: Remaining samples from `hiyouga/geometry3k`
- Reward: **SymPy-based mathematical equivalence** verification
- Method: Compare model predictions with ground truth symbolically

## Project Structure

```
Geometry-R1/
├── configs/
│   ├── sft_config.yaml       # Phase 1 SFT hyperparameters
│   └── grpo_config.yaml      # Phase 2 GRPO hyperparameters
├── data/
│   ├── images/
│   │   ├── sft/              # SFT images
│   │   └── rl/               # RL images
│   ├── sft_geometry3k.jsonl  # SFT training data
│   └── rl_geometry3k.jsonl   # RL training data
├── output/                   # Model checkpoints (git-ignored)
├── scripts/
│   ├── prepare_geometry3k.py # Data preparation script
│   ├── setup_autodl.sh       # AutoDL environment setup
│   ├── run_sft.sh            # SFT training launcher
│   └── run_grpo.sh           # GRPO training launcher
├── src/
│   ├── __init__.py
│   ├── config.py             # Path configuration module
│   ├── eval.py               # Offline evaluation script
│   ├── train_sft.py          # SFT training entrypoint
│   ├── train_grpo.py         # GRPO training entrypoint
│   ├── train_utils.py        # Shared training/evaluation utilities
│   └── reward/
│       ├── __init__.py
│       └── reward.py         # SymPy-based reward functions
├── tests/
│   └── test_reward.py        # Unit tests for reward parsing/equivalence
├── pyproject.toml            # Project configuration (uv)
└── README.md
```

## Installation

### Local (Mac/Linux)

```bash
# Clone the repository
git clone <your-repo-url>
cd Geometry-R1

# Install dependencies with uv
uv sync
```

### AutoDL Platform

```bash
# Clone the repository
git clone <your-repo-url>
cd Geometry-R1

# Run setup script
chmod +x scripts/setup_autodl.sh
./scripts/setup_autodl.sh
```

## Configuration

### Path Configuration

All paths are managed through `src/config.py` and support environment variable overrides:

```python
from src.config import get_data_dir, get_output_dir, setup_directories

# Get configured paths
print(get_data_dir())     # -> data/ or $DATA_DIR
print(get_output_dir())   # -> output/ or $OUTPUT_DIR
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_ROOT` | (auto-detect) | Project root directory |
| `DATA_DIR` | `data/` | Data directory for datasets and images |
| `OUTPUT_DIR` | `output/` | Output directory for checkpoints |
| `HF_ENDPOINT` | - | HuggingFace mirror URL (for China) |
| `HF_HOME` | `.cache/` | HuggingFace cache directory |
| `DASHSCOPE_API_KEY` | - | DashScope API key for distillation |

### AutoDL Configuration

For AutoDL, data and models should be stored on the data disk:

```bash
# Set environment variables
export HF_ENDPOINT=https://hf-mirror.com
export DATA_DIR=/root/autodl-tmp/data
export OUTPUT_DIR=/root/autodl-tmp/output
export HF_HOME=/root/autodl-tmp/hf_cache
```

Or use the setup script:
```bash
./scripts/setup_autodl.sh  # Automatically sets these variables
```

## Data Preparation

### Step 1: Set up DashScope API Key

```bash
# Option 1: Environment variable
export DASHSCOPE_API_KEY="your_api_key_here"

# Option 2: Command line argument
# --api-key "your_api_key_here"
```

### Step 2: Prepare geometry3k Dataset

```bash
# Local (mock mode for testing)
uv run python scripts/prepare_geometry3k.py --sft-samples 500

# With real API distillation
uv run python scripts/prepare_geometry3k.py --sft-samples 500 --use-api

# Explicitly choose the teacher model
uv run python scripts/prepare_geometry3k.py \
  --sft-samples 500 \
  --use-api \
  --teacher-model qwen3-vl-plus

# AutoDL (custom data directory)
export DATA_DIR=/root/autodl-tmp/data
python scripts/prepare_geometry3k.py --sft-samples 500 --use-api
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--sft-samples` | 500 | Number of samples for SFT split |
| `--data-dir` | `data/` | Data directory (overrides `DATA_DIR` env var) |
| `--use-api` | False | Use DashScope multimodal API for pseudo-CoT generation |
| `--api-key` | None | DashScope API key (or set env var) |
| `--teacher-model` | `qwen3-vl-plus` | DashScope teacher model used for distillation |

## Training

### Phase 1: SFT Training

```bash
# Run directly
python -m src.train_sft \
  --config configs/sft_config.yaml \
  --train_file data/sft_geometry3k.jsonl \
  --output_dir output/sft_phase1

# Or use the helper script
./scripts/run_sft.sh
```

### Phase 2: GRPO Training

```bash
# Run directly
python -m src.train_grpo \
  --config configs/grpo_config.yaml \
  --model_name_or_path output/sft_phase1 \
  --train_file data/rl_geometry3k.jsonl \
  --output_dir output/grpo_phase2

# Or use the helper script
./scripts/run_grpo.sh
```

### Evaluation

```bash
python -m src.eval \
  --model_name_or_path output/grpo_phase2 \
  --data_file data/rl_geometry3k.jsonl \
  --output_file output/eval_grpo_phase2.json \
  --max_samples 128
```

The evaluation script reports:
- `format_rate`: fraction of generations that contain a valid `<answer>...</answer>` block
- `exact_match`: exact string match on extracted answers
- `symbolic_accuracy`: symbolic equivalence accuracy using SymPy-backed verification

## Reward Function

Phase 2 uses a **SymPy-based reward function** that:

1. Extracts answer from `<answer>...</answer>` tags
2. Normalizes LaTeX strings
3. Converts common LaTeX constructs to SymPy-friendly expressions when the LaTeX parser is unavailable
4. Compares mathematical equivalence:
   - Direct string matching
   - Symbolic simplification (`simplify(pred - gt) == 0`)
   - Numerical evaluation fallback

```python
from src.reward import accuracy_reward

rewards = accuracy_reward(
    prompts=["Q1", "Q2"],
    completions=[
        "½ Reasoning... ½ <answer>\\frac{12}{5}</answer>",
        "<answer>5</answer>"
    ],
    answer=["\\frac{24}{10}", "6"]  # Ground truth
)
# Returns: [1.0, 0.0]  (12/5 == 24/10, but 5 != 6)
```

### Supported Equivalence Checks

| Predicted | Ground Truth | Equivalent |
|-----------|-------------|------------|
| `12` | `12` | ✓ |
| `\frac{12}{5}` | `\frac{12}{5}` | ✓ |
| `\frac{24}{10}` | `\frac{12}{5}` | ✓ (symbolic) |
| `\sqrt{2}` | `$\sqrt{2}$` | ✓ |
| `5` | `6` | ✗ |

## Development

### Code Formatting

```bash
# Format with Ruff
uv run ruff format .

# Check linting
uv run ruff check .
```

### Run Unit Tests

```bash
python -m unittest tests/test_reward.py
```

### Print Current Configuration

```bash
python -c "from src.config import print_config; print_config()"
```

## Dependencies

- **Core**: torch, transformers, accelerate, peft, trl
- **Vision**: qwen-vl-utils, pillow
- **Data**: datasets
- **Math**: sympy, antlr4-python3-runtime (4.11.1)
- **API**: dashscope (for Qwen-VL-Max distillation)
- **Utils**: pyyaml, tqdm

## License

MIT License

## Acknowledgments

- Base model: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- Dataset: [hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)
- Training framework: [TRL](https://github.com/huggingface/trl)
- Distillation API: [DashScope Vision Models](https://help.aliyun.com/zh/model-studio/vision)
