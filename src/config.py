"""
Geometry-R1 Configuration Module

Centralized configuration management for cross-platform compatibility.
Supports environment variable overrides for AutoDL deployment.
"""

import os
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    # Try environment variable first (for AutoDL)
    if os.environ.get("PROJECT_ROOT"):
        return Path(os.environ["PROJECT_ROOT"])

    # Otherwise, find project root relative to this file
    # src/config.py -> project root is parent of src/
    return Path(__file__).parent.parent.resolve()


def get_data_dir() -> Path:
    """
    Get the data directory.

    Priority:
    1. DATA_DIR environment variable (for AutoDL: /root/autodl-tmp/data)
    2. Project root / data
    """
    if os.environ.get("DATA_DIR"):
        return Path(os.environ["DATA_DIR"])
    return get_project_root() / "data"


def get_output_dir() -> Path:
    """
    Get the output directory for model checkpoints.

    Priority:
    1. OUTPUT_DIR environment variable (for AutoDL: /root/autodl-tmp/output)
    2. Project root / output
    """
    if os.environ.get("OUTPUT_DIR"):
        return Path(os.environ["OUTPUT_DIR"])
    return get_project_root() / "output"


def get_cache_dir() -> Path:
    """
    Get the cache directory for Hugging Face datasets and models.

    Priority:
    1. HF_HOME / HF_CACHE_DIR environment variable
    2. Project root / .cache
    """
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"])
    if os.environ.get("HF_CACHE_DIR"):
        return Path(os.environ["HF_CACHE_DIR"])
    return get_project_root() / ".cache"


def get_model_name_or_path(model_name: str = "Qwen/Qwen2-VL-2B-Instruct") -> str:
    """
    Get the model path, supporting local checkpoints.

    Args:
        model_name: HuggingFace model name or local path

    Returns:
        Model path (local path if exists, otherwise HuggingFace model name)
    """
    # Check if it's a local path
    local_path = get_output_dir() / model_name
    if local_path.exists():
        return str(local_path)

    # Check if model_name is already a valid local path
    if Path(model_name).exists():
        return model_name

    return model_name


def get_sft_data_path() -> Path:
    """Get the SFT training data path."""
    return get_data_dir() / "sft_geometry3k.jsonl"


def get_rl_data_path() -> Path:
    """Get the RL training data path."""
    return get_data_dir() / "rl_geometry3k.jsonl"


def get_sft_image_dir() -> Path:
    """Get the SFT images directory."""
    return get_data_dir() / "images" / "sft"


def get_rl_image_dir() -> Path:
    """Get the RL images directory."""
    return get_data_dir() / "images" / "rl"


def get_config_path(config_name: str) -> Path:
    """Get the path to a config file."""
    return get_project_root() / "configs" / config_name


def setup_directories() -> None:
    """Create all necessary directories."""
    dirs = [
        get_data_dir(),
        get_output_dir(),
        get_cache_dir(),
        get_sft_image_dir(),
        get_rl_image_dir(),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def print_config() -> None:
    """Print current configuration for debugging."""
    print("=" * 60)
    print("Geometry-R1 Configuration")
    print("=" * 60)
    print(f"Project Root: {get_project_root()}")
    print(f"Data Dir:     {get_data_dir()}")
    print(f"Output Dir:   {get_output_dir()}")
    print(f"Cache Dir:    {get_cache_dir()}")
    print(f"SFT Data:     {get_sft_data_path()}")
    print(f"RL Data:      {get_rl_data_path()}")
    print("=" * 60)


# Environment variables reference for AutoDL:
# export PROJECT_ROOT=/root/Geometry-R1
# export DATA_DIR=/root/autodl-tmp/data
# export OUTPUT_DIR=/root/autodl-tmp/output
# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HOME=/root/autodl-tmp/hf_cache

if __name__ == "__main__":
    print_config()
