#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"

echo "Starting v2 experiment..."

# 1. Validate dataset
echo "Validating v2 dataset..."
python3 "$repo_root/scripts/validate_dataset_v2.py"

# 2. Validate snippets
echo "Validating v2 snippets..."
python3 "$repo_root/scripts/validate_rustc_snippets_v2.py"

# 3. Check for model
if [[ ! -d "/tmp/model-lab-mlx/qwen2.5-coder-1.5b" ]]; then
    echo "Model weights are unavailable. Training run is blocked."
    echo "To run, ensure you have the MLX model downloaded and converted at /tmp/model-lab-mlx/qwen2.5-coder-1.5b"
    exit 0
fi

echo "Running v2 MLX Base Evaluation..."
python3 "$repo_root/scripts/evaluate_mlx_v2.py" --model /tmp/model-lab-mlx/qwen2.5-coder-1.5b --output-dir reports/v2-eval --baseline-only

echo "Running v2 MLX Training..."
export MLX_ITERS=50
export MLX_LEARNING_RATE=1e-4
"$repo_root/scripts/train_mlx_lora_v2.sh" /tmp/model-lab-mlx/qwen2.5-coder-1.5b

run_dir=$(ls -td "$repo_root/run-"* | head -1)
adapters_dir="$run_dir/adapters"

echo "Running v2 MLX LoRA Evaluation..."
python3 "$repo_root/scripts/evaluate_mlx_v2.py" --model /tmp/model-lab-mlx/qwen2.5-coder-1.5b --adapter "$adapters_dir" --output-dir reports/v2-eval

echo "Experiment complete."
