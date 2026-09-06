#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${MLX_MODEL_DIR:-/tmp/model-lab-mlx/qwen2.5-coder-1.5b}"
data_dir="${MLX_DATA_DIR:-/tmp/model-lab-data}"
adapter_dir="${MLX_ADAPTER_DIR:-/tmp/model-lab-adapters}"
mlx_lora_bin="${MLX_LM_LORA_BIN:-mlx_lm.lora}"

if [[ ! -d "$model_dir" ]]; then
  echo "MLX model directory not found: $model_dir" >&2
  echo "Convert the pinned base model first; see README.md and training-manifest.json." >&2
  exit 2
fi

if [[ "$mlx_lora_bin" == */* ]]; then
  [[ -x "$mlx_lora_bin" ]] || {
    echo "MLX LoRA executable not found: $mlx_lora_bin" >&2
    echo "Install a compatible mlx-lm release in an isolated environment." >&2
    exit 2
  }
else
  command -v "$mlx_lora_bin" >/dev/null 2>&1 || {
    echo "MLX LoRA command not found: $mlx_lora_bin" >&2
    echo "Install a compatible mlx-lm release in an isolated environment." >&2
    exit 2
  }
fi

python3 "$repo_root/scripts/prepare_mlx_data.py" >/dev/null
rm -rf "$data_dir"
mkdir -p "$data_dir"
cp "$repo_root/data/mlx/train.jsonl" "$data_dir/train.jsonl"
cp "$repo_root/data/mlx/valid.jsonl" "$data_dir/valid.jsonl"
rm -rf "$adapter_dir"
mkdir -p "$adapter_dir"

exec "$mlx_lora_bin" \
  --model "$model_dir" \
  --train \
  --data "$data_dir" \
  --fine-tune-type lora \
  --mask-prompt \
  --batch-size 1 \
  --iters "${MLX_ITERS:-50}" \
  --val-batches -1 \
  --learning-rate "${MLX_LEARNING_RATE:-1e-4}" \
  --steps-per-report 5 \
  --steps-per-eval 10 \
  --save-every 25 \
  --adapter-path "$adapter_dir" \
  --seed "${MLX_SEED:-42}" \
  --max-seq-length 512
