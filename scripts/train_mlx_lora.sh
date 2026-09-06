#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"

# Generate a unique run ID for isolated experiments
run_id="${MLX_RUN_ID:-run_$(date -u +%Y%m%d_%H%M%S)_$$}"
base_run_dir="/tmp/model-lab-runs/${run_id}"

model_dir="${MLX_MODEL_DIR:-/tmp/model-lab-mlx/qwen2.5-coder-1.5b}"
data_dir="${MLX_DATA_DIR:-${base_run_dir}/data}"
adapter_dir="${MLX_ADAPTER_DIR:-${base_run_dir}/adapters}"
mlx_lora_bin="${MLX_LM_LORA_BIN:-mlx_lm.lora}"
allow_overwrite="${ALLOW_OVERWRITE:-0}"

validate_path() {
  local target_path="$1"
  local label="$2"

  if [[ -z "$target_path" ]]; then
    echo "Error: $label path cannot be empty." >&2
    exit 2
  fi

  local target_name
  target_name="$(basename "$target_path")"
  local parent_dir
  parent_dir="$(dirname "$target_path")"

  if [[ ! -d "$parent_dir" ]]; then
    mkdir -p "$parent_dir" 2>/dev/null || {
      echo "Error: Cannot create parent directory for $label: $parent_dir" >&2
      exit 2
    }
  fi

  local resolved_parent
  resolved_parent="$(cd "$parent_dir" && pwd -P)"
  local canonical="$resolved_parent/$target_name"

  if [[ "$canonical" == "/" ]]; then
    echo "Error: $label cannot be filesystem root (/)." >&2
    exit 2
  fi

  local home_dir
  home_dir="$(cd "$HOME" && pwd -P)"
  if [[ "$canonical" == "$home_dir" ]]; then
    echo "Error: $label cannot be user home directory ($home_dir)." >&2
    exit 2
  fi

  if [[ "$canonical" == "$repo_root" || "$repo_root" == "$canonical/"* ]]; then
    echo "Error: $label cannot be inside or contain the repository root ($repo_root)." >&2
    exit 2
  fi

  if [[ -L "$target_path" ]]; then
    echo "Error: $label cannot be a symlink: $target_path" >&2
    exit 2
  fi
}

validate_path "$model_dir" "MLX_MODEL_DIR"
validate_path "$data_dir" "MLX_DATA_DIR"
validate_path "$adapter_dir" "MLX_ADAPTER_DIR"

if [[ ! -d "$model_dir" ]]; then
  echo "Error: MLX base model directory not found: $model_dir" >&2
  echo "Convert the pinned base model first using prepare_mlx_model.sh." >&2
  exit 2
fi

# Collision checks
data_canon="$(cd "$(dirname "$data_dir")" && pwd -P)/$(basename "$data_dir")"
adapter_canon="$(cd "$(dirname "$adapter_dir")" && pwd -P)/$(basename "$adapter_dir")"
model_canon="$(cd "$(dirname "$model_dir")" && pwd -P)/$(basename "$model_dir")"

if [[ "$data_canon" == "$adapter_canon" ]]; then
  echo "Error: Training data directory and adapter directory cannot be the same ($data_canon)." >&2
  exit 2
fi

if [[ "$model_canon" == "$adapter_canon" ]]; then
  echo "Error: Base model directory and adapter directory cannot be the same ($model_canon)." >&2
  exit 2
fi

# Check LoRA executable
if [[ "$mlx_lora_bin" == */* ]]; then
  [[ -x "$mlx_lora_bin" ]] || {
    echo "Error: MLX LoRA executable not found or not executable: $mlx_lora_bin" >&2
    echo "Please install mlx-lm or activate your virtual environment." >&2
    exit 2
  }
else
  command -v "$mlx_lora_bin" >/dev/null 2>&1 || {
    echo "Error: MLX LoRA command  not found in PATH." >&2
    echo "Please install mlx-lm or set MLX_LM_LORA_BIN." >&2
    exit 2
  }
fi

# Prepare data without silent deletion of existing non-empty directory
if [[ -d "$data_dir" ]] && [[ -n "$(ls -A "$data_dir" 2>/dev/null)" ]]; then
  if [[ "$allow_overwrite" != "1" ]]; then
    echo "Error: Data directory exists and is not empty: $data_dir" >&2
    echo "Set ALLOW_OVERWRITE=1 to overwrite, or specify a unique MLX_DATA_DIR / MLX_RUN_ID." >&2
    exit 2
  fi
  rm -rf "$data_dir"
fi

mkdir -p "$data_dir"
python3 "$repo_root/scripts/prepare_mlx_data.py" >/dev/null
cp "$repo_root/data/mlx/train.jsonl" "$data_dir/train.jsonl"
cp "$repo_root/data/mlx/valid.jsonl" "$data_dir/valid.jsonl"

# Check adapter directory
if [[ -d "$adapter_dir" ]] && [[ -n "$(ls -A "$adapter_dir" 2>/dev/null)" ]]; then
  if [[ "$allow_overwrite" != "1" ]]; then
    echo "Error: Adapter directory exists and is not empty: $adapter_dir" >&2
    echo "Set ALLOW_OVERWRITE=1 to overwrite, or specify a unique MLX_ADAPTER_DIR / MLX_RUN_ID." >&2
    exit 2
  fi
  rm -rf "$adapter_dir"
fi

mkdir -p "$adapter_dir"

echo "Running MLX LoRA training in isolated directory: $adapter_dir"
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
