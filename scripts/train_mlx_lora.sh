#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"

canonical_path() {
  python3 -c "import os, sys; print(os.path.realpath(os.path.abspath(sys.argv[1])))" "$1"
}

home_canon="$(canonical_path "$HOME")"
repo_canon="$(canonical_path "$repo_root")"

validate_single_path() {
  local target_path="$1"
  local label="$2"

  if [[ -z "$target_path" ]]; then
    echo "Error: $label path cannot be empty." >&2
    exit 2
  fi

  if [[ -L "$target_path" ]]; then
    echo "Error: $label cannot be a symlink: $target_path" >&2
    exit 2
  fi

  local canon
  canon="$(canonical_path "$target_path")"

  if [[ "$canon" == "/" ]]; then
    echo "Error: $label cannot be filesystem root (/)." >&2
    exit 2
  fi

  if [[ "$canon" == "$home_canon" ]]; then
    echo "Error: $label cannot be user home directory ($home_canon)." >&2
    exit 2
  fi

  if [[ "$canon" == "$repo_canon" || "$canon" == "$repo_canon/"* || "$repo_canon" == "$canon/"* ]]; then
    echo "Error: $label cannot be inside or contain the repository root ($repo_canon)." >&2
    exit 2
  fi
}

check_pair_conflict() {
  local p1="$1"
  local p2="$2"
  local l1="$3"
  local l2="$4"

  if [[ "$p1" == "$p2" ]]; then
    echo "Error: $l1 ($p1) and $l2 ($p2) cannot be the same directory." >&2
    exit 2
  fi

  if [[ "$p1/" == "$p2/"* ]]; then
    echo "Error: $l1 ($p1) cannot be inside $l2 ($p2)." >&2
    exit 2
  fi

  if [[ "$p2/" == "$p1/"* ]]; then
    echo "Error: $l2 ($p2) cannot be inside $l1 ($p1)." >&2
    exit 2
  fi
}

# 1. Determine run directory and identifier safely
if [[ -n "${MLX_RUN_ID:-}" ]]; then
  if [[ ! "$MLX_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Error: MLX_RUN_ID must contain only alphanumeric characters, underscores, and dashes: $MLX_RUN_ID" >&2
    exit 2
  fi
  run_id="$MLX_RUN_ID"
  base_run_dir="/tmp/model-lab-runs/${run_id}"
else
  mkdir -p /tmp/model-lab-runs
  base_run_dir="$(mktemp -d /tmp/model-lab-runs/run-XXXXXX)"
  run_id="$(basename "$base_run_dir")"
fi

model_dir="${MLX_MODEL_DIR:-/tmp/model-lab-mlx/qwen2.5-coder-1.5b}"
data_dir="${MLX_DATA_DIR:-${base_run_dir}/data}"
adapter_dir="${MLX_ADAPTER_DIR:-${base_run_dir}/adapters}"
mlx_lora_bin="${MLX_LM_LORA_BIN:-mlx_lm.lora}"

# --- READ-ONLY VALIDATIONS (NO MUTATION BEFORE ALL CHECKS PASS) ---
validate_single_path "$model_dir" "MLX_MODEL_DIR"
validate_single_path "$data_dir" "MLX_DATA_DIR"
validate_single_path "$adapter_dir" "MLX_ADAPTER_DIR"

model_canon="$(canonical_path "$model_dir")"
data_canon="$(canonical_path "$data_dir")"
adapter_canon="$(canonical_path "$adapter_dir")"

# Check all pairs for equality and directory nesting
check_pair_conflict "$model_canon" "$data_canon" "Model directory" "Data directory"
check_pair_conflict "$model_canon" "$adapter_canon" "Model directory" "Adapter directory"
check_pair_conflict "$data_canon" "$adapter_canon" "Data directory" "Adapter directory"

if [[ ! -d "$model_dir" ]]; then
  echo "Error: MLX base model directory not found: $model_dir" >&2
  echo "Convert the pinned base model first using prepare_mlx_model.sh." >&2
  exit 2
fi

# Reject existing non-empty target directories
if [[ -d "$data_dir" ]] && [[ -n "$(ls -A "$data_dir" 2>/dev/null)" ]]; then
  echo "Error: Data directory exists and is not empty: $data_dir" >&2
  echo "Refusing to overwrite existing data. Please specify a clean directory." >&2
  exit 2
fi

if [[ -d "$adapter_dir" ]] && [[ -n "$(ls -A "$adapter_dir" 2>/dev/null)" ]]; then
  echo "Error: Adapter directory exists and is not empty: $adapter_dir" >&2
  echo "Refusing to overwrite existing adapter. Please specify a clean directory." >&2
  exit 2
fi

# LoRA binary check
if [[ "$mlx_lora_bin" == */* ]]; then
  [[ -x "$mlx_lora_bin" ]] || {
    echo "Error: MLX LoRA executable not found or not executable: $mlx_lora_bin" >&2
    echo "Please install mlx-lm or activate your virtual environment." >&2
    exit 2
  }
else
  command -v "$mlx_lora_bin" >/dev/null 2>&1 || {
    echo "Error: MLX LoRA command not found in PATH: $mlx_lora_bin" >&2
    echo "Please install mlx-lm or set MLX_LM_LORA_BIN." >&2
    exit 2
  }
fi

# --- MUTATIONS BEGIN ONLY AFTER ALL VALIDATIONS PASS ---
mkdir -p "$base_run_dir"
mkdir -p "$data_dir"
mkdir -p "$adapter_dir"

python3 "$repo_root/scripts/prepare_mlx_data.py" >/dev/null
cp "$repo_root/data/mlx/train.jsonl" "$data_dir/train.jsonl"
cp "$repo_root/data/mlx/valid.jsonl" "$data_dir/valid.jsonl"

echo "Running MLX LoRA training in isolated directory: $adapter_dir"
"$mlx_lora_bin" \
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

# Write run manifest for subsequent evaluation
manifest_file="$base_run_dir/run_manifest.json"
python3 -c "
import json, time, sys
manifest = {
    'run_id': sys.argv[1],
    'timestamp_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'model_dir': sys.argv[2],
    'data_dir': sys.argv[3],
    'adapter_dir': sys.argv[4],
    'iters': int(sys.argv[5]),
    'learning_rate': float(sys.argv[6]),
    'fine_tune_type': 'lora'
}
with open(sys.argv[7], 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)
print(f'Wrote run manifest: {sys.argv[7]}')
" "$run_id" "$model_canon" "$data_canon" "$adapter_canon" "${MLX_ITERS:-50}" "${MLX_LEARNING_RATE:-1e-4}" "$manifest_file"

echo "Training complete. Run manifest generated at: $manifest_file"
