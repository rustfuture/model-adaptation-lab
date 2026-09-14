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

# 1. Determine run identifier and check existing run if MLX_RUN_ID is set
if [[ -n "${MLX_RUN_ID:-}" ]]; then
  if [[ ! "$MLX_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Error: MLX_RUN_ID must contain only alphanumeric characters, underscores, and dashes: $MLX_RUN_ID" >&2
    exit 2
  fi
  run_id="$MLX_RUN_ID"
  base_run_dir="/tmp/model-lab-runs/${run_id}"

  # Reject existing run directory if it has a manifest or is not empty
  if [[ -f "$base_run_dir/run_manifest.json" ]] || { [[ -d "$base_run_dir" ]] && [[ -n "$(ls -A "$base_run_dir" 2>/dev/null)" ]]; }; then
    echo "Error: Run directory or manifest already exists for explicit MLX_RUN_ID ($run_id): $base_run_dir" >&2
    echo "Refusing to overwrite or reuse existing run. Choose a fresh run ID." >&2
    exit 2
  fi
fi

model_dir="${MLX_MODEL_DIR:-/tmp/model-lab-mlx/qwen2.5-coder-1.5b}"
mlx_lora_bin="${MLX_LM_LORA_BIN:-mlx_lm.lora}"

# --- READ-ONLY VALIDATIONS (NO MUTATION BEFORE ALL CHECKS PASS) ---
validate_single_path "$model_dir" "MLX_MODEL_DIR"
model_canon="$(canonical_path "$model_dir")"

if [[ ! -d "$model_dir" ]]; then
  echo "Error: MLX base model directory not found: $model_dir" >&2
  echo "Convert the pinned base model first using prepare_mlx_model.sh." >&2
  exit 2
fi

# If explicit data_dir or adapter_dir provided, validate them before mutating anything
if [[ -n "${MLX_DATA_DIR:-}" ]]; then
  validate_single_path "$MLX_DATA_DIR" "MLX_DATA_DIR"
  data_canon="$(canonical_path "$MLX_DATA_DIR")"
  check_pair_conflict "$model_canon" "$data_canon" "Model directory" "Data directory"
fi

if [[ -n "${MLX_ADAPTER_DIR:-}" ]]; then
  validate_single_path "$MLX_ADAPTER_DIR" "MLX_ADAPTER_DIR"
  adapter_canon="$(canonical_path "$MLX_ADAPTER_DIR")"
  check_pair_conflict "$model_canon" "$adapter_canon" "Model directory" "Adapter directory"
fi

if [[ -n "${MLX_DATA_DIR:-}" && -n "${MLX_ADAPTER_DIR:-}" ]]; then
  check_pair_conflict "$data_canon" "$adapter_canon" "Data directory" "Adapter directory"
fi

if [[ -n "${MLX_DATA_DIR:-}" ]] && [[ -d "$MLX_DATA_DIR" ]] && [[ -n "$(ls -A "$MLX_DATA_DIR" 2>/dev/null)" ]]; then
  echo "Error: Data directory exists and is not empty: $MLX_DATA_DIR" >&2
  echo "Refusing to overwrite existing data. Please specify a clean directory." >&2
  exit 2
fi

if [[ -n "${MLX_ADAPTER_DIR:-}" ]] && [[ -d "$MLX_ADAPTER_DIR" ]] && [[ -n "$(ls -A "$MLX_ADAPTER_DIR" 2>/dev/null)" ]]; then
  echo "Error: Adapter directory exists and is not empty: $MLX_ADAPTER_DIR" >&2
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

# --- PRE-FLIGHT: Explicit vs Default path collision check ---
# When MLX_RUN_ID is set, we know base_run_dir ahead of time.
# Check if any explicit path collides with a would-be default path.
if [[ -n "${MLX_RUN_ID:-}" ]]; then
  projected_data_default="$(canonical_path "${base_run_dir}/data")"
  projected_adapter_default="$(canonical_path "${base_run_dir}/adapters")"

  # If MLX_DATA_DIR is set but MLX_ADAPTER_DIR is not:
  # explicit data_dir vs projected default adapter_dir
  if [[ -n "${MLX_DATA_DIR:-}" && -z "${MLX_ADAPTER_DIR:-}" ]]; then
    check_pair_conflict "$data_canon" "$projected_adapter_default" "Data directory" "Adapter directory (default)"
  fi

  # If MLX_ADAPTER_DIR is set but MLX_DATA_DIR is not:
  # explicit adapter_dir vs projected default data_dir
  if [[ -z "${MLX_DATA_DIR:-}" && -n "${MLX_ADAPTER_DIR:-}" ]]; then
    check_pair_conflict "$projected_data_default" "$adapter_canon" "Data directory (default)" "Adapter directory"
  fi

  # Also check explicit paths vs base_run_dir itself for nesting
  projected_base_canon="$(canonical_path "$base_run_dir")"
  if [[ -n "${MLX_DATA_DIR:-}" ]]; then
    if [[ "$data_canon" == "$projected_base_canon" ]]; then
      echo "Error: MLX_DATA_DIR ($data_canon) cannot be the run directory itself." >&2
      exit 2
    fi
  fi
  if [[ -n "${MLX_ADAPTER_DIR:-}" ]]; then
    if [[ "$adapter_canon" == "$projected_base_canon" ]]; then
      echo "Error: MLX_ADAPTER_DIR ($adapter_canon) cannot be the run directory itself." >&2
      exit 2
    fi
  fi
fi

# --- ALL PRE-FLIGHT VALIDATIONS PASSED: NOW ALLOCATE RUN DIRECTORY ---
if [[ -n "${MLX_RUN_ID:-}" ]]; then
  # run_id and base_run_dir were assigned and validated above
  mkdir -p /tmp/model-lab-runs
  if ! mkdir "$base_run_dir"; then
    echo "Error: Run directory already exists or cannot be created; ownership was not acquired: $base_run_dir" >&2
    exit 2
  fi
else
  mkdir -p /tmp/model-lab-runs
  base_run_dir="$(mktemp -d /tmp/model-lab-runs/run-XXXXXX)"
  run_id="$(basename "$base_run_dir")"
fi

run_dir_owner=true
cleanup_run_dir() {
  if [[ "${run_dir_owner:-false}" == "true" && -n "${base_run_dir:-}" && -d "$base_run_dir" ]]; then
    rm -rf "$base_run_dir"
  fi
}
trap cleanup_run_dir EXIT

data_dir="${MLX_DATA_DIR:-${base_run_dir}/data}"
adapter_dir="${MLX_ADAPTER_DIR:-${base_run_dir}/adapters}"

validate_single_path "$data_dir" "MLX_DATA_DIR"
validate_single_path "$adapter_dir" "MLX_ADAPTER_DIR"

data_canon="$(canonical_path "$data_dir")"
adapter_canon="$(canonical_path "$adapter_dir")"

# Check all pairs for equality and directory nesting
check_pair_conflict "$model_canon" "$data_canon" "Model directory" "Data directory"
check_pair_conflict "$model_canon" "$adapter_canon" "Model directory" "Adapter directory"
check_pair_conflict "$data_canon" "$adapter_canon" "Data directory" "Adapter directory"

mkdir -p "$data_dir"
mkdir -p "$adapter_dir"

python3 "$repo_root/scripts/prepare_mlx_data.py" >/dev/null
cp "$repo_root/data/mlx/train.jsonl" "$data_dir/train.jsonl"
cp "$repo_root/data/mlx/valid.jsonl" "$data_dir/valid.jsonl"
if [[ -e "$data_dir/test.jsonl" || -L "$data_dir/test.jsonl" ]]; then
  echo "Error: Test split must remain outside the training data directory: $data_dir/test.jsonl" >&2
  exit 2
fi

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

# A successful process exit is not sufficient evidence of an adapter. Refuse
# to publish a completed manifest unless MLX emitted the expected weight file.
if [[ ! -s "$adapter_dir/adapters.safetensors" ]]; then
  echo "Error: MLX training finished without a non-empty adapters.safetensors file: $adapter_dir" >&2
  echo "Refusing to write a completed run manifest without adapter weights." >&2
  exit 2
fi

# Write run manifest for subsequent evaluation
manifest_file="$base_run_dir/run_manifest.json"
python3 -c "
import hashlib, importlib.metadata, json, os, platform, sys, time
from pathlib import Path

repo_root = Path(sys.argv[8])
dataset_path = repo_root / 'data' / 'rust_errors.jsonl'
dataset_bytes = dataset_path.read_bytes()
records = [json.loads(line) for line in dataset_bytes.decode('utf-8').splitlines() if line.strip()]
split_records = {split: [r for r in records if r['split'] == split] for split in ('train', 'validation', 'test')}
def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not_available'

adapter_root = Path(sys.argv[4])
weight_files = []
for path in sorted(adapter_root.rglob('*')):
    if path.is_file() and not path.is_symlink():
        weight_files.append({
            'path': path.relative_to(adapter_root).as_posix(),
            'size_bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        })
manifest = {
    'schema_version': 'model-adaptation-lab.run-manifest.v1',
    'run_id': sys.argv[1],
    'timestamp_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'model_dir': sys.argv[2],
    'data_dir': sys.argv[3],
    'adapter_dir': sys.argv[4],
    'iters': int(sys.argv[5]),
    'learning_rate': float(sys.argv[6]),
    'fine_tune_type': 'lora',
    'seed': int(os.environ.get('MLX_SEED', '42')),
    'base_model_revision': os.environ.get('MODEL_REVISION'),
    'dataset': {
        'path': 'data/rust_errors.jsonl',
        'sha256': hashlib.sha256(dataset_bytes).hexdigest(),
        'split_counts': {split: len(rows) for split, rows in split_records.items()},
        'split_record_ids': {split: [r['id'] for r in rows] for split, rows in split_records.items()},
        'training_data_policy': 'Only train and validation records are copied; test remains held out.',
    },
    'environment': {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'mlx_version': package_version('mlx'),
        'mlx_lm_version': package_version('mlx-lm'),
    },
    'artifacts': {
        'adapter_weights': weight_files,
        'adapter_weights_preserved_in_run_dir': True,
    },
}
with open(sys.argv[7], 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)
print(f'Wrote run manifest: {sys.argv[7]}')
" "$run_id" "$model_canon" "$data_canon" "$adapter_canon" "${MLX_ITERS:-50}" "${MLX_LEARNING_RATE:-1e-4}" "$manifest_file" "$repo_root"

echo "Training complete. Run manifest generated at: $manifest_file"

# Success — disable cleanup trap so the run directory is preserved
run_dir_owner=false
