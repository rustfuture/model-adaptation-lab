#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"

model_repo="${HF_MODEL_REPO:-https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct.git}"
model_revision="${MODEL_REVISION:-2e1fd397ee46e1388853d2af2c993145b0f1098a}"
hf_model_dir="${HF_MODEL_DIR:-/tmp/model-lab-hf/qwen2.5-coder-1.5b-instruct}"
mlx_model_dir="${MLX_MODEL_DIR:-/tmp/model-lab-mlx/qwen2.5-coder-1.5b}"
mlx_convert_bin="${MLX_CONVERT_BIN:-mlx_lm.convert}"

if [[ ! "$model_revision" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "Error: MODEL_REVISION must be a full 40-character hexadecimal commit SHA: $model_revision" >&2
  exit 2
fi

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
    echo "Error: $label cannot be the filesystem root (/)." >&2
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

# --- READ-ONLY VALIDATIONS (NO MUTATION BEFORE ALL CHECKS PASS) ---
validate_single_path "$hf_model_dir" "HF_MODEL_DIR"
validate_single_path "$mlx_model_dir" "MLX_MODEL_DIR"

hf_canon="$(canonical_path "$hf_model_dir")"
mlx_canon="$(canonical_path "$mlx_model_dir")"
check_pair_conflict "$hf_canon" "$mlx_canon" "HF source directory" "MLX destination directory"

# Destination safety: never delete or overwrite existing non-empty directory
if [[ -d "$mlx_model_dir" ]] && [[ -n "$(ls -A "$mlx_model_dir" 2>/dev/null)" ]]; then
  echo "Error: Destination MLX model directory exists and is not empty: $mlx_model_dir" >&2
  echo "Destructive deletion and in-place overwriting are disabled to protect existing models." >&2
  echo "Please specify a clean, empty destination directory via MLX_MODEL_DIR." >&2
  exit 2
fi

# Source safety: protect existing dirty checkouts
if [[ -d "$hf_model_dir/.git" ]]; then
  if [[ -n "$(git -C "$hf_model_dir" status --porcelain 2>/dev/null)" ]]; then
    echo "Error: Existing Hugging Face checkout at $hf_model_dir contains uncommitted changes." >&2
    echo "Refusing to modify or checkout to protect user work. Specify a different HF_MODEL_DIR." >&2
    exit 2
  fi
fi

# Converter binary check
if [[ "$mlx_convert_bin" == */* ]]; then
  [[ -x "$mlx_convert_bin" ]] || {
    echo "Error: MLX converter executable not found: $mlx_convert_bin" >&2
    echo "Please install mlx-lm or activate your Python virtualenv." >&2
    exit 2
  }
else
  command -v "$mlx_convert_bin" >/dev/null 2>&1 || {
    echo "Error: MLX converter command not found in PATH: $mlx_convert_bin" >&2
    echo "Please install mlx-lm or set MLX_CONVERT_BIN." >&2
    exit 2
  }
fi

# --- MUTATIONS BEGIN ONLY AFTER ALL VALIDATIONS PASS ---

if [[ ! -d "$hf_model_dir/.git" ]]; then
  mkdir -p "$hf_model_dir"
  git clone "$model_repo" "$hf_model_dir"
fi

git -C "$hf_model_dir" fetch --quiet --depth=1 origin "$model_revision"
git -C "$hf_model_dir" checkout --quiet --detach "$model_revision"

if command -v git-lfs >/dev/null 2>&1; then
  git -C "$hf_model_dir" lfs pull --include="*.safetensors,*.json,*.model,*.txt"
else
  if grep -q "version https://git-lfs.github.com/spec/v1" "$hf_model_dir"/*.safetensors 2>/dev/null; then
    echo "Error: git-lfs is required to pull model weights, but git-lfs was not found in PATH." >&2
    echo "The repository at $hf_model_dir contains unresolved LFS pointer files." >&2
    exit 2
  fi
fi

if [[ "$(git -C "$hf_model_dir" rev-parse HEAD)" != "$model_revision" ]]; then
  echo "Error: Model revision verification failed. Expected $model_revision, got $(git -C "$hf_model_dir" rev-parse HEAD)" >&2
  exit 2
fi

mkdir -p "$mlx_model_dir"

exec "$mlx_convert_bin" \
  --hf-path "$hf_model_dir" \
  --mlx-path "$mlx_model_dir" \
  --quantize \
  --q-bits 4
