#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"

model_repo="${HF_MODEL_REPO:-https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct.git}"
model_revision="${MODEL_REVISION:-2e1fd397ee46e1388853d2af2c993145b0f1098a}"
hf_model_dir="${HF_MODEL_DIR:-/tmp/model-lab-hf/qwen2.5-coder-1.5b-instruct}"
mlx_model_dir="${MLX_MODEL_DIR:-/tmp/model-lab-mlx/qwen2.5-coder-1.5b}"
mlx_convert_bin="${MLX_CONVERT_BIN:-mlx_lm.convert}"
allow_overwrite="${ALLOW_OVERWRITE:-0}"
allow_reuse="${ALLOW_REUSE:-0}"

validate_path() {
  local target_path="$1"
  local label="$2"

  if [[ -z "$target_path" ]]; then
    echo "Error: $label path cannot be empty." >&2
    exit 2
  fi

  # Fast check for root directory
  if [[ "$target_path" == "/" || "$target_path" == "/." || "$target_path" == "/.." ]]; then
    echo "Error: $label cannot be the filesystem root (/)." >&2
    exit 2
  fi

  if [[ -L "$target_path" ]]; then
    echo "Error: $label cannot be a symlink: $target_path" >&2
    exit 2
  fi

  local canonical
  if [[ -d "$target_path" ]]; then
    canonical="$(cd "$target_path" && pwd -P)"
  else
    local parent_dir
    parent_dir="$(dirname "$target_path")"
    if [[ ! -d "$parent_dir" ]]; then
      mkdir -p "$parent_dir" 2>/dev/null || {
        echo "Error: Cannot create parent directory for $label: $parent_dir" >&2
        exit 2
      }
    fi
    canonical="$(cd "$parent_dir" && pwd -P)/$(basename "$target_path")"
  fi

  if [[ "$canonical" == "/" ]]; then
    echo "Error: $label cannot be the filesystem root (/)." >&2
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
}

validate_path "$hf_model_dir" "HF_MODEL_DIR"
validate_path "$mlx_model_dir" "MLX_MODEL_DIR"

# Check collision between source and destination
hf_canon="$(cd "$(dirname "$hf_model_dir")" && pwd -P)/$(basename "$hf_model_dir")"
mlx_canon="$(cd "$(dirname "$mlx_model_dir")" && pwd -P)/$(basename "$mlx_model_dir")"
if [[ "$hf_canon" == "$mlx_canon" ]]; then
  echo "Error: HF source ($hf_canon) and MLX destination ($mlx_canon) cannot be the same directory." >&2
  exit 2
fi

# Clone or fetch pinned Hugging Face repository
if [[ ! -d "$hf_model_dir/.git" ]]; then
  mkdir -p "$hf_model_dir"
  git clone "$model_repo" "$hf_model_dir"
fi

git -C "$hf_model_dir" fetch --quiet --depth=1 origin "$model_revision"
git -C "$hf_model_dir" checkout --quiet --detach "$model_revision"

# LFS check and pull
if command -v git-lfs >/dev/null 2>&1; then
  git -C "$hf_model_dir" lfs pull --include="*.safetensors,*.json,*.model,*.txt"
else
  # Check if model files are unresolved git-lfs pointers
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

# MLX convert executable check
if [[ "$mlx_convert_bin" == */* ]]; then
  [[ -x "$mlx_convert_bin" ]] || {
    echo "Error: MLX converter executable not found: $mlx_convert_bin" >&2
    echo "Please install mlx-lm or activate your Python virtualenv." >&2
    exit 2
  }
else
  command -v "$mlx_convert_bin" >/dev/null 2>&1 || {
    echo "Error: MLX converter command  not found in PATH." >&2
    echo "Please install mlx-lm or set MLX_CONVERT_BIN." >&2
    exit 2
  }
fi

# Check destination directory
if [[ -d "$mlx_model_dir" ]] && [[ -n "$(ls -A "$mlx_model_dir" 2>/dev/null)" ]]; then
  if [[ -f "$mlx_model_dir/config.json" ]] && [[ "$allow_reuse" == "1" ]]; then
    echo "Notice: MLX model directory already contains converted model and ALLOW_REUSE=1. Reusing existing model."
    exit 0
  fi
  if [[ "$allow_overwrite" != "1" ]]; then
    echo "Error: Destination MLX model directory exists and is not empty: $mlx_model_dir" >&2
    echo "Set ALLOW_OVERWRITE=1 to overwrite, or ALLOW_REUSE=1 to reuse existing converted model." >&2
    exit 2
  fi
  echo "Overwriting existing directory $mlx_model_dir (ALLOW_OVERWRITE=1)."
  rm -rf "$mlx_model_dir"
fi

mkdir -p "$mlx_model_dir"

exec "$mlx_convert_bin" \
  --hf-path "$hf_model_dir" \
  --mlx-path "$mlx_model_dir" \
  --quantize \
  --q-bits 4
