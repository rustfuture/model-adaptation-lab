#!/usr/bin/env bash
set -euo pipefail

model_repo="${HF_MODEL_REPO:-https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct.git}"
model_revision="${MODEL_REVISION:-2e1fd397ee46e1388853d2af2c993145b0f1098a}"
hf_model_dir="${HF_MODEL_DIR:-/tmp/model-lab-hf/qwen2.5-coder-1.5b-instruct}"
mlx_model_dir="${MLX_MODEL_DIR:-/tmp/model-lab-mlx/qwen2.5-coder-1.5b}"
mlx_convert_bin="${MLX_CONVERT_BIN:-mlx_lm.convert}"

if [[ ! -d "$hf_model_dir/.git" ]]; then
  mkdir -p "$(dirname "$hf_model_dir")"
  git clone "$model_repo" "$hf_model_dir"
fi

git -C "$hf_model_dir" fetch --quiet --depth=1 origin "$model_revision"
git -C "$hf_model_dir" checkout --quiet --detach "$model_revision"
if command -v git-lfs >/dev/null 2>&1; then
  git -C "$hf_model_dir" lfs pull --include='*.safetensors,*.json,*.model,*.txt'
fi

[[ "$(git -C "$hf_model_dir" rev-parse HEAD)" == "$model_revision" ]] || {
  echo "Model revision verification failed" >&2
  exit 2
}

if [[ "$mlx_convert_bin" == */* ]]; then
  [[ -x "$mlx_convert_bin" ]] || { echo "MLX converter not found: $mlx_convert_bin" >&2; exit 2; }
else
  command -v "$mlx_convert_bin" >/dev/null 2>&1 || { echo "MLX converter not found: $mlx_convert_bin" >&2; exit 2; }
fi

rm -rf "$mlx_model_dir"
mkdir -p "$mlx_model_dir"
exec "$mlx_convert_bin" \
  --hf-path "$hf_model_dir" \
  --mlx-path "$mlx_model_dir" \
  --quantize \
  --q-bits 4
