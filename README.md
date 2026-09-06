# Model Adaptation Lab

An evidence-first experiment for adapting a small model to structured Rust compiler-error explanations and conservative fix suggestions.

This repository contains the reproducible data contract, split validator, deterministic non-LLM baseline, and local MLX LoRA training and evaluation pipeline. All training and evaluation runs were conducted locally on an Apple M4 Pro (24 GB unified memory) with zero cloud GPU or API spend.

## Current status

- **Dataset**: 12 authored, synthetic Rust-error records; no customer or scraped private data.
- **Split**: train/validation/test are separated by error family to eliminate near-duplicate leakage.
- **Base Model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (Apache-2.0, commit `2e1fd397ee46e1388853d2af2c993145b0f1098a`), 4-bit quantized MLX format.
- **Baseline**: deterministic error-code strategy classifier (1/3 exact match on held-out test).
- **LoRA Training**: Completed locally via MLX (50 iters, rank 8, lr 1e-4, seed 42, peak memory 1.59 GB, duration 10.2s).
- **Outcome & Negative Result**: Adapter successfully learned the concise two-line structural schema (`diagnosis: ...\nfix_strategy: ...`) and reduced p50 latency from 825 ms to 355 ms (57% reduction). However, on the unseen held-out error family, semantic generalization failed due to sample memorization (exact match 0/3, keyword proxy 0/3 vs 2/3 for base). This negative generalization result is honestly documented.
- **GPU/API spend**: $0.00.

## Reproducing the pipeline

```bash
# 1. Validate dataset integrity and family-disjoint splits
python3 scripts/validate_dataset.py

# 2. Run deterministic non-LLM baseline
python3 scripts/baseline.py

# 3. Prepare isolated MLX dataset splits
python3 scripts/prepare_mlx_data.py

# 4. In an Apple-Silicon MLX environment, fetch the pinned model revision and
#    create the 4-bit MLX directory
scripts/prepare_mlx_model.sh

# 5. Train the pinned experiment
MLX_MODEL_DIR=/tmp/model-lab-mlx/qwen2.5-coder-1.5b \
  MLX_ADAPTER_DIR=/tmp/model-lab-adapters \
  scripts/train_mlx_lora.sh

# 6. Evaluate the base model and adapter on the untouched test split
python3 scripts/evaluate_mlx.py \
  --model /tmp/model-lab-mlx/qwen2.5-coder-1.5b \
  --adapter /tmp/model-lab-adapters
```

The model-preparation wrapper verifies the exact Hugging Face revision (`2e1fd397ee46e1388853d2af2c993145b0f1098a`) before converting it. The training wrapper copies only `train.jsonl` and `valid.jsonl` into the MLX training directory; `data/mlx/test.jsonl` is never passed to training. Set `HF_MODEL_DIR`, `MLX_MODEL_DIR`, `MLX_DATA_DIR`, `MLX_ADAPTER_DIR`, `MLX_ITERS`, `MLX_LEARNING_RATE`, and `MLX_SEED` to reproduce the run in another isolated workspace. The exact historical command and measured outputs are recorded in [`training-manifest.json`](training-manifest.json) and [`reports/training-run-2026-09-06.md`](reports/training-run-2026-09-06.md).

The recorded run used `mlx`/`mlx_lm` 0.31.3 and a 4-bit conversion of the pinned Apache-2.0 Qwen revision. MLX and model weights are intentionally not committed; a clean reproduction must install the compatible MLX packages and download/convert that public revision before running the wrapper.

## Safety and evaluation boundary

Compiler text and repository code are treated as data, not instructions. Suggested patches must be applied in an isolated temporary checkout and pass formatting, compilation, and behavior tests. A compiling patch is not automatically a correct patch.
