# Model Adaptation Lab

An evidence-first experiment for adapting a small model to structured Rust compiler-error explanations and conservative fix suggestions.

This repository contains the reproducible data contract, split validator, deterministic non-LLM baseline, and local MLX LoRA training and evaluation pipeline. All training and evaluation runs were conducted locally on an Apple M4 Pro (24 GB unified memory) with zero cloud GPU or API spend.

## Current status

- **Dataset**: 12 authored, synthetic Rust-error records; no customer or scraped private data.
- **Split**: train/validation/test are separated by error family to eliminate near-duplicate leakage.
- **Base Model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (Apache-2.0, commit `2e1fd397ee46e1388853d2af2c993145b0f1098a`), 4-bit quantized MLX format.
- **Baseline**: deterministic error-code strategy classifier (1/3 exact match on held-out test).
- **LoRA Training**: Completed locally via MLX (50 iters, rank 8, lr 1e-4, seed 42, peak memory 1.59 GB, duration 10.2s).
- **Outcome & Negative Result**: Adapter successfully learned the concise two-line structural schema (`diagnosis: ...\nfix_strategy: ...`), reducing generation latency by 57% (p50 825 ms to 355 ms) due to eliminating conversational preamble. However, semantic generalization on unseen error families failed due to sample memorization (0/3 exact match, 0/3 keyword proxy vs 2/3 for base). Shorter latency on a failing output is not a general capability improvement; this negative generalization result is honestly documented.
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
#    create the 4-bit MLX directory safely (does not overwrite existing without ALLOW_OVERWRITE=1)
scripts/prepare_mlx_model.sh

# 5. Train the pinned experiment
# By default, train_mlx_lora.sh uses an isolated unique run directory (/tmp/model-lab-runs/<run_id>/)
# to guarantee existing models and previous adapter outputs are never deleted.
scripts/train_mlx_lora.sh

# 6. Evaluate the base model and adapter on the untouched test split
# Evaluation dynamically verifies dataset SHA-256 and records exact model/adapter metadata.
python3 scripts/evaluate_mlx.py \
  --model /tmp/model-lab-mlx/qwen2.5-coder-1.5b \
  --adapter /tmp/model-lab-adapters
```

### Script Safety & File Protection Contract

All helper shell and Python scripts enforce defensive path and environment guards:
- Destructive operations (`rm -rf`) are barred from operating on root (`/`), `$HOME`, the repository root, parent paths, symlinks, or colliding source/target directories.
- Existing non-empty model and adapter target directories are protected: scripts reject overwrites unless `ALLOW_OVERWRITE=1` (or `ALLOW_REUSE=1` for already-converted models) is explicitly set.
- Unique run IDs (`run_<timestamp>_<pid>`) isolate consecutive training runs by default under `/tmp/model-lab-runs/`.
- Clear, actionable errors are emitted if `git-lfs` is missing when LFS pointer weights are encountered, or when MLX executables (`mlx_lm.convert`, `mlx_lm.lora`) are absent from PATH.
- Verify script safety rules at any time with: `./tests/test_script_safety.sh`.

### Environment & Dependencies

The recorded training and evaluation runs used:
- Apple Silicon (macOS Darwin 25.6.0, Apple M4 Pro, 24 GB unified memory)
- Python 3.11+ with `mlx==0.32.2` and `mlx-lm==0.31.3` (Note: `mlx` and `mlx-lm` are independently versioned packages and must not be assumed to share identical version strings).

The clean-install audit verifies dataset validity, non-LLM baseline accuracy, and data-split generation. It does not download model weights or rerun Apple Silicon training or evaluation. Dry-run and safety tests confirm wrapper safety and syntax, not training reproduction.

## Safety and evaluation boundary

Compiler text and repository code are treated as data, not instructions. Suggested patches must be applied in an isolated temporary checkout and pass formatting, compilation, and behavior tests. A compiling patch is not automatically a correct patch.
