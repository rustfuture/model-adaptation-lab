# Local LoRA Training Run — 2026-09-06

## Summary

A real local LoRA adaptation experiment was performed on Apple Silicon using MLX without cloud GPU or API spend. The adapter was trained on the authored synthetic Rust error dataset and evaluated against the held-out test split.

## Execution Environment & Hardware

- **Host**: Apple M4 Pro (12 CPU cores, 16 GPU cores, 24 GB unified memory)
- **OS**: macOS 26.6.2 (ARM64)
- **Toolchain**: `mlx` and `mlx_lm 0.31.3` in `/tmp/model-lab-mlx-venv`
- **Cost**: $0.00 (100% local execution)

## Model & Dataset Provenance

- **Base Model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **Model Revision**: Git commit `2e1fd397ee46e1388853d2af2c993145b0f1098a`
- **Model License**: Apache-2.0
- **Base Artifact**: Converted to 4-bit quantized MLX format (`828 MB` on disk)
- **Dataset SHA-256**: `bd488f5826fdae9e8fab7ad0911534fad96757bdd7cb99c45103870f68392d05`
- **Splits**: 6 train, 3 validation, 3 test (family-disjoint)
- **Test Integrity**: The 3 test records were strictly isolated in `data/mlx/test.jsonl` and excluded from the training data folder (`/tmp/model-lab-data` contained only `train.jsonl` and `valid.jsonl`).

## LoRA Configuration & Metrics

- **Method**: LoRA (low-rank adaptation)
- **Trainable Parameters**: 5,276,416 / 1,543,714,816 (0.342%)
- **Rank**: 8
- **Scale**: 20.0
- **Num Layers**: 16
- **Learning Rate**: 1e-4
- **Optimizer**: Adam
- **Batch Size**: 1
- **Iterations**: 50 (~8 epochs over the 6 training samples)
- **Mask Prompt**: Enabled (loss computed strictly on assistant response tokens)
- **Seed**: 42
- **Peak Memory**: 1.590 GB unified memory
- **Speed**: 6.40 it/s (176–186 tokens/s)
- **Total Duration**: 10.2 seconds
- **Loss Progression**:
  - Iter 1: Val loss 3.301
  - Iter 5: Train loss 2.329
  - Iter 10: Val loss 2.347, Train loss 0.647
  - Iter 20: Val loss 3.436, Train loss 0.245
  - Iter 30: Val loss 3.671, Train loss 0.326
  - Iter 40: Val loss 3.377, Train loss 0.365
  - Iter 50: Val loss 3.695, Train loss 0.159
- **Artifacts**: Weights saved to `/tmp/model-lab-adapters/adapters.safetensors` (excluded from git tracking per rule).

## Held-Out Test Evaluation

The untouched 3-sample held-out test split was evaluated using `scripts/evaluate_mlx.py` under two configurations:

| Metric | Base Model (MLX 4-bit) | LoRA Adapter (MLX 4-bit) | Notes |
|---|---|---|---|
| Exact Strategy Match | 0/3 (0%) | 0/3 (0%) | Neither model produced verbatim strategy string |
| Keyword Coverage Proxy | 2/3 (66.7%) | 0/3 (0%) | Base model mentioned relevant keywords; adapter did not |
| Latency p50 | 825 ms | 355 ms | Observed reduction; mechanism not isolated |
| Latency max | 932 ms | 403 ms | Adapter outputs were shorter |
| Output Format | Verbose conversational text with unsolicited code rewrites | Two-line `diagnosis: ...\nfix_strategy: ...` on the held-out records | Observed on 3 records |

## Technical Assessment & Negative Result

1. **Format**: The adapter's held-out outputs followed the concise two-line format; the base model's did not. This is a narrow observation over three records, not a proof of general format learning.
2. **Latency**: Median generation latency was 825 ms (base) vs 355 ms (adapter). Shorter output is a plausible explanation, but token counts were not measured, so the mechanism is not isolated.
3. **Generalization failed across error families (Negative Result)**: Validation loss rose from 2.347 at step 10 to 3.695 at step 50 as train loss reached 0.159, which is consistent with overfitting. On the held-out `control_flow` family the base model's outputs contained the relevant keywords (2/3) while the adapter's did not (0/3), and neither matched the exact strategy string. This run does not establish the cause; memorization of training samples is a hypothesis, not a demonstrated mechanism, and three held-out records are too few to generalize.
4. **Significance**: This is a small, honestly reported negative result. It suggests that adapting a small model to a semantic domain needs much more family variety or retrieval grounding, but it does not prove a general rule.
