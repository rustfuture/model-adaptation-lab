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
| Latency p50 | 825 ms | 355 ms | **57% latency reduction** with adapter |
| Latency max | 932 ms | 403 ms | Adapter consistently faster due to concise generation |
| Output Format | Verbose conversational text with unsolicited code rewrites | **Strict 2-line target format** (`diagnosis: ...\nfix_strategy: ...`) | Adapter successfully learned structural format |

## Technical Assessment & Negative Result

1. **Format adherence was learned**: The base model answered conversationally with extensive explanations and unrequested markdown code blocks. The adapter reliably adhered to the concise, non-conversational `diagnosis: ...\nfix_strategy: ...` format without markdown fences.
2. **Inference latency dropped sharply**: By constraining output verbosity to concise target lines, token count dropped and median generation latency fell from 825 ms to 355 ms.
3. **Generalization failed across error families (Negative Result)**: Validation loss rose from 2.347 at step 10 to 3.695 at step 50 as train loss reached 0.159, indicating clear overfitting. On the held-out test set (representing the unseen `control_flow` family), the adapter recited phrases memorized from the indexing/borrowing training examples rather than generalizing.
4. **Significance**: This establishes a verified, honest baseline. Adapting small models to complex semantic domains requires substantial family variety or few-shot retrieval grounding rather than 6-example parameter updates. This negative finding is recorded faithfully.
