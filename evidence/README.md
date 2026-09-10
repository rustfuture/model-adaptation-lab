# Evaluation evidence package

This directory records what is preserved for the recorded model-adaptation experiment and what is not.
It contains no secrets and no model weights.

## Data

- Corpus: `data/rust_errors.jsonl` — 12 authored synthetic Rust-error records. No customer or scraped
  private data.
- Dataset SHA-256: `bd488f5826fdae9e8fab7ad0911534fad96757bdd7cb99c45103870f68392d05`
- Splits: family-disjoint (train: `parsing`, `indexing`; validation: `ownership`; test: `control_flow`).
- Test hold-out: 3 records.

## Model

- Base: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, Apache-2.0.
- Pinned revision: `2e1fd397ee46e1388853d2af2c993145b0f1098a`.
- The 4-bit MLX weights are not committed (large, redistributed separately under the upstream license).

## Adapter

- Configuration: LoRA, 50 iterations, rank 8, learning rate 1e-4, seed 42, 4-bit MLX.
- **Not preserved**: the adapter weights lived under `/tmp/model-lab-adapters/` and were not retained, so
  no adapter hash is available. The training configuration is recorded in `training-manifest.json`.

## Recorded commands

~~~bash
python3 scripts/validate_dataset.py
python3 scripts/baseline.py
python3 scripts/prepare_mlx_data.py
MODEL_REVISION=2e1fd397ee46e1388853d2af2c993145b0f1098a scripts/prepare_mlx_model.sh
MLX_MODEL_DIR=/tmp/model-lab-mlx/qwen2.5-coder-1.5b MLX_ADAPTER_DIR=/tmp/model-lab-adapters scripts/train_mlx_lora.sh
python3 scripts/evaluate_mlx.py --model /tmp/model-lab-mlx/qwen2.5-coder-1.5b --adapter /tmp/model-lab-adapters
~~~

## Results (preserved)

- `artifacts/mlx-evaluation-report.json` — held-out metrics for the quantized base and the adapter.
- `artifacts/mlx_base_quantized-test.jsonl`, `artifacts/mlx_lora_adapter-test.jsonl` — raw outputs.
- Base: keyword coverage 2/3, exact string match 0/3, p50 825 ms. Adapter: keyword coverage 0/3, exact
  string match 0/3, p50 355 ms.
- The adapter did not demonstrate a quality gain. Latency is not a quality metric. The cause of the
  held-out failure is not established by this experiment.

## Environment

- Apple M4 Pro, 24 GB unified memory; macOS Darwin 25.6.0.
- Python 3.11+, `mlx==0.32.2`, `mlx-lm==0.31.3`.
- Cost: $0.00 (local execution).

## Scope of the safety checks

`tests/test_script_safety.sh` verifies that the helper scripts reject unsafe path relationships, refuse
to delete existing non-empty destinations, and use isolated atomic run directories. These are filesystem
guards, not an adversarial OS sandbox, and they do not validate model output or training reproduction.
