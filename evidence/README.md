# Evaluation evidence package

This directory records what is preserved for the recorded model-adaptation experiment and what is not.
It contains no secrets and no model weights. `evidence/raw/` holds verbatim, unmodified copies of the
small text outputs; nothing there was regenerated. Deterministic metadata is maintained in
[`dataset-validation.json`](dataset-validation.json) and [`metadata.json`](metadata.json).

## Data

- Corpus: `data/rust_errors.jsonl` — 12 authored synthetic Rust-error records. No customer or scraped
  private data.
- Dataset SHA-256: `bd488f5826fdae9e8fab7ad0911534fad96757bdd7cb99c45103870f68392d05`
- Splits: family-disjoint (train: `parsing`, `indexing`; validation: `ownership`; test: `control_flow`).
- Test hold-out: 3 records.
- Split record IDs and per-split ID hashes are preserved in
  [`dataset-validation.json`](dataset-validation.json). The evaluator now validates the same full
  split contract before loading a model.

## Model

- Base: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, Apache-2.0.
- Pinned revision: `2e1fd397ee46e1388853d2af2c993145b0f1098a`.
- The 4-bit MLX weights are not present in this checkout (large, redistributed separately under the
  upstream license); no base-weight hash is available here.

## Adapter

- Configuration: LoRA, 50 iterations, rank 8, scale 20.0, learning rate 1e-4, batch size 1, seed 42,
  4-bit MLX, prompt masking enabled.
- **Not preserved**: the adapter weights were emitted under `/tmp/model-lab-adapters/` during the
  historical run but were not retained, so no adapter hash is available. The training configuration
  and this explicit claim boundary are recorded in `training-manifest.json`.
- Metric recomputation is reproducible from the preserved raw outputs in `evidence/raw/`. Full training
  reproduction is **not** claimed: the adapter weights are gone, so the trained adapter cannot be
  rebuilt or hash-verified from this repository.

## Recorded commands

~~~bash
python3 scripts/validate_dataset.py
python3 scripts/baseline.py
python3 scripts/prepare_mlx_data.py
MODEL_REVISION=2e1fd397ee46e1388853d2af2c993145b0f1098a scripts/prepare_mlx_model.sh
MLX_MODEL_DIR=/tmp/model-lab-mlx/qwen2.5-coder-1.5b MLX_ADAPTER_DIR=/tmp/model-lab-adapters scripts/train_mlx_lora.sh
python3 scripts/evaluate_mlx.py --model /tmp/model-lab-mlx/qwen2.5-coder-1.5b --adapter /tmp/model-lab-adapters
python3 scripts/ollama_baseline.py
~~~

## Preserved outputs

The three files under `evidence/raw/` are byte-for-byte copies of the local `artifacts/` files
(SHA-256 verified) and are the only preserved model outputs. The MLX files were produced against
dataset `bd488f5826fdae9e8fab7ad0911534fad96757bdd7cb99c45103870f68392d05` and base model revision
`2e1fd397ee46e1388853d2af2c993145b0f1098a`.

- `evidence/raw/mlx_base_quantized-test.jsonl`
  - SHA-256: `55c5fadef56b7ff927aa56ed83082248e4fa4b816621d1d0cb8530e3015e1ff7`
  - Produced by: `python3 scripts/evaluate_mlx.py --model /tmp/model-lab-mlx/qwen2.5-coder-1.5b --adapter /tmp/model-lab-adapters` (quantized base variant).
  - Adapter configuration: not applicable (4-bit MLX base, no adapter).
- `evidence/raw/mlx_lora_adapter-test.jsonl`
  - SHA-256: `825ee22b4ede14a8d05dfdbc5960ebc128028b269274bfbb713eede9605ecf2e`
  - Produced by: the same evaluation command (LoRA adapter variant).
  - Adapter configuration: LoRA rank 8, scale 20.0, 50 iterations, lr 1e-4, batch 1, seed 42, mask-prompt, 4-bit MLX. Weights not preserved; no adapter hash exists.
- `evidence/raw/ollama-baseline.jsonl`
  - SHA-256: `b649e0bc796797ddb378fa9bbbbd87e885b3f78bc4c7773c112542339653234d`
  - Produced by: `python3 scripts/ollama_baseline.py` (default `OLLAMA_MODEL=qwen2.5-coder:1.5b`, temperature 0).
  - Base model revision: not pinned — Ollama tag `qwen2.5-coder:1.5b`; adapter configuration not applicable.

### Not preserved

- `artifacts/mlx-evaluation-report.json` is **not preserved** because it is generated output. The
  evaluator now emits portable repository-relative/external path labels, and its metrics are represented
  by the tracked metadata manifest below.
- Local `artifacts/` files are ignored generated output and are not part of the public evidence package.
  The two MLX raw outputs and the Ollama baseline are the preserved, shareable copies under
  `evidence/raw/`.

## Results (independently recomputed)

Recomputed on 2026-09-11 from the preserved raw JSONL, using the same definitions as
`scripts/evaluate_mlx.py`: exact strategy match is normalized whole-response equality with the expected
strategy, keyword coverage uses that script's per-record `REQUIRED_TERMS`, p50 is `statistics.median`.
The preserved JSONL store `id`/`response`/`expected_diagnosis`/`expected_strategy`/`latency_ms`; the
match flags were not stored in them, so they are re-derived from the response and expected fields.

| Variant | Exact strategy match | Keyword coverage | Latency p50 | Latency max |
|---|---|---|---|---|
| Base (MLX 4-bit) | 0/3 | 2/3 | 825 ms | 932 ms |
| LoRA adapter (MLX 4-bit) | 0/3 | 0/3 | 355 ms | 403 ms |

These match [`metadata.json`](metadata.json), `training-manifest.json`, the repository `README.md`, `reports/training-run-2026-09-06.md`,
and the unpreserved `artifacts/mlx-evaluation-report.json`. The deterministic non-LLM baseline is
1/3 exact strategy match (`python3 scripts/baseline.py`).

- The adapter did not demonstrate a quality gain: 0/3 vs 2/3 keyword coverage on 3 held-out records,
  and 0/3 exact strategy match for both variants.
- Shorter latency on incorrect answers is not a quality improvement. The cause of the held-out failure
  is not established by this experiment.

Run `python3 scripts/verify_evidence.py` to validate every preserved output against the test IDs,
expected fields, hashes, and metric definitions without loading model weights or calling a service. The
narrative claim boundary is in [`reports/negative-result.md`](../reports/negative-result.md).

## Environment

- Apple M4 Pro, 24 GB unified memory; macOS Darwin 25.6.0.
- Python 3.11+, `mlx==0.32.2`, `mlx-lm==0.31.3`.
- Cost: $0.00 (local execution).

## Scope of the safety checks

`tests/test_script_safety.sh` verifies that the helper scripts reject unsafe path relationships, refuse
to delete existing non-empty destinations, and use isolated atomic run directories. These are filesystem
guards, not an adversarial OS sandbox, and they do not validate model output or training reproduction.
