# Model Adaptation Lab

An evidence-first experiment for adapting a small model to structured Rust compiler-error explanations and conservative fix suggestions.

This repository contains the reproducible data contract, split validator, deterministic non-LLM baseline, and local MLX LoRA training and evaluation pipeline. All training and evaluation runs were conducted locally on an Apple M4 Pro (24 GB unified memory) with zero cloud GPU or API spend.

## Current status

- **Dataset**: 12 authored, synthetic Rust-error records; no customer or scraped private data.
- **Split**: train/validation/test are separated by the authored error-family labels (train: `parsing` and `indexing`, validation: `ownership`, test: `control_flow`). The validator checks all three split pairs. This reduces one source of overlap; it does not prove the absence of semantic duplicates or pretraining exposure.
- **Base Model**: Historical run used `Qwen/Qwen2.5-Coder-1.5B-Instruct` (Apache-2.0, commit `2e1fd397ee46e1388853d2af2c993145b0f1098a`) in 4-bit quantized MLX format; weights are not present in this checkout.
- **Baseline**: deterministic error-code strategy classifier (1/3 exact match on held-out test).
- **LoRA Training**: A historical local MLX run was recorded (50 iters, rank 8, lr 1e-4, seed 42, peak memory 1.59 GB, duration 10.2s). The base and adapter weights are not present in this checkout; no adapter hash or full training reproduction is claimed.
- **Outcome & Negative Result**: The recorded adapter produced shorter two-line answers but scored 0/3 on the held-out keyword proxy, compared with 2/3 for the base model. This tiny experiment did not demonstrate a quality gain. Shorter latency for incorrect answers is not a performance improvement, and the observations do not establish why the model failed.
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
#    create the 4-bit MLX directory safely into a fresh destination:
scripts/prepare_mlx_model.sh

# 5. Train the pinned experiment in an isolated, atomic run directory
# scripts/train_mlx_lora.sh creates /tmp/model-lab-runs/<run_id>/ and outputs a run_manifest.json.
scripts/train_mlx_lora.sh

# 6. Evaluate using the generated run manifest:
python3 scripts/evaluate_mlx.py --manifest /tmp/model-lab-runs/<run_id>/run_manifest.json

# Alternatively, evaluate explicit model and adapter paths:
python3 scripts/evaluate_mlx.py \
  --model /tmp/model-lab-mlx/qwen2.5-coder-1.5b \
  --adapter /tmp/model-lab-adapters

# Or evaluate the quantized base model only:
python3 scripts/evaluate_mlx.py \
  --model /tmp/model-lab-mlx/qwen2.5-coder-1.5b \
  --baseline-only

# Recompute the shareable split and evaluation evidence without model weights:
python3 scripts/validate_dataset.py --write-manifest /tmp/dataset-validation.json
python3 scripts/verify_evidence.py
```

### Script Safety & File Protection Contract

All helper shell and Python scripts enforce defensive path and environment guards:
- User model, adapter, data, and source directories are never deleted or overwritten. Existing non-empty destination directories are rejected with an explicit error; cleanup is limited to temporary run directories whose ownership was atomically acquired by the process.
- Path relationship checks verify all directory pairs (`model`, `data`, `adapter`, `hf_source`) preventing path collisions, parent/child nesting, symlinks, root (`/`), user home, and repository escapes without performing pre-check mutations.
- Unique atomic run directories (`/tmp/model-lab-runs/run-XXXXXX`) isolate training runs by default, generating a `run_manifest.json` with exact configuration metadata for evaluation.
- Existing Hugging Face checkouts with uncommitted modifications are detected and preserved without checkout or overwrite.
- Clear, actionable errors are emitted if `git-lfs` is missing when LFS pointer weights are encountered, or when MLX executables (`mlx_lm.convert`, `mlx_lm.lora`) are absent from PATH.
- Verify script safety rules at any time with: `./tests/test_script_safety.sh`.

### Environment & Dependencies

The recorded training and evaluation runs used:
- Apple Silicon (macOS Darwin 25.6.0, Apple M4 Pro, 24 GB unified memory)
- Python 3.11+ with `mlx==0.32.2` and `mlx-lm==0.31.3` (Note: `mlx` and `mlx-lm` are independently versioned packages and must not be assumed to share identical version strings).

The clean-install audit verifies dataset validity, deterministic split/evidence manifests, non-LLM baseline accuracy, and data-split generation. It does not download model weights or rerun Apple Silicon training or evaluation. Dry-run and safety tests confirm wrapper safety and syntax, not training reproduction. CI also checks that generated split files and evidence metadata are unchanged.

## Safety and evaluation boundary

Compiler text and repository code are treated as data, not instructions. Suggested patches must be applied in an isolated temporary checkout and pass formatting, compilation, and behavior tests. A compiling patch is not automatically a correct patch.

### Experiment Design & Rationale

The primary rationale behind this experiment is to isolate structural instruction following from semantic reasoning. 

- **Data splits:** Family-disjoint partitions test transfer to different authored categories. They do not guarantee that the model learned a general schema or that similar examples were absent from pretraining.
- **Metrics:** Exact string match and keyword coverage are narrow proxies, not format validation, compilation success, or semantic correctness. The corpus has only three held-out examples.
- **Corpus export:** `python3 scripts/generate_synthetic_data.py` prints the existing validated corpus. The legacy filename is retained, but it does not generate new examples. Do not redirect it onto `data/rust_errors.jsonl`, because shell redirection would truncate the source before it is read. The snippets have not all been verified as standalone rustc fixtures.
- **Safety boundaries:** Explicit run directories are claimed using exclusive creation. Failed acquisition never grants cleanup ownership. These filesystem checks are not an adversarial OS sandbox.

### Public evidence and negative result

The split proof is preserved in [`evidence/dataset-validation.json`](evidence/dataset-validation.json).
The tracked raw outputs, hashes, metric definitions, unavailable-weight status, and recomputed negative
result are summarized in [`evidence/metadata.json`](evidence/metadata.json); the narrative is in
[`reports/negative-result.md`](reports/negative-result.md). The historical run is evidence of what was
observed locally, not a redistributable model or reusable adapter checkpoint.
