# Model Adaptation Lab

This software tests whether LoRA fine-tuning helps one small model (Qwen2.5-Coder-1.5B) explain Rust compiler errors, using a 12-record dataset.

[![CI](https://github.com/rustfuture/model-adaptation-lab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rustfuture/model-adaptation-lab/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Open validation in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rustfuture/model-adaptation-lab/blob/main/notebooks/validation_colab.ipynb)

**Status:** Research prototype; dataset checks pass and the negative result is preserved.

- Checks dataset fields and keeps Rust error families apart across data splits ([dataset](data/rust_errors.jsonl), [validator](scripts/validate_dataset.py)).
- Compares model answers with one deterministic rule baseline ([baseline](scripts/baseline.py)); the recorded run showed no gain ([report](reports/negative-result.md), [evidence](evidence/metadata.json)). On the three held-out records the rule baseline scores 1/3 exact strategy match, ahead of both model variants (0/3 each); its rule table is hand-written and includes an entry for a held-out error code, so it is a plumbing floor, not a tuned competitor (details in the [report](reports/negative-result.md)).
- Can tune a small set of model weights with LoRA (Low-Rank Adaptation) using Apple's MLX toolkit on Apple Silicon ([training](scripts/train_mlx_lora.sh), [evaluation](scripts/evaluate_mlx.py)).
- Checks Rust examples with the Rust compiler and records evaluation evidence ([compile checks](scripts/validate_rustc_snippets_v2.py)).

## Quick start

Run the data checks and baseline without model weights:

```bash
python3 scripts/validate_dataset.py
python3 scripts/validate_dataset_v2.py
python3 scripts/validate_rustc_snippets.py
python3 scripts/validate_rustc_snippets_v2.py
python3 scripts/baseline.py
python3 scripts/verify_evidence.py --write-manifest /tmp/evidence-metadata.json
diff -u evidence/metadata.json /tmp/evidence-metadata.json
python3 -m unittest discover -s tests -p 'test_*.py'
./tests/test_script_safety.sh
```

Model training needs an Apple Silicon computer, MLX, mlx-lm, and the model weights, which are not included here:

```bash
python3 scripts/prepare_mlx_data.py
scripts/prepare_mlx_model.sh
scripts/train_mlx_lora.sh
python3 scripts/evaluate_mlx.py --manifest /tmp/model-lab-runs/<run_id>/run_manifest.json
```

The v2 runner executes its full pipeline and reports when model weights are unavailable: `./scripts/run_v2_experiment.sh`.

## How it works

- The validators check required dataset fields and keep each Rust error family in one split. They also compile source examples and check for the expected compiler error.
- Preparation scripts turn examples into prompts and answers for model training and evaluation.
- On Apple Silicon, MLX can tune a small set of model weights with LoRA, a method called Low-Rank Adaptation; run data stays in isolated temporary folders.
- The evaluators compare saved model answers with expected answers. The v2 evaluator also formats generated Rust code and checks whether it compiles.
- Evidence scripts recalculate file digests and metrics from saved outputs.

See [docs/reference.md](docs/reference.md) for experiment details, results, file safety rules, and the repository map.

## Correctness Levels

The v2 evaluator checks whether generated Rust code compiles. Passing that check does not show that the code behaves correctly or that the explanation is right.

| Level | Meaning here | Established by |
|---|---|---|
| Compiles | The snippet builds as a standalone Rust 2021 binary | `validate_rustc_snippets_v2.py`, v2 evaluation |
| Behaviorally correct | The fixed code passes behavior tests | Not claimed; the dataset has no behavior tests |
| Semantically correct | The diagnosis and fix are right | Not claimed; keyword and exact-match proxies only |

Exact text match and keyword coverage are narrow measures. They do not establish semantic correctness.

## Tests

The CI workflow in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs these commands:

```bash
python -m pip install --disable-pip-version-check --no-input "pyflakes==3.2.0"
python -m pip check
pyflakes scripts/
python3 scripts/validate_dataset.py
python3 scripts/validate_rustc_snippets.py
python3 scripts/validate_dataset_v2.py
python3 scripts/validate_rustc_snippets_v2.py
python3 scripts/validate_dataset.py --write-manifest "$RUNNER_TEMP/dataset-validation.json"
diff -u evidence/dataset-validation.json "$RUNNER_TEMP/dataset-validation.json"
python3 -m unittest discover -s tests -p 'test_dataset_contract.py'
python3 -m unittest discover -s tests -p 'test_v2_provenance.py'
python3 scripts/baseline.py
python3 scripts/prepare_mlx_data.py
git diff --exit-code -- data/mlx
python3 scripts/verify_evidence.py --write-manifest "$RUNNER_TEMP/evidence-metadata.json"
diff -u evidence/metadata.json "$RUNNER_TEMP/evidence-metadata.json"
./tests/test_script_safety.sh
```

The tests check script lint, dataset fields and splits, Rust compilation, baseline scoring, prepared data, saved evidence, and filesystem path safety.

## Limitations

- The recorded run showed no quality gain. This single small experiment says nothing general about LoRA or Qwen models.
- The recorded run trained on dataset SHA-256 `bd488f58...`; the committed `data/rust_errors.jsonl` is now `505ba845...` because five records' `code` fields were edited on 2026-09-15 (commit `cf0e452`). `data/mlx/*` regenerated from the current file is therefore not byte-identical to the recorded run's training data. See [evidence/README.md](evidence/README.md#dataset-hash-note).
- The historical adapter is not included, so its full training run cannot be repeated from this checkout.
- The small test set and hand-written keyword measure cannot establish significance or semantic correctness.
- The loss pattern fits overfitting, but the experiment does not establish the cause.
- MLX training only runs on Apple Silicon. Colab checks the data, baseline, and evidence; it does not run MLX training.

## License

MIT; see [LICENSE](LICENSE).
