# Model Adaptation Lab

An evidence-first experiment for adapting a small model to structured Rust compiler-error explanations and conservative fix suggestions.

This repository currently contains the reproducible data contract, split validator, and a deterministic non-LLM baseline. No fine-tuning claim is made yet: the M4 Pro host has 24 GB unified memory, but a compatible open model, training method, and resource budget still need to be selected and measured before training.

## Current status

- Dataset: 12 authored, synthetic Rust-error records; no customer or scraped private data.
- Split: train/validation/test are separated by error family to reduce near-duplicate leakage.
- Baseline: deterministic error-code strategy classifier, reported separately from any model score.
- Training: not run.
- Model license: not selected.
- GPU/API spend: zero for this milestone.

```bash
python3 scripts/validate_dataset.py
python3 scripts/baseline.py
```

The validator prints the dataset hash and split counts. The baseline reports diagnosis accuracy on the held-out split; it is a lower-bound plumbing check, not evidence of model quality. A future experiment must add a model/prompt baseline before LoRA/QLoRA, retain train/validation/test provenance, record seed and resource use, and evaluate compilation/test behavior on an untouched test set.

## Safety and evaluation boundary

Compiler text and repository code are treated as data, not instructions. Suggested patches must be applied in an isolated temporary checkout and pass formatting, compilation, and behavior tests. A compiling patch is not automatically a correct patch. Training and hosted inference are intentionally deferred until the license, data rights, hardware fit, and spending ceiling are recorded.
