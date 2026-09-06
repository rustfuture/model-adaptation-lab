# Training readiness gate (historical checkpoint)

Status: superseded by the completed local MLX run documented in
[`training-run-2026-09-06.md`](training-run-2026-09-06.md).

This file records the pre-training gate, not the current project status. At that
checkpoint the local host was an Apple M4 Pro with 24 GB unified memory; the
selected Apache-2.0 candidate was available only in a quantized Ollama format,
and the Python/MLX training stack was not installed. The Ollama artifact was
correctly not treated as a fine-tuning checkpoint.

The gate required an exact model revision, a compatible training path, package
versions, resource measurements, and a family-disjoint train/validation/test
manifest. Those requirements were subsequently met by the pinned MLX LoRA run:
see [`training-manifest.json`](../training-manifest.json),
[`scripts/train_mlx_lora.sh`](../scripts/train_mlx_lora.sh), and the held-out
evaluation report. The $0 spend ceiling and no-hosted-GPU boundary remain in
force.
