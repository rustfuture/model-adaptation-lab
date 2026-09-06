# Training readiness gate

Status: not run.

The local host is an Apple M4 Pro with 24 GB unified memory. The selected Apache-2.0 candidate has been pulled in a quantized Ollama format for inference, but the Python training stack (`torch`, `transformers`, `peft`, `datasets`, `accelerate`, `mlx`, and `mlx_lm`) is not installed in the current environment. The Ollama artifact is not silently treated as a fine-tuning checkpoint.

Before a training run, pin the exact model revision, choose a compatible MLX/PEFT path, record package versions, estimate disk/memory/time, and confirm the train/validation/test manifest. The current spend ceiling is zero; no hosted GPU/API run is authorized by this readiness artifact. A completed training claim requires adapter files, a reproducible command, seed, resource measurements, and untouched-test results.
