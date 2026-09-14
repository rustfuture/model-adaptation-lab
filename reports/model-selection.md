# Model and resource decision record

Date: 2026-09-06

## Candidate

Initial local baseline candidate: `Qwen/Qwen2.5-Coder-1.5B-Instruct`.

- Official model card lists Apache-2.0 licensing: <https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct>.
- Official license file: <https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/blob/main/LICENSE>.
- PEFT/LoRA reference: <https://huggingface.co/docs/peft/main/conceptual_guides/lora>.
- Local host: Apple M4 Pro, 24 GB unified memory, 16-core integrated GPU.
- Local inference path: Ollama; no hosted API or paid GPU is used in this decision.

The candidate is suitable for a small prompt baseline and a possible adapter experiment, but suitability is not assumed from the license alone. Actual load time, memory, latency, output format, and patch/test behavior must be measured locally. The model card and license should be rechecked at the exact pinned revision before publication.

## Spend and stop gates

- Current spend: $0.
- Do not start hosted inference or rented GPU without a concrete budget entry.
- Stop local work if memory pressure, unacceptable latency, or poor held-out behavior makes the experiment less useful than the deterministic baseline.
- A historical fine-tuning run and separate untouched test result are recorded, but reusable fine-tuning
  remains unclaimed because the adapter weights were not preserved. See
  [`negative-result.md`](negative-result.md) and [`evidence/metadata.json`](../evidence/metadata.json).
