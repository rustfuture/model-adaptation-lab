# Negative result: the recorded adapter did not improve held-out quality

This report describes one small, local experiment. It is a negative result, not
a claim about LoRA or Qwen models in general.

## Claim boundary

- The run was recorded on 2026-09-06 on an Apple M4 Pro with no cloud GPU or
  API spend.
- The base model revision was pinned to
  `2e1fd397ee46e1388853d2af2c993145b0f1098a`.
- The authored corpus has 12 records. The six training records and three
  validation records are family-disjoint from the three-record `control_flow`
  test holdout. The machine-readable proof is
  [`evidence/dataset-validation.json`](../evidence/dataset-validation.json).
- The base model weights and LoRA adapter weights are not in this repository.
  The adapter was emitted under `/tmp` during the historical run, but was not
  retained, so it has no verifiable adapter hash and cannot be reloaded from
  this checkout. The preserved raw outputs support metric recomputation only;
  full training reproduction is not claimed.

## Held-out observations

| Variant | Exact strategy match | Keyword coverage | Latency p50 | Latency max |
| --- | ---: | ---: | ---: | ---: |
| MLX quantized base | 0/3 | 2/3 | 825 ms | 932 ms |
| MLX LoRA adapter | 0/3 | 0/3 | 355 ms | 403 ms |

The adapter produced concise two-line responses in these three examples, while
the base responses were longer. Lower latency coincided with incorrect answers
and is therefore not treated as a quality improvement. Validation loss rose
from 2.347 at step 10 to 3.695 at step 50 while final training loss was 0.159;
this is consistent with overfitting, but does not establish its cause.

## Interpretation and limitations

The only supported conclusion is that this run did not demonstrate a quality
gain on the held-out proxy: keyword coverage fell from 2/3 to 0/3, with exact
strategy match at 0/3 for both variants. Memorization is a hypothesis, not a
demonstrated mechanism. Three test records and a hand-authored keyword proxy
cannot establish generalization, compilation success, semantic correctness, or
production safety.

Raw outputs, hashes, metric definitions, split IDs, and unavailable-weight
status are maintained in [`evidence/metadata.json`](../evidence/metadata.json).
Recompute them without model weights with:

```bash
python3 scripts/verify_evidence.py
```
