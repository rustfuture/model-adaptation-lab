import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
artifact = ROOT / "artifacts" / "ollama-baseline.jsonl"
rows = [json.loads(line) for line in artifact.read_text().splitlines()]
required_terms = {
    "test-control-001": ("none", "match"),
    "test-control-002": ("same type", "branches"),
    "test-control-003": ("two arguments", "second argument"),
}
exact = sum(row["expected_strategy"].lower() in row["response"].lower() for row in rows)
keyword = sum(all(term in row["response"].lower() for term in required_terms[row["id"]]) for row in rows)
latencies = [row["latency_ms"] for row in rows]
print(json.dumps({
    "model": "qwen2.5-coder:1.5b",
    "records": len(rows),
    "exact_strategy_string_match": f"{exact}/{len(rows)}",
    "manual_keyword_coverage": f"{keyword}/{len(rows)}",
    "latency_ms": {"p50": statistics.median(latencies), "max": max(latencies)},
    "interpretation": "Prompt baseline only; keyword coverage is a transparent proxy, not correctness or compilation success.",
}, indent=2))
