#!/usr/bin/env python3
"""Verify and summarize the tracked, shareable evaluation evidence.

This script intentionally reads only the authored dataset and tracked raw
outputs. It never loads model weights, calls a service, or changes the raw
evidence files. The optional manifest output is deterministic so CI can detect
drift between the checked-in evidence and its source files.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from validate_dataset import build_manifest, read_dataset, validate_records

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "evidence" / "raw"

REQUIRED_TERMS = {
    "test-control-001": ("none", "match"),
    "test-control-002": ("same type", "branches"),
    "test-control-003": ("two arguments", "second argument"),
}

ARTIFACTS = {
    "mlx_base_quantized-test.jsonl": "mlx_base_quantized",
    "mlx_lora_adapter-test.jsonl": "mlx_lora_adapter",
    "ollama-baseline.jsonl": "ollama_prompt_baseline",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_artifact(path: Path, expected_ids: set[str]) -> dict:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read {path}: {error}") from error

    if not rows:
        raise ValueError(f"evidence file is empty: {path}")
    ids = [row.get("id") for row in rows]
    if set(ids) != expected_ids or len(ids) != len(set(ids)):
        raise ValueError(f"evidence IDs do not exactly match the test holdout: {path}")

    for row in rows:
        for field in ("id", "response", "expected_diagnosis", "expected_strategy", "latency_ms"):
            if field not in row:
                raise ValueError(f"{path}: row {row.get('id', 'unknown')} missing {field}")
        if not isinstance(row["latency_ms"], (int, float)) or row["latency_ms"] < 0:
            raise ValueError(f"{path}: invalid latency for {row['id']}")

    exact = 0
    keyword = 0
    for row in rows:
        response = row["response"].strip().lower()
        expected = row["expected_strategy"].strip().lower()
        exact += response == expected
        terms = REQUIRED_TERMS.get(row["id"])
        if terms is not None:
            keyword += all(term in response for term in terms)

    latencies = [row["latency_ms"] for row in rows]
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "records": len(rows),
        "exact_strategy_match": f"{exact}/{len(rows)}",
        "keyword_coverage": f"{keyword}/{len(rows)}",
        "latency_ms": {
            "p50": statistics.median(latencies),
            "max": max(latencies),
        },
    }


def build_evidence_manifest() -> dict:
    records, data_bytes = read_dataset()
    validate_records(records)
    dataset_manifest = build_manifest(records, data_bytes)
    test_records = {record["id"]: record for record in records if record["split"] == "test"}
    expected_ids = set(test_records)

    artifacts = {}
    for filename, variant in ARTIFACTS.items():
        path = EVIDENCE_ROOT / filename
        if not path.is_file():
            raise ValueError(f"tracked evidence file not found: {path}")
        summary = summarize_artifact(path, expected_ids)
        summary["variant"] = variant
        for row in [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]:
            source = test_records[row["id"]]
            if row["expected_diagnosis"] != source["diagnosis"] or row["expected_strategy"] != source["fix_strategy"]:
                raise ValueError(f"{path}: expected fields do not match the held-out dataset")
        artifacts[filename] = summary

    return {
        "schema_version": "model-adaptation-lab.evidence.v1",
        "dataset": dataset_manifest,
        "raw_outputs_are_preserved": True,
        "artifacts": artifacts,
        "unavailable_artifacts": {
            "base_model_weights": {
                "status": "not_preserved",
                "sha256": None,
            },
            "lora_adapter_weights": {
                "status": "not_preserved",
                "sha256": None,
            },
        },
        "evaluation_definitions": {
            "exact_strategy_match": "normalized whole-response equality",
            "keyword_coverage": "all explicitly listed per-record terms occur in the response",
            "latency_p50": "statistics.median of recorded generation latency_ms",
        },
        "negative_result": {
            "comparison": "mlx_base_quantized vs mlx_lora_adapter",
            "quality_metric": "keyword_coverage",
            "base": artifacts["mlx_base_quantized-test.jsonl"]["keyword_coverage"],
            "adapter": artifacts["mlx_lora_adapter-test.jsonl"]["keyword_coverage"],
            "conclusion": "no_quality_gain_demonstrated",
            "cause": "not_established_by_this_experiment",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Verify tracked evaluation evidence and print deterministic metadata.")
    parser.add_argument("--write-manifest", type=Path, help="Write the metadata manifest to this path.")
    args = parser.parse_args()
    try:
        manifest = build_evidence_manifest()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    output = json.dumps(manifest, indent=2) + "\n"
    print(output, end="")
    if args.write_manifest:
        target = args.write_manifest
        if target.is_symlink() or target.resolve() == (ROOT / "data" / "rust_errors.jsonl").resolve():
            raise SystemExit("manifest destination cannot be a symlink or the source dataset")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
