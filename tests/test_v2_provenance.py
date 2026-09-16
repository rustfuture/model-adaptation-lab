"""Deterministic v2 provenance guard (no model weights or MLX required).

The v2 evaluation path must derive its split manifest from the v2 dataset and
the v2 validator. The v1 dataset identity (``data/rust_errors.jsonl``) and the
v1 validator must never appear in a v2 report's provenance.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_mlx_v2  # noqa: E402
import validate_dataset_v2  # noqa: E402
from validate_dataset_v2 import build_manifest, read_dataset  # noqa: E402

V1_DATASET = "data/rust_errors.jsonl"
V2_DATASET = "data/rust_errors_v2.jsonl"


class V2ProvenanceTests(unittest.TestCase):
    def test_v2_evaluator_uses_the_v2_validator(self):
        # The v2 report's dataset.split_manifest is built by this exact callable.
        self.assertIs(evaluate_mlx_v2.build_manifest, validate_dataset_v2.build_manifest)

    def test_real_v2_dataset_manifest_identifies_v2_and_never_v1(self):
        records, data_bytes = read_dataset()
        manifest = build_manifest(records, data_bytes)
        self.assertEqual(manifest["dataset"], V2_DATASET)
        serialized = json.dumps(manifest, sort_keys=True)
        self.assertNotIn(V1_DATASET, serialized)
        # "data/rust_errors_v2.jsonl" does not contain "rust_errors.jsonl", so
        # this catches any leaked v1 dataset name anywhere in the manifest.
        self.assertNotIn("rust_errors.jsonl", serialized)

    def test_minimal_record_manifest_does_not_inherit_v1_identity(self):
        data_bytes = b'{"id": "rec_0", "split": "train", "family": "indexing"}\n'
        records = [{"id": "rec_0", "split": "train", "family": "indexing"}]
        manifest = build_manifest(records, data_bytes)
        self.assertEqual(manifest["dataset"], V2_DATASET)
        self.assertNotIn(V1_DATASET, json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
