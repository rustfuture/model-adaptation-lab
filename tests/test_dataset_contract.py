import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_dataset import DATA, build_manifest, validate_records


class DatasetContractTests(unittest.TestCase):
    def setUp(self):
        self.records = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]

    def test_authored_corpus_is_valid(self):
        validate_records(self.records)

    def test_every_pair_of_splits_must_be_disjoint(self):
        for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            with self.subTest(left=left, right=right):
                records = copy.deepcopy(self.records)
                family = next(r["family"] for r in records if r["split"] == left)
                next(r for r in records if r["split"] == right)["family"] = family
                with self.assertRaisesRegex(ValueError, "family leakage"):
                    validate_records(records)

    def test_empty_fields_and_duplicate_ids_are_rejected(self):
        records = copy.deepcopy(self.records)
        records[0]["code"] = " "
        with self.assertRaisesRegex(ValueError, "non-empty field"):
            validate_records(records)
        with self.assertRaisesRegex(ValueError, "duplicate record id"):
            validate_records(self.records + [self.records[0]])

    def test_export_preserves_corpus_and_does_not_modify_source(self):
        before = DATA.read_bytes()
        result = subprocess.run([sys.executable, str(ROOT / "scripts/generate_synthetic_data.py")], check=True, capture_output=True, text=True)
        exported = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(exported, self.records)
        self.assertEqual(DATA.read_bytes(), before)
        validate_records(exported)

    def test_mlx_exports_are_exact_split_projections(self):
        expected = {
            split: {record["id"] for record in self.records if record["split"] == split}
            for split in ("train", "validation", "test")
        }
        paths = {
            "train": ROOT / "data" / "mlx" / "train.jsonl",
            "validation": ROOT / "data" / "mlx" / "valid.jsonl",
            "test": ROOT / "data" / "mlx" / "test.jsonl",
        }
        actual = {}
        for split, path in paths.items():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            actual[split] = {row["id"] for row in rows}
            self.assertEqual(actual[split], expected[split])
            self.assertEqual(len(rows), len(actual[split]))
        self.assertTrue(actual["test"].isdisjoint(actual["train"] | actual["validation"]))

    def test_tracked_dataset_manifest_matches_source(self):
        manifest_path = ROOT / "evidence" / "dataset-validation.json"
        manifest = json.loads(manifest_path.read_text())
        expected = build_manifest(self.records, DATA.read_bytes())
        self.assertEqual(manifest, expected)

    def test_existing_empty_run_is_not_owned_or_removed(self):
        run_id = "ownership_test_" + uuid.uuid4().hex
        run_dir = Path("/tmp/model-lab-runs") / run_id
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir()
        try:
            with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
                model = Path(temporary) / "model"
                model.mkdir()
                env = os.environ.copy()
                for key in ["MLX_DATA_DIR", "MLX_ADAPTER_DIR"]:
                    env.pop(key, None)
                env.update(MLX_MODEL_DIR=str(model), MLX_RUN_ID=run_id, MLX_LM_LORA_BIN="/usr/bin/true")
                result = subprocess.run(["bash", str(ROOT / "scripts/train_mlx_lora.sh")], env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn("ownership was not acquired", result.stderr)
                self.assertTrue(run_dir.is_dir())
                self.assertEqual(list(run_dir.iterdir()), [])
        finally:
            run_dir.rmdir()
