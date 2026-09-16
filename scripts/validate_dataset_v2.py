import hashlib
import json
import argparse
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rust_errors_v2.jsonl"
REQUIRED = {"id", "split", "family", "error_code", "compiler_error", "code", "diagnosis", "fix_strategy", "expected_change", "fixed_code", "fixed_code"}
ALLOWED_SPLITS = {"train", "validation", "test"}


def validate_records(records):
    if not records:
        raise ValueError("dataset must not be empty")
    ids = set()
    families = {split: set() for split in ALLOWED_SPLITS}
    for number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"record {number}: expected object")
        for field in REQUIRED:
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"record {number}: missing non-empty field {field}")
        if record["split"] not in ALLOWED_SPLITS:
            raise ValueError(f"record {number}: invalid split")
        if record["id"] in ids:
            raise ValueError("duplicate record id")
        ids.add(record["id"])
        families[record["split"]].add(record["family"])
    for split, values in families.items():
        if not values:
            raise ValueError(f"missing split: {split}")
    for left, right in combinations(sorted(ALLOWED_SPLITS), 2):
        if families[left] & families[right]:
            raise ValueError(f"{left}/{right} family leakage")
    return families


def build_manifest(records, data_bytes):
    """Return portable, deterministic evidence for the dataset contract."""
    split_records = {
        split: [record for record in records if record["split"] == split]
        for split in sorted(ALLOWED_SPLITS)
    }
    split_ids = {
        split: [record["id"] for record in rows]
        for split, rows in split_records.items()
    }
    split_families = {
        split: sorted({record["family"] for record in rows})
        for split, rows in split_records.items()
    }
    # Hash canonical record IDs rather than absolute paths or platform-specific
    # line endings. The complete dataset hash below still covers the source file.
    split_hashes = {
        split: hashlib.sha256(
            ("\n".join(split_ids[split]) + "\n").encode("utf-8")
        ).hexdigest()
        for split in split_ids
    }
    return {
        "schema_version": "model-adaptation-lab.dataset-validation.v1",
        "dataset": "data/rust_errors_v2.jsonl",
        "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "records": len(records),
        "split_counts": {split: len(rows) for split, rows in split_records.items()},
        "split_families": split_families,
        "split_record_ids": split_ids,
        "split_record_id_sha256": split_hashes,
        "family_disjoint": True,
        "test_holdout_record_ids": split_ids["test"],
    }


def read_dataset(path=DATA):
    data_bytes = path.read_bytes()
    records = [
        json.loads(line)
        for line in data_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    return records, data_bytes


def main():
    parser = argparse.ArgumentParser(description="Validate the authored dataset split contract.")
    parser.add_argument(
        "--write-manifest",
        type=Path,
        help="Write deterministic split evidence to this file after validation.",
    )
    args = parser.parse_args()

    try:
        records, data = read_dataset()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"failed to read dataset: {error}") from error
    try:
        validate_records(records)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    manifest = build_manifest(records, data)
    # Preserve the concise legacy field names in stdout while exposing the
    # complete machine-readable manifest to callers that need split proof.
    print(json.dumps({
        "records": manifest["records"],
        "counts": manifest["split_counts"],
        "families": manifest["split_families"],
        "sha256": manifest["dataset_sha256"],
    }, indent=2))

    if args.write_manifest:
        target = args.write_manifest
        if target.is_symlink() or target.resolve() == DATA.resolve():
            raise SystemExit("manifest destination cannot be a symlink or the source dataset")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote dataset manifest: {target}")


if __name__ == "__main__":
    main()
