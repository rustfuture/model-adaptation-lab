import hashlib
import json
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rust_errors.jsonl"
REQUIRED = {"id", "split", "family", "error_code", "compiler_error", "code", "diagnosis", "fix_strategy", "expected_change"}
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


def main():
    data = DATA.read_bytes()
    records = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    try:
        families = validate_records(records)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({
        "records": len(records),
        "counts": {split: sum(r["split"] == split for r in records) for split in sorted(ALLOWED_SPLITS)},
        "families": {split: sorted(values) for split, values in sorted(families.items())},
        "sha256": hashlib.sha256(data).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
