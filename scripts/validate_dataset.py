import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rust_errors.jsonl"
REQUIRED = {"id", "split", "family", "error_code", "compiler_error", "code", "diagnosis", "fix_strategy", "expected_change"}
ALLOWED_SPLITS = {"train", "validation", "test"}

records = []
for line_number, line in enumerate(DATA.read_text().splitlines(), 1):
    record = json.loads(line)
    missing = REQUIRED - record.keys()
    if missing:
        raise SystemExit(f"line {line_number}: missing fields {sorted(missing)}")
    if record["split"] not in ALLOWED_SPLITS:
        raise SystemExit(f"line {line_number}: invalid split")
    records.append(record)

ids = [record["id"] for record in records]
if len(ids) != len(set(ids)):
    raise SystemExit("duplicate record id")
families_by_split = {}
for record in records:
    families_by_split.setdefault(record["split"], set()).add(record["family"])
if families_by_split.get("train", set()) & families_by_split.get("test", set()):
    raise SystemExit("train/test family leakage")
if families_by_split.get("validation", set()) & families_by_split.get("test", set()):
    raise SystemExit("validation/test family leakage")

digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
print(json.dumps({"records": len(records), "counts": {split: sum(r["split"] == split for r in records) for split in sorted(ALLOWED_SPLITS)}, "families": {split: sorted(families) for split, families in sorted(families_by_split.items())}, "sha256": digest}, indent=2))
