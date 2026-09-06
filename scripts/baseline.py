import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
records = [json.loads(line) for line in (ROOT / "data" / "rust_errors.jsonl").read_text().splitlines()]

# This is deliberately a deterministic plumbing baseline, not a language model.
STRATEGIES = {
    "E0382": "borrow, clone, or reorder the use according to ownership needs",
    "E0515": "return an owned value or move ownership to a longer-lived owner",
    "E0597": "extend the owner lifetime or keep the reference inside the owner scope",
    "E0004": "add the missing variant or a wildcard arm",
}
held_out = [record for record in records if record["split"] == "test"]
correct = sum(STRATEGIES.get(record["error_code"]) == record["fix_strategy"] for record in held_out)
print("baseline=deterministic_error_code_strategy")
print(f"test_records={len(held_out)}")
print(f"exact_strategy_accuracy={correct}/{len(held_out)}")
print("interpretation=deterministic plumbing lower bound; model and fine-tuning results are evaluated separately")
