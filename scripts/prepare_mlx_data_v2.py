import json
from pathlib import Path

from validate_dataset_v2 import validate_records

ROOT = Path(__file__).resolve().parents[1]
data_file = ROOT / "data" / "rust_errors_v2.jsonl"
out_dir = ROOT / "data" / "mlx_v2"

records = [json.loads(line) for line in data_file.read_text().splitlines() if line.strip()]
validate_records(records)
out_dir.mkdir(exist_ok=True, parents=True)

system_prompt = (
    "You are evaluating a Rust compiler-error example. "
    "Provide a concise diagnosis and conservative fix strategy, then provide the fixed code inside a markdown ```rust code block. "
    "Do not include fn main() in the block if it wasn't there."
)

split_records = {"train": [], "validation": [], "test": []}

for record in records:
    split = record["split"]
    if split not in split_records:
        continue
    user_content = (
        f"error_code: {record['error_code']}\n"
        f"compiler_error: {record['compiler_error']}\n"
        f"code:\n{record['code']}"
    )
    assistant_content = (
        f"diagnosis: {record['diagnosis']}\n"
        f"fix_strategy: {record['fix_strategy']}"
    )
    item = {
        "id": record["id"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }
    split_records[split].append(item)

for split, target_name in [("train", "train.jsonl"), ("validation", "valid.jsonl"), ("test", "test.jsonl")]:
    target_path = out_dir / target_name
    lines = [json.dumps(row, ensure_ascii=False) for row in split_records[split]]
    target_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} records to {target_path}")
