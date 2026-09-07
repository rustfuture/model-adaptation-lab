#!/usr/bin/env python3
"""Export the existing authored corpus after validating its split contract.

This legacy filename is retained for compatibility. This is NOT a generator
of new training examples and does not verify snippets with rustc.
Run without output redirection to inspect the corpus, or redirect to a NEW
file. Never redirect to data/rust_errors.jsonl: the shell truncates it first.
"""
import json

from validate_dataset import DATA, validate_records


def main():
    records = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    validate_records(records)
    for record in records:
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
