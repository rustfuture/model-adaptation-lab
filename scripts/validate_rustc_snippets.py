#!/usr/bin/env python3
"""Compile every authored snippet and require its declared rustc error code."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rust_errors.jsonl"
ERROR_RE = re.compile(r"error\[E(\d{4})\]")


def main() -> int:
    records = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="rust-error-validation-") as directory:
        source = Path(directory) / "snippet.rs"
        for record in records:
            source.write_text(f"fn main() {{ {record['code']} }}\n")
            result = subprocess.run(
                ["rustc", "--edition", "2021", str(source)],
                capture_output=True, text=True, check=False
            )
            codes = set(ERROR_RE.findall(result.stderr))
            expected = record["error_code"][1:]
            if result.returncode == 0 or expected not in codes:
                failures.append(
                    f"{record['id']}: expected {record['error_code']}, "
                    f"got {sorted('E' + code for code in codes) or ['no errors']}"
                )
    if failures:
        print("rustc snippet validation failed:\n" + "\n".join(failures))
        return 1
    print(f"rustc snippet validation passed: {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
