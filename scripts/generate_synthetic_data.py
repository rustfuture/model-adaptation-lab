#!/usr/bin/env python3
import json
import os

"""
Synthetic Data Generation Methodology
-------------------------------------
The model-adaptation-lab relies on high-quality structural examples to teach the model 
how to interpret Rust compiler errors and suggest fixes deterministically. 

This script demonstrates how synthetic data can be programmatically expanded.
For this experiment, data is generated cleanly to avoid leakage:
1. 'train' split: indexing and parsing families
2. 'validation' split: ownership family
3. 'test' split: control_flow family

Usage:
    python3 scripts/generate_synthetic_data.py > new_rust_errors.jsonl
"""

def generate_sample(id_str, split, family, code, error_msg, diag, fix, change):
    return {
        "id": id_str,
        "split": split,
        "family": family,
        "error_code": code,
        "compiler_error": error_msg,
        "code": "/* source snippet */",
        "diagnosis": diag,
        "fix_strategy": fix,
        "expected_change": change
    }

if __name__ == "__main__":
    # Example logic to augment data (currently prints a template sample)
    sample = generate_sample(
        "train-synthetic-001",
        "train",
        "macro_expansion",
        "E0123",
        "missingTokens in macro",
        "the macro invocation lacks required tokens",
        "provide the missing tokens to the macro",
        "add tokens to macro"
    )
    print(json.dumps(sample))
