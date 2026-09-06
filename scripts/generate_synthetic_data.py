#!/usr/bin/env python3
import json

"""
Synthetic Data Generation Methodology
-------------------------------------
The model-adaptation-lab relies on high-quality structural examples to teach the model 
how to interpret Rust compiler errors and suggest fixes deterministically. 

This script demonstrates how synthetic data is programmatically generated with
valid schema adherence and separated testing splits.

Usage:
    python3 scripts/generate_synthetic_data.py > data/rust_errors.jsonl
"""

def generate_sample(id_str, split, family, code, error_msg, diag, fix, change):
    return {
        "id": id_str,
        "split": split,
        "family": family,
        "error_code": "E0000",
        "compiler_error": error_msg,
        "code": code,
        "diagnosis": diag,
        "fix_strategy": fix,
        "expected_change": change
    }

if __name__ == "__main__":
    samples = [
        generate_sample(
            "train-struct-001", "train", "structs", 
            "struct Point { x: i32 } let p = Point { x: 1, y: 2 };", 
            "no field `y` on type `Point`", 
            "the struct is instantiated with an undeclared field", 
            "remove the unknown field or add it to the struct definition", 
            "remove the y field from the instantiation"
        ),
        generate_sample(
            "val-struct-001", "validation", "structs", 
            "struct Color(i32); let c = Color;", 
            "expected function, tuple struct or tuple variant, found struct `Color`", 
            "a tuple struct is instantiated without its fields", 
            "provide the required tuple fields", 
            "add the tuple arguments (0)"
        ),
        generate_sample(
            "test-struct-001", "test", "structs", 
            "struct User { name: String } let u = User { name: \"Bob\" };", 
            "expected `String`, found `&str`", 
            "a struct field requires an owned String but a string slice was provided", 
            "convert the string slice to an owned String", 
            "call .to_string() on the string literal"
        )
    ]
    for s in samples:
        print(json.dumps(s))
