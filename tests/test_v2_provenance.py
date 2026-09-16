import os
import sys
from pathlib import Path

# Add scripts to path so we can import
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_dataset_v2 import build_manifest

def test_v2_provenance_manifest():
    # Provide a minimal mock dataset
    data_bytes = b'{"id": "rec_0", "split": "train", "family": "indexing"}\n'
    records = [{"id": "rec_0", "split": "train", "family": "indexing"}]
    
    manifest = build_manifest(records, data_bytes)
    
    assert manifest["dataset"] == "data/rust_errors_v2.jsonl", "v1 provenance leaked into v2 manifest!"

if __name__ == "__main__":
    test_v2_provenance_manifest()
    print("Provenance test passed!")
