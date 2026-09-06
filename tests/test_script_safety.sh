#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"

echo "=== Running MLX script safety tests ==="

# 1. Shell syntax check
bash -n "$repo_root/scripts/prepare_mlx_model.sh"
bash -n "$repo_root/scripts/train_mlx_lora.sh"
python3 -m py_compile "$repo_root/scripts/evaluate_mlx.py"
echo "[PASS] Shell and Python syntax checks passed."

# Create temporary test sandbox
tmp_test_dir="$(mktemp -d /tmp/mlx-script-safety-test-XXXXXX)"
trap "rm -rf '$tmp_test_dir'" EXIT

mock_hf="$tmp_test_dir/mock_hf"
mkdir -p "$mock_hf"

# 2. Test: Prepare script rejects root directory
set +e
out=$(HF_MODEL_DIR="$mock_hf" MLX_MODEL_DIR="/" bash "$repo_root/scripts/prepare_mlx_model.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] prepare_mlx_model.sh should reject root directory"; exit 1; }
[[ "$out" == *"cannot be the filesystem root"* ]] || { echo "[FAIL] expected error message for root, got: $out"; exit 1; }
echo "[PASS] prepare_mlx_model.sh correctly rejected root directory."

# 3. Test: Prepare script rejects HOME directory
set +e
out=$(HF_MODEL_DIR="$mock_hf" MLX_MODEL_DIR="$HOME" bash "$repo_root/scripts/prepare_mlx_model.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] prepare_mlx_model.sh should reject HOME directory"; exit 1; }
[[ "$out" == *"cannot be user home directory"* ]] || { echo "[FAIL] expected error message for HOME, got: $out"; exit 1; }
echo "[PASS] prepare_mlx_model.sh correctly rejected HOME directory."

# 4. Test: Prepare script rejects repository root
set +e
out=$(HF_MODEL_DIR="$mock_hf" MLX_MODEL_DIR="$repo_root" bash "$repo_root/scripts/prepare_mlx_model.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] prepare_mlx_model.sh should reject repo root"; exit 1; }
[[ "$out" == *"cannot be inside or contain the repository root"* ]] || { echo "[FAIL] expected error message for repo root, got: $out"; exit 1; }
echo "[PASS] prepare_mlx_model.sh correctly rejected repository root."

# 5. Test: Prepare script rejects source/dest collision
set +e
same_dir="$tmp_test_dir/colliding_model"
mkdir -p "$same_dir"
out=$(HF_MODEL_DIR="$same_dir" MLX_MODEL_DIR="$same_dir" bash "$repo_root/scripts/prepare_mlx_model.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] prepare_mlx_model.sh should reject same source and dest"; exit 1; }
[[ "$out" == *"cannot be the same directory"* ]] || { echo "[FAIL] expected error message for source/dest collision, got: $out"; exit 1; }
echo "[PASS] prepare_mlx_model.sh correctly rejected source/destination collision."

# 6. Test: Train script rejects non-empty adapter directory without ALLOW_OVERWRITE
mock_model="$tmp_test_dir/mock_model"
mkdir -p "$mock_model"
sentinel_dir="$tmp_test_dir/protected_adapter"
mkdir -p "$sentinel_dir"
echo "CRITICAL_SENTINEL_DO_NOT_DELETE" > "$sentinel_dir/sentinel.txt"

set +e
out=$(MLX_LM_LORA_BIN="/usr/bin/true" MLX_MODEL_DIR="$mock_model" MLX_ADAPTER_DIR="$sentinel_dir" bash "$repo_root/scripts/train_mlx_lora.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] train_mlx_lora.sh should reject non-empty adapter directory"; exit 1; }
[[ "$out" == *"exists and is not empty"* ]] || { echo "[FAIL] expected error message for existing adapter dir, got: $out"; exit 1; }
[[ -f "$sentinel_dir/sentinel.txt" ]] || { echo "[FAIL] Sentinel file was deleted!"; exit 1; }
content=$(cat "$sentinel_dir/sentinel.txt")
[[ "$content" == "CRITICAL_SENTINEL_DO_NOT_DELETE" ]] || { echo "[FAIL] Sentinel content corrupted!"; exit 1; }
echo "[PASS] train_mlx_lora.sh preserved existing non-empty directory and sentinel file."

# 7. Test: Train script rejects colliding data and adapter dirs
set +e
colliding_train_dir="$tmp_test_dir/shared_train_dir"
out=$(MLX_LM_LORA_BIN="/usr/bin/true" MLX_MODEL_DIR="$mock_model" MLX_DATA_DIR="$colliding_train_dir" MLX_ADAPTER_DIR="$colliding_train_dir" bash "$repo_root/scripts/train_mlx_lora.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] train_mlx_lora.sh should reject identical data and adapter dirs"; exit 1; }
[[ "$out" == *"cannot be the same"* ]] || { echo "[FAIL] expected error message for colliding dirs, got: $out"; exit 1; }
echo "[PASS] train_mlx_lora.sh correctly rejected colliding data and adapter directories."

# 8. Test: Evaluate script requires existing model and dataset
set +e
out=$(python3 "$repo_root/scripts/evaluate_mlx.py" --model "$tmp_test_dir/nonexistent_model" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] evaluate_mlx.py should exit on nonexistent model"; exit 1; }
[[ "$out" == *"MLX model directory not found"* ]] || { echo "[FAIL] expected error for nonexistent model, got: $out"; exit 1; }
echo "[PASS] evaluate_mlx.py verified model directory existence."

# 9. Test: Evaluate script requires existing data file
set +e
out=$(python3 "$repo_root/scripts/evaluate_mlx.py" --data "$tmp_test_dir/nonexistent_data.jsonl" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] evaluate_mlx.py should exit on nonexistent data"; exit 1; }
[[ "$out" == *"Dataset file not found"* ]] || { echo "[FAIL] expected error for nonexistent dataset, got: $out"; exit 1; }
echo "[PASS] evaluate_mlx.py verified dataset file existence."

echo "=== All 9 MLX safety tests passed successfully ==="

