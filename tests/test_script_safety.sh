#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"

echo "=== Running Comprehensive MLX Script Safety & Evaluator Test Suite ==="

# 1. Shell & Python syntax validation
bash -n "$repo_root/scripts/prepare_mlx_model.sh"
bash -n "$repo_root/scripts/train_mlx_lora.sh"
python3 -m py_compile "$repo_root/scripts/evaluate_mlx.py"
python3 -m py_compile "$repo_root/scripts/baseline.py"
python3 -m py_compile "$repo_root/scripts/validate_dataset.py"
python3 -m py_compile "$repo_root/scripts/prepare_mlx_data.py"

# Optional pyflakes check if available
if command -v pyflakes >/dev/null 2>&1; then
  pyflakes "$repo_root/scripts/"
  echo "[PASS] pyflakes static analysis passed with zero undefined names."
elif [[ -x "/tmp/model-lab-mlx-venv/bin/pyflakes" ]]; then
  /tmp/model-lab-mlx-venv/bin/pyflakes "$repo_root/scripts/"
  echo "[PASS] venv pyflakes static analysis passed with zero undefined names."
fi
echo "[PASS] Shell and Python syntax verification passed."

# Create isolated test environment with strict ownership tracking
test_sandbox="$(mktemp -d /tmp/mlx-suite-sandbox-XXXXXX)"

# Only track run dirs that we atomically create and confirm exist
declare -a CREATED_RUN_DIRS=()

cleanup() {
  rm -rf "$test_sandbox"
  # Only clean dirs that we confirmed we created
  for rdir in ${CREATED_RUN_DIRS+"${CREATED_RUN_DIRS[@]}"}; do
    if [[ -d "$rdir" ]]; then
      rm -rf "$rdir"
    fi
  done
}
trap cleanup EXIT

mock_model="$test_sandbox/mock_model"
mkdir -p "$mock_model"
cat <<'JSON' > "$mock_model/config.json"
{
  "model_type": "qwen2",
  "architectures": ["Qwen2ForCausalLM"]
}
JSON

mock_adapter="$test_sandbox/mock_adapter"
mkdir -p "$mock_adapter"
cat <<'JSON' > "$mock_adapter/adapter_config.json"
{
  "adapter_path": "/tmp/mock_adapter",
  "batch_size": 1,
  "fine_tune_type": "lora",
  "iters": 50,
  "learning_rate": 0.0001,
  "lora_parameters": {
    "rank": 8,
    "scale": 20.0,
    "dropout": 0.0
  },
  "model": "/tmp/mock_model"
}
JSON

# Setup mock mlx_lm package for Python tests
mock_site_packages="$test_sandbox/mock_packages"
mkdir -p "$mock_site_packages/mlx_lm"
cat <<'PY' > "$mock_site_packages/mlx_lm/__init__.py"
class MockModel:
    pass

class MockTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "mock_prompt"

def load(model_path, adapter_path=None):
    return MockModel(), MockTokenizer()

def generate(model, tokenizer, prompt, max_tokens=150, verbose=False):
    return "diagnosis: mock diagnosed error\nfix_strategy: match none and use branches"
PY

# Setup mock MLX LoRA executable
mock_bin_dir="$test_sandbox/mock_bins"
mkdir -p "$mock_bin_dir"
mock_lora_exec="$mock_bin_dir/mock_mlx_lm_lora"
cat <<'SH' > "$mock_lora_exec"
#!/usr/bin/env bash
adapter_path=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapter-path) adapter_path="$2"; shift 2 ;;
    *) shift ;;
  esac
done
mkdir -p "$adapter_path"
echo '{"lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0}, "iters": 50, "fine_tune_type": "lora"}' > "$adapter_path/adapter_config.json"
echo "mock weights" > "$adapter_path/adapters.safetensors"
SH
chmod +x "$mock_lora_exec"

# 2. Test: Rejection of root, HOME, and repo root
set +e
out=$(MLX_MODEL_DIR="/" bash "$repo_root/scripts/prepare_mlx_model.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] prepare_mlx_model.sh should reject root"; exit 1; }
[[ "$out" == *"cannot be the filesystem root"* ]] || { echo "[FAIL] unexpected root error: $out"; exit 1; }

set +e
out=$(MLX_MODEL_DIR="$HOME" bash "$repo_root/scripts/prepare_mlx_model.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] prepare_mlx_model.sh should reject HOME"; exit 1; }
[[ "$out" == *"cannot be user home directory"* ]] || { echo "[FAIL] unexpected HOME error: $out"; exit 1; }

set +e
out=$(MLX_MODEL_DIR="$repo_root" bash "$repo_root/scripts/prepare_mlx_model.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] prepare_mlx_model.sh should reject repo root"; exit 1; }
[[ "$out" == *"cannot be inside or contain the repository root"* ]] || { echo "[FAIL] unexpected repo root error: $out"; exit 1; }
echo "[PASS] Root, HOME, and repo-root guards verified."

# 3. Test: Existing non-empty destination rejection (No destructive deletion)
existing_nonempty="$test_sandbox/existing_model"
mkdir -p "$existing_nonempty"
echo "SENTINEL_DO_NOT_DELETE" > "$existing_nonempty/model_weight.bin"

set +e
out=$(MLX_MODEL_DIR="$existing_nonempty" bash "$repo_root/scripts/prepare_mlx_model.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] prepare_mlx_model.sh should reject existing non-empty directory"; exit 1; }
[[ "$out" == *"exists and is not empty"* ]] || { echo "[FAIL] unexpected non-empty error: $out"; exit 1; }
[[ -f "$existing_nonempty/model_weight.bin" ]] || { echo "[FAIL] Sentinel file was deleted!"; exit 1; }
echo "[PASS] Non-empty model directory protection verified (sentinel preserved)."

# 4. Test: Path collision and nested path rejection with sentinel preservation
set +e
shared_dir="$test_sandbox/shared_path"
mkdir -p "$shared_dir"
echo "SENTINEL_DATA" > "$shared_dir/sentinel.txt"
out=$(MLX_LM_LORA_BIN="$mock_lora_exec" MLX_MODEL_DIR="$mock_model" MLX_DATA_DIR="$shared_dir" MLX_ADAPTER_DIR="$shared_dir" bash "$repo_root/scripts/train_mlx_lora.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] train_mlx_lora.sh should reject identical data and adapter dirs"; exit 1; }
[[ "$out" == *"cannot be the same directory"* ]] || { echo "[FAIL] unexpected collision error: $out"; exit 1; }
[[ -f "$shared_dir/sentinel.txt" ]] || { echo "[FAIL] Sentinel file in colliding dir was destroyed!"; exit 1; }

# Nested path test
nested_sub="$shared_dir/nested_adapter"
set +e
out=$(MLX_LM_LORA_BIN="$mock_lora_exec" MLX_MODEL_DIR="$mock_model" MLX_DATA_DIR="$shared_dir" MLX_ADAPTER_DIR="$nested_sub" bash "$repo_root/scripts/train_mlx_lora.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] train_mlx_lora.sh should reject nested adapter dir inside data dir"; exit 1; }
[[ "$out" == *"cannot be inside"* ]] || { echo "[FAIL] unexpected nested dir error: $out"; exit 1; }
echo "[PASS] Path collision and nested path rejection verified."

# 5. Test: Read-only validation (No mutations on pre-validation failures)
precheck_fail_dir="$test_sandbox/fail_target_parent/fail_child"
set +e
MLX_MODEL_DIR="/" MLX_DATA_DIR="$precheck_fail_dir" bash "$repo_root/scripts/train_mlx_lora.sh" >/dev/null 2>&1 || true
set -e
[[ ! -d "$test_sandbox/fail_target_parent" ]] || { echo "[FAIL] Pre-validation must not create directories!"; exit 1; }

# Also verify that invalid train arguments create NO run directory under /tmp/model-lab-runs
mkdir -p /tmp/model-lab-runs
runs_before=$(find /tmp/model-lab-runs -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
set +e
MLX_MODEL_DIR="/tmp/nonexistent_model_${RANDOM}" bash "$repo_root/scripts/train_mlx_lora.sh" >/dev/null 2>&1 || true
set -e
runs_after=$(find /tmp/model-lab-runs -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
[[ "$runs_before" -eq "$runs_after" ]] || { echo "[FAIL] Invalid train input created orphan run directory!"; exit 1; }

# Also verify that invalid report-name does NOT create output-dir in evaluate_mlx.py
never_created_dir="$test_sandbox/never_created_dir_${RANDOM}"
set +e
PYTHONPATH="$mock_site_packages" python3 "$repo_root/scripts/evaluate_mlx.py" \
  --model "$mock_model" \
  --baseline-only \
  --output-dir "$never_created_dir" \
  --report-name "../bad_escape.json" >/dev/null 2>&1 || true
set -e
[[ ! -d "$never_created_dir" ]] || { echo "[FAIL] evaluate_mlx.py created output directory on invalid report name!"; exit 1; }
echo "[PASS] Pre-validation mutation-free guarantee verified."

# 6. Test: Sentinel protection, truly concurrent workers, and MLX_RUN_ID reuse rejection
sentinel_run_id="sentinel_isolated_${$}_${RANDOM}"
sentinel_dir="/tmp/model-lab-runs/$sentinel_run_id"
mkdir -p "$sentinel_dir"
echo "CANARY_DATA_DO_NOT_DELETE" > "$sentinel_dir/canary.txt"

# --- Truly concurrent workers: run in background and wait ---
worker1_run_id="worker1_concurrent_${$}_${RANDOM}"
worker2_run_id="worker2_concurrent_${$}_${RANDOM}"

# Launch both workers concurrently in background
MLX_LM_LORA_BIN="$mock_lora_exec" \
MLX_MODEL_DIR="$mock_model" \
MLX_RUN_ID="$worker1_run_id" \
bash "$repo_root/scripts/train_mlx_lora.sh" >/dev/null &
pid1=$!

MLX_LM_LORA_BIN="$mock_lora_exec" \
MLX_MODEL_DIR="$mock_model" \
MLX_RUN_ID="$worker2_run_id" \
bash "$repo_root/scripts/train_mlx_lora.sh" >/dev/null &
pid2=$!

# Wait for both and capture exit codes
set +e
wait "$pid1"; exit1=$?
wait "$pid2"; exit2=$?
set -e

[[ $exit1 -eq 0 ]] || { echo "[FAIL] Concurrent worker 1 failed with exit $exit1!"; exit 1; }
[[ $exit2 -eq 0 ]] || { echo "[FAIL] Concurrent worker 2 failed with exit $exit2!"; exit 1; }

# Track for cleanup only after confirming they were created
if [[ -d "/tmp/model-lab-runs/$worker1_run_id" ]]; then
  CREATED_RUN_DIRS+=("/tmp/model-lab-runs/$worker1_run_id")
fi
if [[ -d "/tmp/model-lab-runs/$worker2_run_id" ]]; then
  CREATED_RUN_DIRS+=("/tmp/model-lab-runs/$worker2_run_id")
fi

# Assert both runs produced distinct valid manifests
[[ -f "/tmp/model-lab-runs/$worker1_run_id/run_manifest.json" ]] || { echo "[FAIL] Concurrent worker 1 manifest missing!"; exit 1; }
[[ -f "/tmp/model-lab-runs/$worker2_run_id/run_manifest.json" ]] || { echo "[FAIL] Concurrent worker 2 manifest missing!"; exit 1; }

# Assert one worker's cleanup doesn't affect the other
# Clean up worker 1 and verify worker 2 is intact
rm -rf "/tmp/model-lab-runs/$worker1_run_id"
[[ -f "/tmp/model-lab-runs/$worker2_run_id/run_manifest.json" ]] || { echo "[FAIL] Worker 2 was affected by Worker 1 cleanup!"; exit 1; }

# Assert pre-existing sentinel was completely untouched by concurrent runs
[[ -f "$sentinel_dir/canary.txt" ]] || { echo "[FAIL] Sentinel canary file was destroyed by concurrent runs!"; exit 1; }
sentinel_content="$(cat "$sentinel_dir/canary.txt")"
[[ "$sentinel_content" == "CANARY_DATA_DO_NOT_DELETE" ]] || { echo "[FAIL] Sentinel canary content altered! Got: $sentinel_content"; exit 1; }

# Explicit MLX_RUN_ID reuse rejection test (worker2 still exists)
set +e
out=$(MLX_LM_LORA_BIN="$mock_lora_exec" MLX_MODEL_DIR="$mock_model" MLX_RUN_ID="$worker2_run_id" bash "$repo_root/scripts/train_mlx_lora.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] train_mlx_lora.sh should reject reusing existing MLX_RUN_ID"; exit 1; }
[[ "$out" == *"already exists for explicit MLX_RUN_ID"* ]] || { echo "[FAIL] unexpected reuse error: $out"; exit 1; }

rm -rf "/tmp/model-lab-runs/$worker2_run_id"
rm -rf "$sentinel_dir"
echo "[PASS] Truly concurrent workers, sentinel preservation, and MLX_RUN_ID reuse rejection verified."

# 6b. Test: Sentinel byte-for-byte preservation after failure + EXIT cleanup
failure_sentinel_dir="$test_sandbox/failure_sentinel_workspace"
mkdir -p "$failure_sentinel_dir"
failure_sentinel_file="$failure_sentinel_dir/precious_data.bin"
echo "FAILURE_SENTINEL_BYTE_FOR_BYTE_EXACT" > "$failure_sentinel_file"
expected_sentinel_hash="$(shasum -a 256 "$failure_sentinel_file" | cut -d' ' -f1)"

# Run train with invalid model (will fail) — the EXIT trap should not touch our sentinel
set +e
MLX_LM_LORA_BIN="$mock_lora_exec" \
MLX_MODEL_DIR="/tmp/nonexistent_model_${$}_${RANDOM}" \
MLX_DATA_DIR="$failure_sentinel_dir" \
bash "$repo_root/scripts/train_mlx_lora.sh" >/dev/null 2>&1
set -e

# Verify sentinel is byte-for-byte intact after failure + EXIT trap cleanup
[[ -f "$failure_sentinel_file" ]] || { echo "[FAIL] Sentinel file destroyed after failure + EXIT cleanup!"; exit 1; }
actual_sentinel_hash="$(shasum -a 256 "$failure_sentinel_file" | cut -d' ' -f1)"
[[ "$expected_sentinel_hash" == "$actual_sentinel_hash" ]] || {
  echo "[FAIL] Sentinel content modified after failure! Expected hash: $expected_sentinel_hash, got: $actual_sentinel_hash"
  exit 1
}
echo "[PASS] Sentinel byte-for-byte preserved after failure + EXIT cleanup."

# 6c. Test: Same explicit MLX_RUN_ID race — atomic ownership, loser must not delete winner
race_run_id="race_test_${$}_${RANDOM}"
race_dir="/tmp/model-lab-runs/$race_run_id"

# First process wins: create it normally
MLX_LM_LORA_BIN="$mock_lora_exec" \
MLX_MODEL_DIR="$mock_model" \
MLX_RUN_ID="$race_run_id" \
bash "$repo_root/scripts/train_mlx_lora.sh" >/dev/null
CREATED_RUN_DIRS+=("$race_dir")

# Verify winner's manifest exists
[[ -f "$race_dir/run_manifest.json" ]] || { echo "[FAIL] Race winner manifest missing!"; exit 1; }
winner_manifest_hash="$(shasum -a 256 "$race_dir/run_manifest.json" | cut -d' ' -f1)"

# Second process (loser) tries same ID — must be rejected
set +e
out=$(MLX_LM_LORA_BIN="$mock_lora_exec" MLX_MODEL_DIR="$mock_model" MLX_RUN_ID="$race_run_id" bash "$repo_root/scripts/train_mlx_lora.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] Loser process should have been rejected!"; exit 1; }

# Verify loser did NOT delete winner's files
[[ -f "$race_dir/run_manifest.json" ]] || { echo "[FAIL] Loser deleted winner's manifest!"; exit 1; }
post_race_manifest_hash="$(shasum -a 256 "$race_dir/run_manifest.json" | cut -d' ' -f1)"
[[ "$winner_manifest_hash" == "$post_race_manifest_hash" ]] || {
  echo "[FAIL] Loser modified winner's manifest! Before: $winner_manifest_hash, After: $post_race_manifest_hash"
  exit 1
}
rm -rf "$race_dir"
echo "[PASS] Same MLX_RUN_ID race: atomic ownership, loser cannot delete winner's files."

# 6d. Test: Explicit vs default path collision caught pre-allocation (Item 4 regression)
preflight_run_id="preflight_collision_${$}_${RANDOM}"
runs_before=$(find /tmp/model-lab-runs -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')

set +e
out=$(MLX_LM_LORA_BIN="$mock_lora_exec" \
  MLX_MODEL_DIR="$mock_model" \
  MLX_RUN_ID="$preflight_run_id" \
  MLX_DATA_DIR="/tmp/model-lab-runs/${preflight_run_id}/adapters" \
  bash "$repo_root/scripts/train_mlx_lora.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] Explicit data dir == default adapter dir collision not caught!"; exit 1; }
[[ "$out" == *"cannot be the same directory"* ]] || [[ "$out" == *"cannot be inside"* ]] || {
  echo "[FAIL] Expected collision error for explicit vs default path, got: $out"
  exit 1
}

# Verify no orphan run dir was created
runs_after=$(find /tmp/model-lab-runs -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
[[ "$runs_before" -eq "$runs_after" ]] || {
  echo "[FAIL] Pre-flight collision left orphan run directory! Before: $runs_before, After: $runs_after"
  exit 1
}

# Also test reverse: explicit adapter dir == default data dir
preflight_run_id2="preflight_reverse_${$}_${RANDOM}"
set +e
out=$(MLX_LM_LORA_BIN="$mock_lora_exec" \
  MLX_MODEL_DIR="$mock_model" \
  MLX_RUN_ID="$preflight_run_id2" \
  MLX_ADAPTER_DIR="/tmp/model-lab-runs/${preflight_run_id2}/data" \
  bash "$repo_root/scripts/train_mlx_lora.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] Reverse explicit vs default collision not caught!"; exit 1; }
echo "[PASS] Explicit vs default path collision caught before allocation (no orphan dirs)."

# 6e. Test: Post-allocation failure cleans up run dir (cleanup trap)
postalloc_run_id="postalloc_cleanup_${$}_${RANDOM}"
set +e
# Use a nonexistent lora binary to trigger failure AFTER allocation
MLX_LM_LORA_BIN="/tmp/nonexistent_lora_binary_${RANDOM}" \
MLX_MODEL_DIR="$mock_model" \
MLX_RUN_ID="$postalloc_run_id" \
bash "$repo_root/scripts/train_mlx_lora.sh" >/dev/null 2>&1
set -e
# The cleanup trap should have removed the allocated run directory
[[ ! -d "/tmp/model-lab-runs/$postalloc_run_id" ]] || {
  echo "[FAIL] Post-allocation failure did not clean up run directory!"
  rm -rf "/tmp/model-lab-runs/$postalloc_run_id"
  exit 1
}
echo "[PASS] Post-allocation failure cleanup trap works correctly."

# 7. Test: Real symlink guards in evaluate_mlx.py (pointing inside and outside)
symlink_test_dir="$test_sandbox/symlink_guards"
mkdir -p "$symlink_test_dir"

# Inside symlink test
ln -s internal_target.json "$symlink_test_dir/symlink_inside.json"
set +e
out=$(PYTHONPATH="$mock_site_packages" python3 "$repo_root/scripts/evaluate_mlx.py" \
  --model "$mock_model" \
  --baseline-only \
  --output-dir "$symlink_test_dir" \
  --report-name "symlink_inside.json" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] evaluate_mlx.py should reject symlink inside output_dir"; exit 1; }
[[ "$out" == *"Target report path cannot be a symlink"* ]] || { echo "[FAIL] unexpected inside symlink error: $out"; exit 1; }

# Outside symlink test
ln -s /tmp/outside_target.json "$symlink_test_dir/symlink_outside.json"
set +e
out=$(PYTHONPATH="$mock_site_packages" python3 "$repo_root/scripts/evaluate_mlx.py" \
  --model "$mock_model" \
  --baseline-only \
  --output-dir "$symlink_test_dir" \
  --report-name "symlink_outside.json" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] evaluate_mlx.py should reject symlink outside output_dir"; exit 1; }
[[ "$out" == *"Target report path cannot be a symlink"* ]] || { echo "[FAIL] unexpected outside symlink error: $out"; exit 1; }
echo "[PASS] Real symlink guards (inside and outside targets) verified."

# 8. Test: Overwrite protection against protected inputs
set +e
out=$(PYTHONPATH="$mock_site_packages" python3 "$repo_root/scripts/evaluate_mlx.py" \
  --model "$mock_model" \
  --baseline-only \
  --output-dir "$mock_model" \
  --overwrite 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] evaluate_mlx.py should reject --output-dir == protected input"; exit 1; }
[[ "$out" == *"conflicts with protected input path"* ]] || { echo "[FAIL] unexpected overwrite error: $out"; exit 1; }
[[ -f "$mock_model/config.json" ]] || { echo "[FAIL] Protected model config was destroyed!"; exit 1; }
echo "[PASS] Overwrite protection against protected inputs verified."

# 9. Test: Evaluator full successful pipeline with mock MLX backend and provenance verification
eval_out_dir="$test_sandbox/eval_output"
mkdir -p "$eval_out_dir"
PYTHONPATH="$mock_site_packages" python3 "$repo_root/scripts/evaluate_mlx.py" \
  --model "$mock_model" \
  --adapter "$mock_adapter" \
  --output-dir "$eval_out_dir" \
  --report-name "test-report.json" > "$test_sandbox/eval_stdout.json"

[[ -f "$eval_out_dir/test-report.json" ]] || { echo "[FAIL] test-report.json not produced!"; exit 1; }

# Verify schema, metadata, and new provenance fields in output report
python3 -c "
import json
with open('$eval_out_dir/test-report.json') as f:
    r = json.load(f)

assert r['model']['metadata']['model_type'] == 'qwen2'
assert r['adapter']['metadata']['rank'] == 8
assert r['adapter']['metadata']['scale'] == 20.0
assert r['adapter']['metadata']['iters'] == 50
assert 'strategy_substring_contained' in r['fine_tuned']
assert 'strategy_exact_match' in r['fine_tuned']
assert 'manual_keyword_coverage' in r['fine_tuned']
assert r['parameters']['max_tokens'] == 150

# Provenance assertions
assert 'git_commit' in r and len(r['git_commit']) > 0
assert 'git_dirty' in r
assert 'evaluator' in r and len(r['evaluator']['sha256']) == 64
assert 'environment' in r and 'python_version' in r['environment']
assert 'invocation' in r and 'command' in r['invocation']

print('[PASS] Mock MLX evaluation report metadata, schema, and provenance successfully verified.')
"

# 10. Test: Unknown sample ID handling in evaluator
dataset_with_unknown="$test_sandbox/dataset_unknown.jsonl"
python3 -c "
import json
with open('$repo_root/data/rust_errors.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
# Add unknown test record
lines.append({
    'id': 'unknown-test-sample-999',
    'split': 'test',
    'family': 'novel_family',
    'error_code': 'E0999',
    'compiler_error': 'novel compiler error',
    'code': 'fn test() {}',
    'diagnosis': 'novel diagnosis',
    'fix_strategy': 'novel fix'
})
with open('$dataset_with_unknown', 'w') as f:
    for item in lines:
        f.write(json.dumps(item) + '\n')
"

PYTHONPATH="$mock_site_packages" python3 "$repo_root/scripts/evaluate_mlx.py" \
  --model "$mock_model" \
  --adapter "$mock_adapter" \
  --data "$dataset_with_unknown" \
  --output-dir "$test_sandbox/unknown_eval_out" \
  --report-name "unknown-report.json" >/dev/null

python3 -c "
import json
with open('$test_sandbox/unknown_eval_out/unknown-report.json') as f:
    r = json.load(f)
cov = r['fine_tuned']['manual_keyword_coverage']
assert 'unmeasured' in cov, f'Expected unmeasured annotation in keyword coverage, got: {cov}'
print('[PASS] Unknown sample ID properly recorded as unmeasured in manual_keyword_coverage.')
"

# 11. Test: Missing required field in dataset record fails explicitly
dataset_missing_field="$test_sandbox/dataset_missing.jsonl"
python3 -c "
import json
with open('$repo_root/data/rust_errors.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
for r in lines:
    if r.get('split') == 'test':
        r['fix_strategy'] = ''  # Empty required field in held-out test split
        break
with open('$dataset_missing_field', 'w') as f:
    for item in lines:
        f.write(json.dumps(item) + '\n')
"
set +e
out=$(PYTHONPATH="$mock_site_packages" python3 "$repo_root/scripts/evaluate_mlx.py" \
  --model "$mock_model" \
  --adapter "$mock_adapter" \
  --data "$dataset_missing_field" \
  --output-dir "$test_sandbox/missing_out" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] evaluate_mlx.py should reject empty fix_strategy"; exit 1; }
[[ "$out" == *"missing non-empty string field"* ]] || { echo "[FAIL] unexpected missing field error: $out"; exit 1; }
echo "[PASS] Dataset schema validation (missing/empty required field) verified."

# 12. Test: Missing adapter fails explicitly without --baseline-only
set +e
out=$(PYTHONPATH="$mock_site_packages" python3 "$repo_root/scripts/evaluate_mlx.py" \
  --model "$mock_model" \
  --adapter "$test_sandbox/nonexistent_adapter_dir" \
  --output-dir "$test_sandbox/no_adapter_out" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] evaluate_mlx.py should reject nonexistent adapter"; exit 1; }
[[ "$out" == *"adapter directory not found"* ]] || { echo "[FAIL] unexpected adapter error: $out"; exit 1; }
echo "[PASS] Missing adapter explicit rejection verified."

# 13. Test: Baseline-only mode runs without adapter
PYTHONPATH="$mock_site_packages" python3 "$repo_root/scripts/evaluate_mlx.py" \
  --model "$mock_model" \
  --baseline-only \
  --output-dir "$test_sandbox/baseline_only_out" \
  --report-name "base-only.json" >/dev/null

python3 -c "
import json
with open('$test_sandbox/baseline_only_out/base-only.json') as f:
    r = json.load(f)
assert r['fine_tuned'] is None
assert r['parameters']['baseline_only'] is True
print('[PASS] Baseline-only evaluation verified.')
"

# 14. Test: Report name traversal guards
set +e
out=$(PYTHONPATH="$mock_site_packages" python3 "$repo_root/scripts/evaluate_mlx.py" \
  --model "$mock_model" \
  --baseline-only \
  --report-name "../escaped.json" \
  --output-dir "$test_sandbox/traversal_out" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] evaluate_mlx.py should reject path traversal in --report-name"; exit 1; }
[[ "$out" == *"must be a simple filename without path traversal"* ]] || { echo "[FAIL] unexpected traversal error: $out"; exit 1; }
echo "[PASS] Report name path traversal protection verified."

# 15. Test: Consecutive evaluations preserve atomic run directories
PYTHONPATH="$mock_site_packages" python3 "$repo_root/scripts/evaluate_mlx.py" \
  --model "$mock_model" \
  --baseline-only \
  --output-dir "$test_sandbox/run_isolation_out" >/dev/null
PYTHONPATH="$mock_site_packages" python3 "$repo_root/scripts/evaluate_mlx.py" \
  --model "$mock_model" \
  --baseline-only \
  --output-dir "$test_sandbox/run_isolation_out" >/dev/null

run_count=$(find "$test_sandbox/run_isolation_out" -maxdepth 1 -type d -name "eval-run-*" | wc -l | tr -d ' ')
[[ "$run_count" -ge 2 ]] || { echo "[FAIL] Expected at least 2 atomic run directories, found $run_count"; exit 1; }
echo "[PASS] Atomic run directory isolation preserved across consecutive executions."

# 16. Test: End-to-end Mock Train -> Run Manifest -> Evaluation with Unique Run ID
unique_pipeline_run_id="pipeline_run_${$}_${RANDOM}"

MLX_LM_LORA_BIN="$mock_lora_exec" \
MLX_MODEL_DIR="$mock_model" \
MLX_RUN_ID="$unique_pipeline_run_id" \
bash "$repo_root/scripts/train_mlx_lora.sh" >/dev/null

# Only track after confirmed creation
if [[ -d "/tmp/model-lab-runs/$unique_pipeline_run_id" ]]; then
  CREATED_RUN_DIRS+=("/tmp/model-lab-runs/$unique_pipeline_run_id")
fi

manifest_path="/tmp/model-lab-runs/$unique_pipeline_run_id/run_manifest.json"
[[ -f "$manifest_path" ]] || { echo "[FAIL] run_manifest.json was not generated!"; exit 1; }

# Now evaluate using the generated manifest
PYTHONPATH="$mock_site_packages" python3 "$repo_root/scripts/evaluate_mlx.py" \
  --manifest "$manifest_path" \
  --output-dir "$test_sandbox/manifest_eval_out" >/dev/null

python3 -c "
import json
with open('$test_sandbox/manifest_eval_out/mlx-evaluation-report.json', 'r') as f:
    data = json.load(f)
assert data['adapter']['metadata']['rank'] == 8
print('[PASS] End-to-end mock train -> run manifest -> evaluation verified.')
"

# Clean up the specific pipeline run directory created for this test
rm -rf "/tmp/model-lab-runs/$unique_pipeline_run_id"

echo "=== All 21 MLX safety, regression, isolation, concurrency, and evaluator tests passed successfully ==="
