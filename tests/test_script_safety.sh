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

# Create isolated test environment
test_sandbox="$(mktemp -d /tmp/mlx-suite-sandbox-XXXXXX)"
trap 'rm -rf "$test_sandbox"' EXIT

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
out=$(MLX_MODEL_DIR="$mock_model" MLX_DATA_DIR="$shared_dir" MLX_ADAPTER_DIR="$shared_dir" bash "$repo_root/scripts/train_mlx_lora.sh" 2>&1)
code=$?
set -e
[[ $code -ne 0 ]] || { echo "[FAIL] train_mlx_lora.sh should reject identical data and adapter dirs"; exit 1; }
[[ "$out" == *"cannot be the same directory"* ]] || { echo "[FAIL] unexpected collision error: $out"; exit 1; }
[[ -f "$shared_dir/sentinel.txt" ]] || { echo "[FAIL] Sentinel file in colliding dir was destroyed!"; exit 1; }

# Nested path test
nested_sub="$shared_dir/nested_adapter"
set +e
out=$(MLX_MODEL_DIR="$mock_model" MLX_DATA_DIR="$shared_dir" MLX_ADAPTER_DIR="$nested_sub" bash "$repo_root/scripts/train_mlx_lora.sh" 2>&1)
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
echo "[PASS] Pre-validation mutation-free guarantee verified."

# 6. Test: Evaluator full successful pipeline with mock MLX backend
eval_out_dir="$test_sandbox/eval_output"
mkdir -p "$eval_out_dir"
PYTHONPATH="$mock_site_packages" python3 "$repo_root/scripts/evaluate_mlx.py" \
  --model "$mock_model" \
  --adapter "$mock_adapter" \
  --output-dir "$eval_out_dir" \
  --report-name "test-report.json" > "$test_sandbox/eval_stdout.json"

[[ -f "$eval_out_dir/test-report.json" ]] || { echo "[FAIL] test-report.json not produced!"; exit 1; }

# Verify schema and metadata in output report
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
print('[PASS] Mock MLX evaluation report metadata and schema successfully verified.')
"

# 7. Test: Unknown sample ID handling in evaluator
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

# 8. Test: Missing required field in dataset record fails explicitly
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

# 9. Test: Missing adapter fails explicitly without --baseline-only
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

# 10. Test: Baseline-only mode runs without adapter
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

# 11. Test: Report name traversal and symlink guards
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

# 12. Test: Consecutive evaluations preserve atomic run directories
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

# 13. Test: End-to-end Mock Train -> Run Manifest -> Evaluation
mock_bin_dir="$test_sandbox/mock_bins"
mkdir -p "$mock_bin_dir"
mock_lora_exec="$mock_bin_dir/mock_mlx_lm_lora"
cat <<'SH' > "$mock_lora_exec"
#!/usr/bin/env bash
# Mock MLX LoRA binary that creates mock adapter outputs
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

mock_train_run_id="mock_test_run_123"
MLX_LM_LORA_BIN="$mock_lora_exec" \
MLX_MODEL_DIR="$mock_model" \
MLX_RUN_ID="$mock_train_run_id" \
bash "$repo_root/scripts/train_mlx_lora.sh" >/dev/null

manifest_path="/tmp/model-lab-runs/$mock_train_run_id/run_manifest.json"
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

# Cleanup mock run
rm -rf "/tmp/model-lab-runs/$mock_train_run_id"

echo "=== All 13 MLX safety, regression, and evaluator tests passed successfully ==="
