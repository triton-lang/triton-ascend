#!/usr/bin/env bash
set -euo pipefail

test_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$test_dir" rev-parse --show-toplevel)
lock_file="$test_dir/fa_repro_lock.json"
python=${PYTHON:-python3}
output=${1:-"$PWD/fa-cvsplit-reproduction"}
device=${ASCEND_RT_VISIBLE_DEVICES:-6}
variants=(baseline dcvp cvsplit auto default)
msprof=${MSPROF:-$(command -v msprof || true)}

export ASCEND_RT_VISIBLE_DEVICES="$device"
export TRITON_ASCEND_SOC_VERSION=Ascend950PR_9589
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TRITON_ALWAYS_COMPILE=1

if [[ -z "$msprof" || ! -x "$msprof" ]]; then
  echo "msprof is required; source CANN or set MSPROF=/path/to/msprof" >&2
  exit 1
fi
command -v bishengir-compile >/dev/null || {
  echo "bishengir-compile is required on PATH" >&2
  exit 1
}

"$python" - "$repo_root" "$lock_file" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
lock = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True).strip()

required = lock["compiler_implementation_commit"]
if subprocess.call(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", required,
         "HEAD"]) != 0:
    raise SystemExit(f"checkout does not contain compiler commit {required}")

submodule = root / "third_party/ascend/AscendNPU-IR"
actual_submodule = subprocess.check_output(
    ["git", "-C", str(submodule), "rev-parse", "HEAD"], text=True).strip()
expected_submodule = lock["ascend_npu_ir_submodule_commit"]
if actual_submodule != expected_submodule:
    raise SystemExit(
        f"AscendNPU-IR is {actual_submodule}; expected {expected_submodule}")

compiler = Path(subprocess.check_output(
    ["bash", "-lc", "command -v bishengir-compile"], text=True).strip())
digest = hashlib.sha256(compiler.read_bytes()).hexdigest()
expected_digest = lock["bishengir_compile"]["sha256"]
if digest != expected_digest:
    raise SystemExit(
        f"bishengir-compile SHA256 is {digest}; expected {expected_digest}")

print(f"triton_checkout={git('rev-parse', 'HEAD')}")
print(f"ascend_npu_ir={actual_submodule}")
print(f"bishengir_compile={compiler}")
print(f"bishengir_compile_sha256={digest}")
PY

mkdir -p "$output"
cp "$lock_file" "$output/fa_repro_lock.json"
git -C "$repo_root" rev-parse HEAD >"$output/triton_commit.txt"
git -C "$repo_root/third_party/ascend/AscendNPU-IR" rev-parse HEAD \
  >"$output/ascend_npu_ir_commit.txt"
bishengir-compile --version >"$output/bishengir_compile_version.txt" 2>&1
sha256sum "$(command -v bishengir-compile)" \
  >"$output/bishengir_compile_sha256.txt"

common=(
  --sequence-length 1024 --head-dim 64 --block-m 128 --block-n 128
  --core-num 28 --active-blocks 0 --unroll-factor 4)

for variant in "${variants[@]}"; do
  "$python" "$test_dir/profile_fa.py" --variant "$variant" \
    --batch-size 1 --num-heads 1 --warmup 1 --iterations 1 \
    "${common[@]}" | tee "$output/accuracy_${variant}.log"
done

results=()
for variant in "${variants[@]}"; do
  profile="$output/msprof_${variant}"
  application=$(printf '%q ' "$python" "$test_dir/profile_fa.py" \
    --variant "$variant" --batch-size 128 --num-heads 8 \
    --warmup 3 --iterations 10 --skip-accuracy "${common[@]}")
  "$msprof" --output="$profile" --task-time=on --ai-core=on \
    --aic-mode=task-based --application="$application"
  summary="$output/summary_${variant}.json"
  "$python" "$test_dir/summarize_msprof.py" "$profile" \
    --warmup 3 --output "$summary"
  results+=("$variant=$summary")
done

"$python" "$test_dir/compare_fa_results.py" \
  --reference baseline --csv "$output/comparison.csv" \
  --json "$output/comparison.json" "${results[@]}" \
  | tee "$output/comparison.txt"

echo "Reproduction artifacts: $output"
