#!/usr/bin/env bash
set -eo pipefail
# Run one kernel per msopprof process so each case is cold and standalone.
cd "$(dirname "$0")"
source ~/env_ascend.sh
ulimit -n 1048576

run_one () {
  local kernel="$1" tag="$2"
  echo "=== CAModel $tag: $kernel ==="
  msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
    --launch-count=1 --timeout=5 \
    ./load_scalar_o4_runner.sh load_scalar_o4.o "$kernel" 2>&1 | tee "camodel_${tag}.log"
}

run_one simd_main_ld_same_o4        simd_same
run_one simd_main_ld_diff_o4        simd_diff
run_one simt_ld_uniform_same_o4     simt_same
run_one simt_ld_uniform_diff_o4     simt_diff
