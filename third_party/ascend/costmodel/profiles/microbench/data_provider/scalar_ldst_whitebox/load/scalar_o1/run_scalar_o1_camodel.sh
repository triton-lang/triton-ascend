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
    ./load_scalar_o1_runner.sh load_scalar_o1.o "$kernel" 2>&1 | tee "camodel_${tag}.log"
}

run_one simd_main_ld_o1        simd
run_one simt_ld_uniform_o1     simt32
run_one simt_ld_uniform_o1_t1  simt1
