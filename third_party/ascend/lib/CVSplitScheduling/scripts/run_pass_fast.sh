#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
OPT="${TRITON_OPT:-$REPO/python/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt}"
FC="${FILECHECK:-${LLVM_SYSPATH:-/usr}/bin/FileCheck}"
if [[ ! -x "$FC" ]]; then
  FC="$(command -v FileCheck || true)"
fi
LITTEST="$REPO/third_party/ascend/unittest/Conversion/General/CVSplitScheduling/cv_split_scheduling_fa.mlir"

INPUT="${1:-$LITTEST}"
UNROLL="${2:-4}"
OUT="${CVSPLIT_OUT:-/tmp/cvsplit_after_opt.mlir}"
ERR="${CVSPLIT_ERR:-/tmp/cvsplit_opt_stderr.log}"

echo ">>> triton-opt: cv_split_scheduling unroll-factor=$UNROLL on $INPUT"
"$OPT" "$INPUT" "--cv_split_scheduling=compile-on-910-95=true unroll-factor=$UNROLL" \
    > "$OUT" 2> "$ERR" || { echo "PASS CRASHED. Tail of stderr:"; tail -20 "$ERR"; exit 1; }

echo ">>> output: $OUT ($(wc -l < "$OUT") lines)"
echo ">>> structural counts:"
for p in 'scope.scope' 'tcore_type<CUBE>' 'tcore_type<VECTOR>' \
         'hivm.hir.fixpipe' 'dual_dst_mode' 'get_sub_block_idx' \
         'hivm.hir.copy' 'sync_block_set' 'sync_block_wait'; do
    printf '    %-22s %s\n' "$p" "$(grep -c -- "$p" "$OUT" || echo 0)"
done

echo ">>> re-parse (verifier) check:"
if "$OPT" "$OUT" -o /dev/null 2>/dev/null; then echo "    verifier OK"; else echo "    VERIFIER FAILED"; exit 1; fi

if [[ "$INPUT" == "$LITTEST" ]] && [[ -n "$FC" ]] && [[ -x "$FC" ]]; then
  echo ">>> FileCheck (lit test):"
  if "$OPT" "$LITTEST" "--cv_split_scheduling=compile-on-910-95=true unroll-factor=$UNROLL" 2>/dev/null | "$FC" "$LITTEST"; then
    echo "    LIT-CHECK: PASS"
  else
    echo "    LIT-CHECK: FAIL"; exit 1
  fi
fi
