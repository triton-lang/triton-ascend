#!/usr/bin/env bash
set -e
# Invoke through msopprof simulator. Keep its default model libraries, and
# resolve HAL/runtime to that same model rather than the physical driver.
TK="${ASCEND_HOME_PATH:?Source CANN set_env.sh first}"
SIMLIB="$TK/tools/simulator/dav_3510/lib"
shim_dir="$(mktemp -d /tmp/scalar-bench-lib-XXXXXX)"
trap 'rm -f -- "$shim_dir/libascend_hal.so" "$shim_dir/libruntime.so"; rmdir -- "$shim_dir"' EXIT
ln -s "$SIMLIB/libnpu_drv_camodel.so" "$shim_dir/libascend_hal.so"
ln -s "$SIMLIB/libruntime_camodel.so" "$shim_dir/libruntime.so"
export LD_LIBRARY_PATH="$shim_dir:$SIMLIB:${LD_LIBRARY_PATH:-}"
cd -- "$(dirname -- "$0")"
exec ./store_scalar_o4_host "$@"
