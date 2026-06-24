# CVSplit Scheduling POC

Triton-Ascend pass that splits Flash Attention into interleaved CUBE/VECTOR scopes with K-unroll and ROW_SPLIT vector scheduling.

## Build

From `python/` in the triton-ascend repo root:

```bash
LLVM_SYSPATH=/path/to/llvm \
TRITON_BUILD_WITH_CCACHE=true TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME=triton-ascend \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
python3 setup.py bdist_wheel
pip install --force-reinstall python/dist/triton_ascend-*.whl
```

Rebuild only `triton-opt` after C++ edits (~20s):

```bash
cd python/build/cmake.linux-x86_64-cpython-3.12 && ninja bin/triton-opt
```

## Fast pass-only check (<1s)

```bash
./scripts/run_pass_fast.sh
./scripts/run_pass_fast.sh path/to/input.mlir 4
```

Runs `triton-opt --cv_split_scheduling=...` and optional FileCheck against the lit test.

## Simulator E2E (Flash Attention)

Requires docker-sim layout (`docker-sim/run.sh`, host `bishengir-compile`, CANN 9.1.0).

Copy `test/test_fa_accuracy.py` into your docker-sim workspace (or run from a mount that includes this repo).

### Verified working config (accuracy + timing)

```bash
cd docker-sim
sudo env CANN_VER=9.1.0 DOCKER_IT=0 \
  FA_BLOCK_M=32 FA_BLOCK_N=32 CV_SPLIT_UNROLL=4 \
  FA_VARIANT=cvsplit TRITON_FORCE_DCVP=1 USE_HOST_BISHENGIR=1 \
  ./run.sh msprof op simulator python3 /workspace/docker-sim/test_fa_accuracy.py
```

- B=1, H=1, N_CTX=8192, D=64, 1 active query tile (1 AI core)
- Result: **~134 µs**, accuracy PASS vs PyTorch reference

### Baseline comparison (no CVSplit)

```bash
FA_VARIANT=baseline TRITON_FORCE_DCVP=0 ...  # same other env vars
```

### WIP config (compiles, runtime FB bug open)

```bash
FA_BLOCK_M=128 FA_BLOCK_N=128 CV_SPLIT_UNROLL=4 \
TRITON_CVSPLIT_ROWLOOP=1 TRITON_CVSPLIT_HOIST_STATE=1 \
FA_VARIANT=cvsplit TRITON_FORCE_DCVP=1 ...
```

## Env gates (pass)

| Variable | Effect |
|----------|--------|
| `TRITON_FORCE_DCVP=1` | Enable CVSplit pass (via compiler options) |
| `CV_SPLIT_UNROLL` | K-loop unroll factor (default 4) |
| `TRITON_CVSPLIT_ROWLOOP` | Per-row vector loops inside VECTOR scopes |
| `TRITON_CVSPLIT_HOIST_STATE` | Pin softmax state to `mem_unique` UB buffers |
| `TRITON_CVSPLIT_SIMD` | SIMD scope grouping (experimental) |
| `TRITON_CVSPLIT_NO_MULTIBUFFER=1` | Disable auto multibuffer when rowloop on |

## Layout

- `CVSplitScheduling.cpp` — pass implementation
- `reference/target_optimized.ir` — manual golden kernel IR
- `test/test_fa_accuracy.py` — FA accuracy driver
- `scripts/run_pass_fast.sh` — triton-opt harness
- `../../unittest/Conversion/General/CVSplitScheduling/cv_split_scheduling_fa.mlir` — lit test input
