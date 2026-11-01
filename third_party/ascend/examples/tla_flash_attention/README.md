# TLA FlashAttention: focused review package

The package contains the final eight-case kernel, the prior matched-output TLA
baseline, a fair AscendC runner, two self-contained reports and compact evidence.
It excludes rejected experiments, compiler caches, binary dumps and large raw
profiler/simulator directories.

- **[PERFORMANCE.md](PERFORMANCE.md):** complete TLA v4 / final TLA / AscendC table,
  runtime, precision contract, all eight native dispatches and reproduction.
- **[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md):** reusable source/IR/trace workflow,
  layout, packing, pipeline, tail and scheduling optimizations; rejected paths
  and concrete further directions.
- **[kernel.py](kernel.py):** named QK, online-softmax, PV and output stages plus
  six workload-selected schedules.
- **[flash_attention.py](flash_attention.py):** public API and shape-only dispatch.
- **[run.py](run.py):** standalone correctness and paired kernel-only benchmark.
- **[expand_kernel.py](expand_kernel.py):** transparent six-phase source expansion.
- **[verify.py](verify.py):** host-only source/evidence checks.

## Run

Use an Ascend950PR with 28 Cube cores. The validated environment is Python3.11,
triton-ascend 3.6.0.dev20261031+git8ca956c1, torch2.7.1+cpu,
torch_npu2.7.1.post7.dev20260707 and CANN9.1.0. No compiler wheel is bundled.
From this directory:

```bash
conda activate lpc
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=1  # Change to your assigned physical device.
export TRITON_CACHE_DIR="$PWD/cache"
python verify.py
python run.py validate --smoke --output smoke.json
python run.py validate --output precision.json
python run.py benchmark --output comparison.json
```

Use fresh output names; the runner refuses to overwrite evidence. Benchmarking
includes the replayable previous TLA baseline, final TLA and native AscendC.
Other users' jobs need not be stopped; timings are labeled shared-device results.

```python
from flash_attention import flash_attention, prepare_call

# q/k/v: contiguous BF16 [4, 8, S, D] on NPU.
output, row_max, row_sum = flash_attention(q, k, v, causal=True)

# Reuse this closure to exclude Python setup/output allocation from profiling.
call = prepare_call(q, k, v, causal=True)
output, row_max, row_sum = call()
```

## Source organization and preservation of performance

`kernel.py` is a **source template**, not a directly imported TLA module. Its
`inline_*` stages are expanded syntactically before JIT; the six pipeline phases
rotate statically chosen buffers and matching flags. The result is ordinary TLA
Python with the exact measured expression tree. To inspect it without an NPU:

```bash
python expand_kernel.py --output generated
```

All six variant ASTs match the original measured sources. All eight device
binaries are byte-identical, and the reorganized wrapper passed 152 full-output
checks per implementation. Generated files are therefore unnecessary in the ZIP.
The six templates share the arithmetic and named stages while keeping memory
allocations and each workload's mapping visible for review.

## Frontend change status

The PR adds direct FP32 `simd.vexpdif` / `simd.vmuls`, transpose-A and bounded real
MMAD extents, and a safe contiguous `prefix_view`. `native_ops.py` aliases those
public interfaces. Pass `--frontend native` after building the PR's wheel.

The measured wheel predates those bindings. Its default `compat` path uses
`tla_native_ext.py` and `tla_native_tail_ext.py` to emit the existing typed
AVE/HIVM operations; ordinary inlining/canonicalization removes the helper
structure. It does not patch binaries, reduce precision or edit site-packages.
There is no silent fallback from `native` to `compat`.

**Validation boundary:** 54 source-loaded frontend contract tests pass, but the
new direct C++ bindings have not been built into a wheel. The PR is submitted
for review with that build/device check outstanding. The measured and newly
revalidated compatibility path is fully runnable in the stated environment.

## Provenance and limitations

This work continues [AscendNPU-IR issue 104](https://gitcode.com/liupengcheng2012/AscendNPU-IR/issues/104).
The measured final kernel wins FA7 by 4.3%; it does not beat AscendC in the other
seven cases. Precision/workload/core settings match while internal tiling differs.
The guide distinguishes native source deductions, partial simulator evidence and
actual paired device timing.

The ZIP's `frontend.patch` contains the upstream interface/test changes. The
example directory is included directly in the PR. No remote credentials, user
home paths or other users' process information are included.
