# SIMT Cost Model Calibration Notes

Date: 2026-07-13

## Files

- Parser: `extract_simt_camodel_calibration.py`
- SIMT scope camodel JSON: `D:\backup\codex\simt_camodel_calibration_scope_b8.json`
- SIMD baseline instr summary JSON: `D:\backup\codex\simd_camodel_instr_summary_b8.json`

Remote source data:

- SIMT scope: `/data/kaixin/triton_cases/fbgemm/sim_scope_camodel_scope_b8/OPPROF_20260712042738_MECSEVSNUOCUDPXV`
- SIMD baseline: `/data/kaixin/triton_cases/fbgemm/sim_scope_camodel_simd_b8/OPPROF_20260712043444_JIMGWGXCYTWWPYJT`

## Parser Policy

For SIMT op counts, only primary issue/receive-level dumps are counted:

- `rvec.simt.exu0..3.dump`
- `rvec.simt.lsu.dump`
- `rvec.simt.dvg0..3.dump`

Register read/write logs are intentionally excluded because they duplicate a single instruction across register events and hazards.

## FBGEMM Scope SIMT Summary

Aggregate primary SIMT counts:

```text
memory      14196
float_alu    4264
shuffle      2496
control      1612
int_alu      1542
predicate     252
other         732
```

Top ops:

```text
SIMT_LDG       5928
SIMT_LDS       2964
SIMT_FMNMX_I   2548
SIMT_SHFL      2496
SIMT_FMNMX     1664
SIMT_BRANCH    1560
SIMT_STG       2652
SIMT_STS       2652
```

Important interpretation:

- Aggregate counts are across multiple active core/veccore units, while the timestamp span is wall-clock-like. Aggregate `cycles/op` is therefore a throughput seed, not a single-unit latency.
- Per-unit entries, for example `core0.veccore0`, are better for local cost weights.
- For `core0.veccore0`, primary span is about `10789` cycles, with `1092` memory ops, `328` float ALU ops, and `192` shuffle ops.

## SIMD Baseline Summary

SIMD `instr_exe` pipe cycles:

```text
SCALAR 335330
RVECEX  49144
MTE3    18823
PUSHQ   19992
RVECLD  16618
RVECST   9910
MTE2     3179
```

The SIMD baseline is dominated by `SCALAR`, which supports the cost model split:

```text
simd_total = max(vector_roofline, scalar_path)
```

For this FBGEMM case, scalar address/mask/scalarization risk should dominate the SIMD estimate.

## Cost Model Use

Initial practical mapping:

- `SIMT_LDG/STG/LDS/STS` -> SIMT memory bucket
- `SIMT_SHFL` -> shuffle/reduce bucket
- `SIMT_ISETP/PLOP3` -> predicate bucket
- `SIMT_FADD/FMUL/FMNMX` -> float ALU bucket
- `SIMT_BRANCH/END` -> control/divergence bucket

Do not treat the generated naive cycles/op as exact latency. They are calibration seeds to rank SIMD vs SIMT regions.

## SIMD/SIMT Transition Microbench

Source files:

- `D:\projects\triton_cases\SIMT_Test\transition.cce`
- `D:\projects\triton_cases\SIMT_Test\transition_host.cpp`
- `D:\projects\triton_cases\SIMT_Test\run_transition.remote.sh`

Saved result:

- `D:\backup\codex\simt_transition_microbench_tail16_barrier_20260713.txt`

Method:

- Use in-kernel `get_sys_cnt()` slope over `K=100 -> 500`.
- Measure `barrier_only`, SIMD tail only, empty SIMT `async_invoke` only, no-mid-barrier mixed orders, and mid-barrier mixed orders.
- SIMD tail is a dependent UB read/add/write chain with `tail_reps=16`.

Measured SYS_CNT frequency was about `989 MHz`, so one cycle is close to one nanosecond.

Key results:

```text
nwarp  barrier  simt_with_barrier  simt_minus_barrier
1        8.5          190.3              181.8
2        8.5          190.4              181.9
4        8.5          190.4              182.0
8        8.5          190.3              181.8
16       8.5          190.4              181.9
32       8.5          231.7              223.3
```

The mixed-order deltas did not show a positive `SIMT -> SIMD` increment. Even with an explicit mid `pipe_barrier`, `SIMT + barrier + SIMD` was smaller than the sum of standalone SIMT and standalone SIMD baselines. Therefore the current cost model treats `SIMT -> SIMD` extra as `0` unless a future benchmark finds a positive increment.

The transition table currently uses:

```text
num_warps  simd_to_simt  simd_to_simt_with_barrier  simt_to_simd
1..16          182              190                      0
32             223              232                      0
```

The `simd_to_simt` column is a complete mixed-kernel empty-SIMT setup proxy,
not an increment to add after the standalone 115-cycle empty launch.  The
shadow scorer therefore computes:

```text
mixed = (simt_total - standalone_empty_launch) + mixed_setup
```

This avoids counting SIMT launch/setup twice.  The earlier implementation used
`simt_total + mixed_setup` and overestimated the mixed candidate by 115 system
cycles.

## Isolated SIMT UB Memory Microbenchmark

Source files:

- `D:\projects\triton_cases\SIMT_Test\simt_memory.cce`
- `D:\projects\triton_cases\SIMT_Test\simt_memory_host.cpp`
- `D:\projects\triton_cases\SIMT_Test\simt_memory_david_v100_20260725.csv`

Method:

- Eight independent UB loads or stores per active SIMT thread and iteration.
- Four rotating contiguous working sets prevent loop-invariant load hoisting.
- Maximum 32-warp working set is 128 KiB.
- Use the slope over `iters=128 -> 512` with `K=20` to remove launch and timer
  overhead.

Measured saturation:

```text
operation   16 warps          32 warps
UB load     65.54 B/cycle     64.93 B/cycle
UB store    67.91 B/cycle     67.85 B/cycle
```

The isolated profile therefore uses 64.9347 B/system-cycle for UB loads and
67.8525 B/system-cycle for UB stores with medium confidence.  These values do
not replace the still-unmeasured GM LDG/STG rates.

## Isolated SIMT GM Memory Microbenchmark

Source files:

- `D:\projects\triton_cases\SIMT_Test\simt_gm_memory.cce`
- `D:\projects\triton_cases\SIMT_Test\simt_gm_memory_host.cpp`
- `D:\projects\triton_cases\SIMT_Test\simt_gm_memory_david_v100_20260725.csv`

The 32-warp run rotates over a 128 MiB sequential working set and performs
eight independent operations per thread.  The measured single-AIV saturation
rates are:

```text
GM load     22.1516 B/system-cycle
GM store    16.5114 B/system-cycle
```

These replace the workload-aggregate CaModel memory fallback.  Because SIMT
LDG and STG share the LSU, the scorer adds their separately calibrated cycle
costs.  The profile confidence is raised from low to medium for sequential GM
traffic; stride/gather latency remains future work.

## Isolated SIMT Shuffle Microbenchmark

Source files:

- `D:\projects\triton_cases\SIMT_Test\simt_shuffle.cce`
- `D:\projects\triton_cases\SIMT_Test\simt_shuffle_host.cpp`
- `D:\projects\triton_cases\SIMT_Test\simt_shuffle_david_v100_20260725.csv`

The dependent test executes one `__shfl_up` chain.  The throughput test uses
four independent chains and sweeps 1–32 warps.

```text
one-warp dependent latency       27.2706 system cycles
32-warp ILP4 throughput           0.8172 warp shuffles/system-cycle
```

This replaces the FBGEMM aggregate CaModel shuffle fallback of 0.3763 warp
instructions/system-cycle and raises shuffle confidence from low to medium.
