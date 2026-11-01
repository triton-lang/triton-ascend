# Final performance: previous TLA, final TLA and AscendC

Captured 2026-11-01. **Final TLA beats AscendC in 1/8 cases: FA7 by 4.3%.**
The readability refactor launches byte-identical binaries in all eight cases;
it therefore preserves this measured device implementation. The direct frontend
PR has not yet been wheel-built; these numbers apply to the tested compatibility
path and runtime described below, not an untested rebuilt wheel.

## Complete paired comparison

All latencies are microseconds, median of three round medians, 40 kernel samples
per round. The three implementations were measured in the same run. Lower is
better. “TLA v4” means the previous matched-output optimized TLA implementation,
not the original GitCode attachment or a published result from another machine.

| Case | S | D | Causal | Previous TLA v4 | Final TLA v5 | AscendC | Final/native | Gain vs v4 |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|
| FA1 | 8192 | 128 | Yes | 2111.497 | 2111.478 | 1808.023 | 1.168 | 0.00% |
| FA2 | 8192 | 128 | No | 4084.904 | 4088.958 | 3536.142 | 1.156 | -0.10% |
| FA3 | 8192 | 64 | Yes | 1430.079 | 1348.006 | 1312.832 | 1.027 | 5.74% |
| FA4 | 8192 | 64 | No | 2755.961 | 2637.527 | 2371.970 | 1.112 | 4.30% |
| FA5 | 1024 | 128 | Yes | 49.245 | 44.748 | 39.538 | 1.132 | 9.13% |
| FA6 | 1024 | 128 | No | 72.688 | 72.671 | 59.439 | 1.223 | 0.02% |
| FA7 | 1024 | 64 | Yes | 36.365 | 32.233 | 33.669 | 0.957 | 11.37% |
| FA8 | 1024 | 64 | No | 49.182 | 46.106 | 44.231 | 1.042 | 6.25% |

FA1, FA2 and FA6 use the same kernel before and after selection; their tiny
differences are measurement variation. Do not count them as algorithmic gains.
The five changed cases improve by approximately 4.3–11.4% over v4.

| Case | Final TLA round-median range (us) | AscendC round-median range (us) | Final wins every round |
|---|---:|---:|:---:|
| FA1 | 2111.449–2112.204 | 1802.613–1808.840 | No |
| FA2 | 4086.149–4103.505 | 3501.314–3550.848 | No |
| FA3 | 1347.976–1348.009 | 1309.616–1313.718 | No |
| FA4 | 2636.719–2641.103 | 2371.834–2372.323 | No |
| FA5 | 44.739–44.797 | 39.451–39.587 | No |
| FA6 | 72.605–72.796 | 59.408–59.515 | No |
| FA7 | 32.206–32.255 | 33.658–33.837 | Yes |
| FA8 | 46.067–46.158 | 44.117–44.564 | No |

FA7 wins all three paired rounds, with separated round-median ranges. This is a
shared-device observation; there is no isolated-device or statistical confidence
claim. Other users' jobs continued. Every sample is retained without idle-window
filtering or outlier deletion.

## Environment and fairness

| Item | Configuration |
|---|---|
| Device | Ascend950PR_9579; 28 AIC / 56 AIV |
| Runtime | CANN 9.1.0, same environment for both implementations |
| Python | 3.11 |
| triton-ascend | 3.6.0.dev20261031+git8ca956c1 |
| PyTorch / torch_npu | 2.7.1+cpu / 2.7.1.post7.dev20260707 |
| Q/K/V | B4 H8 BNSD, contiguous BF16; KV heads = 8 |
| Outputs | BF16 O; FP32 max and sum, each [B,H,S,8] |
| Math | BF16 matrix operands including P; FP32 accumulation and softmax/state |
| Native API | npu_fusion_attention; inner_precise=0; scale=D**-0.5; keep_prob=1 |
| Mask | Dense sparse_mode=0; causal sparse_mode=2, compressed 2048² Boolean mask |
| Other features | No PSE, padding, dropout, RoPE or variable-length sequences |
| TLA tiling | BM=128, BN=256; per-case buffer/scheduling choices |
| Fixed launch options | sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False |

Internal tiling is independently optimized. Workload, numerical policy, outputs
and cores match. No compiler-option sweep, lower-precision replacement, output
omission, input-value dispatch or hidden preprocessing is used. The native API's
actual specialization and launch metadata were inspected; its default name or
precision flag alone was not treated as proof.

## Native dispatch for each case

The kernel identities and decoded keys are retained in
[evidence/native_dispatch.json](evidence/native_dispatch.json). Layout/buffer
columns below are source deductions from the dispatched template arguments and
`IsDn`, `useDn`, `Bmm2ResBuffSel`, not instruction traces for all eight cases.

| Case | Native S1 | Native S2 | optionalDn | Score layout | PV UB slots |
|---|---:|---:|---:|---|---:|
| FA1 | 128 | 128 | 1 | DN | 1 |
| FA2 | 128 | 128 | 0 | DN | 2 |
| FA3 | 128 | 128 | 1 | DN | 1 |
| FA4 | 128 | 256 | 0 | DN | 2 |
| FA5 | 128 | 128 | 0 | ND | 2 |
| FA6 | 128 | 128 | 0 | DN | 2 |
| FA7 | 128 | 128 | 0 | ND | 2 |
| FA8 | 128 | 256 | 0 | DN | 2 |

All keys select ImplMode=0, Regbase=1 and no dropout/RoPE/PSE. The public API
layout is BNSD; the internal layout enum is not interpreted using a stale comment.

## Precision validation

Both final TLA and AscendC pass **152 full-output checks each** over all eight
cases. Normal seed 20261030 is launched ten times; normal seed 20261031, uniform
scores (Q=0, seed 20261101) and peaked scores (Q and K scaled by 4, seed 20261102)
are each launched three times. Repeats within a family are not independent seeds.
All preallocated TLA outputs are poisoned with NaNs before each checked launch.

The blocked reference uses FP32 matmul with HF32 disabled. Every output element
and all eight max/sum replicas are checked with
`max(abs(actual-reference)) / max(max(abs(reference)), 1e-30)`.
Limits: O ≤ 0.005, max/sum ≤ 0.0001. These are benchmark criteria, not an official
API error-bound guarantee or elementwise relative-error thresholds.

| Implementation | Worst O error | Worst max error | Worst sum error |
|---|---:|---:|---:|
| Final TLA | 0.0033316448 | 3.1996680e-7 | 1.2095586e-5 |
| AscendC | 0.0033316448 | 3.1996680e-7 | 1.2172141e-5 |

The reorganized implementation was separately rerun through the public wrapper:
all 152 checks per side pass and all eight NPU binary SHA-256 values match the
frozen measured artifacts. See [readable_validation.json](evidence/readable_validation.json),
[AST equivalence](evidence/ast_equivalence.json) and
[measured binary identities](evidence/measured_binaries.json).

## Timing evidence and reproduction

[evidence/performance_samples.json](evidence/performance_samples.json) contains
all 2,880 samples from 72 profiling windows, round medians, core metadata and
source hashes. It is a privacy-cleaned extract of the original retained run;
no timing samples were removed. Original full profiler traces remain in the
working investigation, outside this focused review package. The CSV table is
[evidence/performance.csv](evidence/performance.csv).

```bash
conda activate lpc
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=1
export TRITON_CACHE_DIR="$PWD/cache"
python verify.py
python run.py validate --output my_correctness.json
python run.py benchmark --rounds 3 --samples 40 --output my_comparison.json
```

Benchmarking checks correctness before profiling, rotates model order, retains
all samples, checks actual 28/56 core counts and compares trace durations with
op-summary CSV. Compilation, input generation, output allocation and native mask
creation happen before timed windows. Compare kernel-only duration; do not mix
host/event timings or partial simulator cycles into this table.

The baseline is runnable through `prepare_call(..., baseline=True)`: it reuses
exactly the two existing single-PV templates, so the package needs no duplicate
historical kernel files. `--frontend native` selects the new direct APIs and
requires a wheel built with the PR; the measured wheel uses `compat` by default.


The standalone profiler was smoke-tested on FA7 with all three models, one round
and four samples each; all 12 launches passed the core-count and CSV/trace checks.
[evidence/runner_smoke.json](evidence/runner_smoke.json) validates the runner only
and is deliberately not substituted for the 120-sample-per-model final table.
