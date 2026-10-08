# Predicate mask validation

CCE timing checks provide limited 32-bit evidence for retaining the existing
[profile](../../../simd_simt/david_v100_simd_simt_v1.json) factor 1. Separate Triton
simulator checks validate SIMD processing widths. Neither changes profile rates.

## Reproduce

`predicate_ops{,_simt}.cce` and their `_host.cpp` files are SIMD/SIMT probes;
`predicate_check.py` builds/runs them and summarizes the 46 historical CSV rows.
From this directory, set `INC` to AscendNPU-IR's `bishengir/lib/Template/include`.
For hardware runs, inspect `npu-smi info` and explicitly select an idle physical device:

```bash
python3 predicate_check.py --summarize results_20261008.csv
python3 predicate_check.py --build-only \
  --toolkit "$ASCEND_TOOLKIT_HOME" --template-include "$INC"
python3 predicate_check.py --device "$PREDICATE_DEVICE" \
  --toolkit "$ASCEND_TOOLKIT_HOME" --template-include "$INC"
```

The runner prints `OUTPUT=...` for fresh objects, logs, hashes and results;
`--output` must name a new directory. Generated binaries/traces/logs are not committed.

## CCE timing and spill evidence

Recorded 2026-10-08 on Ascend950PR/C310, CANN 9.1.0, CCEC clang 15.0.5.
Fixed modes are compiled separately. Device SYS_CNT differences use three iteration
counts, minima of seven repeats and a midpoint linearity gate below 3%, repeated twice.
The script contains normalization and paired-difference formulas and validates all 46 rows.

| SIMD paired difference | Approx. ticks / 64-element step | Conclusion |
|---|---:|---|
| Four-chain compare/select | 0.303 | Close to existing reference `1/3.3` |
| Two-chain AND/OR/XOR residual | 0.303 | No loop predicate spill in audited traces |
| One-chain logic residual | 0.606 | Dependency-sensitive |
| Four-chain logic residual | 25.79 | Spill; excluded from coefficient evidence |

These are layout-dependent marginal costs, not isolated instruction costs; SIMT
fusion also prevents separating Boolean operations. Add controls are integer operations,
not a remeasurement of f32.add. Readback covers the first SIMD vector at 4 iterations
and SIMT a0 at 4/28 iterations, not every state or long timed run.

For a spill audit, use the built object with CANN's `Ascend950PR_9578` simulator.
In separate fresh directories containing `predicate_ops.o`, run
`predicate_ops_host 15 trace 2` and `predicate_ops_host 15 trace 4`, with
`LD_LIBRARY_PATH` set to the simulator's `camodel`, `lib` and toolkit `lib64`.
Decode `log_ca` with the SDK's `instr_calog_bin_handler.py`; count paired instances
on one core/subcore using `(count4-count2)/2`, not duplicate dispatch/completion records.
Modes 15/16/17 had VCMP/logic/VSEL increments 8/4/4 and no PLDI/PSTI increment;
mode 3 added five of each spill instruction. Reaudit spill and other stack/memory
traffic after compiler/layout changes. Simulator ticks are not hardware SYS_CNT.

## SIMD processing-width checks

Supported graphs are closed, same-shape compare/AND-OR-XOR/select groups with one
numeric select sink, argument/load inputs and store/return outputs. The maximum
processing width feeds `segments * ceil(run * width / vectorWidthBits)`;
source operation totals and SIMT costs stay unchanged.

| Type | Compare / select processing width |
|---|---|
| i8/i16/i32, f16/f32 | Source width |
| bf16 | 32 / 32 |
| f8E4M3FN / f8E5M2 | 32 / 8 |
| i64 | 32-bit parts; its own compare/select retain the previous estimate |

Simulator checks covered 22 initial and 102 dtype-extension cases at 64–512 elements.
Readbacks and scoped instruction counts passed; i64 compare/select expansion was excluded.
A separate f64 load probe failed compilation. Shared masks, loaded i1, constants/NOT,
arithmetic/cast producers, broadcasts and cross-block edges retain the previous estimate.

`predicate_tail.py` additionally checks independent `arange(0, BLOCK) < N` memory masks:
common signed-less-than bounds, contiguous addresses and zero-filled loads only.
MODE 0/1/2/3 selects unmasked/load-mask/store-mask/both; OP 0/1/2 selects AND/OR/XOR.
At BLOCK=256, 104/107 cases passed readback and scoped core/model-count checks;
three FP8 cases failed zero conversion. Three executable i64 cases failed the broader
synchronization audit, so they establish neither no-spill behavior nor timing.
Accounting uses BLOCK, not N; arbitrary masks, nonzero fills and gathers retain fallback.

Example: `probe[(1,)](a,b,c,d,x,y,x,y,out,spare,n,MODE=3,OP=0,BLOCK=256,compile_mode="simd")`.
Allocate BLOCK elements plus guards on the selected device; check CPU `where`, zero-filled
loads and unchanged masked-store sentinels. Padding does not prove out-of-bounds safety.

Width checks do not calibrate conversions, fills, scalar setup, synchronization, i64
expansion, full fusion or cross-group transfers. These results are not a universal
calibration, whole-route accuracy guarantee or cost bound.
