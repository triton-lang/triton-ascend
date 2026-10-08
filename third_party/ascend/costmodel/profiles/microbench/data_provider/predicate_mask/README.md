# Predicate mask CCE reference checks

These probes support the **32-bit reference estimates** in the
[SIMD/SIMT profile](../../../simd_simt/david_v100_simd_simt_v1.json).
These CCE timing checks do not validate mixed-width grouping or end-to-end
SIMD/SIMT route accuracy. The separate Triton geometry checks below cover a
bounded class of compare/logic/select groups. Neither changes profile rates.

Files: `predicate_ops.cce` / `predicate_ops_host.cpp` (SIMD),
`predicate_ops_simt.cce` / `predicate_ops_simt_host.cpp` (SIMT),
`predicate_check.py` (build, run, and summarize), and
`results_20261008.csv` (46 compact historical result rows).
Objects, full traces, and raw run logs are deliberately not checked in.

## Measurement and scope

Each mode is separately compiled with `-DFIXED_MODE=N`, `-O2`,
`--cce-aicore-arch=dav-c310`, and `-DREG_REGISTER_SIZE=256`. There is no runtime
mode branch in the timed body. Timing uses device `get_sys_cnt()`, not host
wall time or simulator ticks. Seven interleaved measurements at each of three
iteration counts are reduced to their minimum; the midpoint checks linearity.
The complete experiment was repeated twice on the same physical device.

| Route / modes | State layout | Per-step work |
|---|---|---|
| SIMD 0..5 | four independent i32 vectors, 64 elements each | add; add+select; add+cmp+select; add+2cmp+AND/OR/XOR+select |
| SIMD 6..11 | one vector updated four times | same six variants |
| SIMD 12..17 | two vectors updated alternately | same six variants |
| SIMT 0..4 | 1024 threads, eight ring-coupled u32 states per thread | add; add+cmp+select; add+2cmp+AND/OR/XOR+select |

SIMD sources also contain three-chain modes 18..23, but these are not part of
the checked-in October 8 result set or the default runner matrix.
SIMD uses `K=20`, iterations `200/400/600`, four steps per iteration:
`s(mode) = (c2-c1) / (400*20*4)` SYS_CNT cycles per 64-element step.
SIMT uses `K=100`, iterations `30/60/90`:
`s(mode) = (c2-c1) / (60*100*1024*8)` cycles per element update.

For each SIMD group with base mode `b=0,6,12`:

```text
select increment  = s(b+1) - s(b)
compare increment = s(b+2) - s(b+1)
logic residual    = s(b+3/4/5) - s(b+2) - compare increment
full combination  = s(b+3/4/5) - s(b)
```

The logic residual is an **effective paired difference for that layout**,
not an isolated instruction's latency: scheduling and dependencies need not
be additive. The summarizer also reports the full-combination difference,
which does not subtract an assumed second comparison cost.
SIMT reports `s(1)-s(0)` and `s(2/3/4)-s(1)`; these pairs cannot separate the
individual comparison, logic, and select costs after compiler fusion.

Runtime errors, output sentinels, CPU/device mismatches, trivial outputs, or
midpoint deviation of 3% or more reject a run. SIMD checks the first vector's
64 results at four short iterations, **not all four state vectors or every
timed iteration**. SIMT checks a0 from all 1024 threads at 4 and 28 iterations,
not all eight intermediate states. Short checks avoid converged/trivial masks;
they are not a correctness proof for every long-running state.

## Recorded results

Recorded on **2026-10-08**, Ascend950PR / C310, physical NPU 2;
`npu-smi` 25.7.rc1 reported OK, 0% utilization and no processes before launch.
CANN 9.1.0, CCEC clang 15.0.5 (`clang-5c68a1cb1231`, build
`2026-07-30T20:53:21+08:00`), host g++ 13.3.0. Compiler SHA256:
`647f5772aab1b93dd59f088721c94c3edae1212c92b6dc4117b3c5742548252a`.
All 46 recorded runs passed the host gates. Ranges below cover the two runs,
not confidence intervals. The CSV retains integer `c1/cm/c2` and gate fields
so these values can be recomputed without rounded intermediate slopes.

| SIMD paired difference | SYS_CNT cycles / 64-element step | Interpretation |
|---|---:|---|
| four-chain select | 0.302844–0.304063 | close to reference `1/3.3` |
| four-chain compare | 0.302750–0.303969 | close to reference `1/3.3` |
| two-chain AND residual | 0.303188–0.303719 | no iteration-dependent predicate spill in audited trace |
| two-chain OR residual | 0.302813–0.303219 | same scope |
| two-chain XOR residual | 0.302813–0.303531 | same scope |
| one-chain logic residuals | 0.604344–0.607625 | dependency-sensitive; not a throughput coefficient |
| four-chain logic residuals | 25.784313–25.791063 | spill-pressure cases; excluded from coefficient evidence |

| SIMT paired difference | SYS_CNT cycles / element update |
|---|---:|
| compare + select | 0.01006213–0.01006472 |
| extra compare + AND | 0.01124552–0.01124870 |
| extra compare + OR | 0.01183748–0.01184111 |
| extra compare + XOR | 0.00384141–0.00384810 |

The f32.add references are the existing profile values, `1/3.3` for SIMD
and `1/141` for SIMT; these probes do **not** remeasure f32.add. Their internal
add-only controls are i32/u32. The evidence supports retaining factor 1 as a
reference, not a universal calibration or a conservative bound. SIMT pairs
are cheaper than two reference operations in these particular cases; that
does not prove they are cheaper for every consumer or lowering pattern.

The table below preserves the source fingerprints from the recorded experiment.
The current host C++ sources have since received clang-format-only changes, and
the SIMT device comment was corrected from "independent" to "ring-coupled".
These changes do not alter the experiment logic; current source hashes may
differ. The reproduction runner records fresh source hashes in its metadata.
The historical measurements and fingerprints below are not rewritten.

| Recorded source | SHA256 |
|---|---|
| `predicate_ops.cce` | `cc41b0a5f51e5036bf7b11df8e60c27287ded3f27bd27adbe57afe6b4affd986` |
| `predicate_ops_host.cpp` | `9b9d775453731494c246fda35e4ece41ccea9acf60857369f39cae4774c67154` |
| `predicate_ops_simt.cce` before comment correction | `4b8a8fa53acb4e506c45e8089fe38c8e2b76d35bc4bbe9b355873fbc5377abad` |
| `predicate_ops_simt_host.cpp` | `047b9b1683abe57ad96ff861b8ee85fe5bc9c9d241f54e6cfcd95043f56ba940` |

## Reproduce

From this directory, first recompute the saved results without CANN or a device:

```bash
python3 predicate_check.py --summarize results_20261008.csv
```

Build only (does not initialize a device). Set `INC` to the matching
AscendNPU-IR `bishengir/lib/Template/include` directory:

```bash
python3 predicate_check.py --build-only \
  --toolkit "$ASCEND_TOOLKIT_HOME" --template-include "$INC"
```

For physical measurements, inspect `npu-smi info`, select an idle device, then
set `PREDICATE_DEVICE` to its physical index. The runner maps it to logical
device 0 and excludes simulator libraries. It runs cases sequentially and
aborts on a failed host or data gate. Do not run it on another workload's device.

```bash
python3 predicate_check.py --device "$PREDICATE_DEVICE" \
  --toolkit "$ASCEND_TOOLKIT_HOME" --template-include "$INC"
```

Both commands print `OUTPUT=...`; a fresh temporary directory contains the
objects, hosts, build logs and source/compiler/object fingerprint metadata.
Hardware runs also write `results.csv` and individual logs. `--output` accepts
an explicitly chosen **new** directory and refuses to reuse an existing one.

## Spill audit and simulator boundary

The historical simulator was CANN 9.1.0 `Ascend950PR_9578`, running C310 CCE
objects. For selected modes, compare 2-iteration and 4-iteration traces:
`(count4-count2)/2` removes fixed setup/readback work. Count paired instruction
instances in one physical core/subcore, not raw dispatch/completion records
or a sum of replicated subcores. The CANN legacy profiler reader uses
`<QIIQ200s200s` (424 bytes); `core >= 32` identifies dispatch records and
`core < 32` completion records. Pair by core, subcore, ID, PC, mnemonic and
execution unit. A different SDK format requires its matching decoder.

Audited per-iteration SIMD counts on core 0 / subcore 2:

| Mode | VCMP | VSEL | Corresponding PAND/POR/PXOR | PLDI | PSTI |
|---|---:|---:|---:|---:|---:|
| 1 / 2 | 0 / 4 | 4 | 0 | 0 | 0 |
| 12 / 13 / 14 | 0 / 0 / 4 | 0 / 4 / 4 | 0 | 0 | 0 |
| 15 / 16 / 17 | 8 | 4 | 4 | 0 | 0 |
| 3 (pressure counterexample) | 8 | 4 | 4 | 5 | 5 |

The audited historical trace objects matched the corresponding physical-run
objects byte-for-byte. The zero PLDI/PSTI iteration increments support the
limited no-spill classification above; inspect other stack/load/store traffic
as well when changing compiler or layout. Four-chain logic is not a no-spill
calibration point. SIMT modes 2/3 add ISETP and SEL relative to mode 1;
mode 4 adds ISETP without the extra SEL. No universal native AND/OR/XOR count
can be inferred from the SIMT source's Boolean operators.

To regenerate an example, set `PRED_OUT` to a build-only output directory.
Use separate directories so the two traces cannot append into each other:

```bash
PRED_SIM="$ASCEND_TOOLKIT_HOME/tools/simulator/Ascend950PR_9578"
for iterations in 2 4; do
  trace_dir="$PRED_OUT/simd_m15/trace_i${iterations}"
  mkdir "$trace_dir"
  cp "$PRED_OUT/simd_m15/predicate_ops.o" "$trace_dir/"
  (cd "$trace_dir" && \
    LD_LIBRARY_PATH="$PRED_SIM/camodel:$PRED_SIM/lib:$ASCEND_TOOLKIT_HOME/lib64" \
    "$PRED_OUT/predicate_ops_host" 15 trace "$iterations")
done
```

Inspect the resulting `log_ca` instruction data with the matching CANN
profiler decoder (`cannsim/prof/src/backend/calog_handlers/instr_calog_bin_handler.py`
in this SDK). Repeat for the other modes; SIMT uses `predicate_ops_simt.o`
and `predicate_ops_simt_host`. Changing SDK or recompiling requires a fresh
spill check. Simulator ticks are not physical SYS_CNT cycles, and this audit
does not validate the cost model's reference-lane fallback for other widths.

## SIMD predicate geometry (separate from CCE timing)

For a closed, same-shape compare/AND-OR-XOR/select graph with one numeric
select sink, SIMD uses the maximum comparison/select data width, not i1.
Inputs must be i16/i32/f16/f32 arguments or unmasked loads, with the select
ending at an unmasked store or function return. This follows
AscendNPU-IR's `HFusion/Transforms/AutoVectorize/FusedNode.cpp`,
`FusedNode::estimateTileSizeForOp`, for the tested single-group lowering.

The existing `segments * ceil(run * width / vectorWidthBits)` formula uses
`simd_predicate_bit_width` when nonzero; source `element_bit_width` and SIMT
costs are unchanged. Source operations are not removed or combined.

On 2026-10-08, full Triton AST/TTIR lowering (`Ascend950PR_9579`) and CANN
9.1.0 simulation (`Ascend950PR_9578`) checked
`where((A < B) OP (C < D), X, Y)` with six independent loaded arrays.
All 22 cases passed readback and matched projected counts: AND/OR/XOR for
the six type combinations below, plus 64/256-element i16/i32 AND cases.
Paired native counts from one core/subcore exclude memory/setup instructions;
no PLDI/PSTI appeared. Twelve saved loaded/shared-mask cases retained fallback.

| 128-element graph | Native VCMP / logic / VSEL | Previous estimate | Projected estimate |
|---|---|---|---|
| all i16 or all f16 | 2 / 1 / 1 | 2 / 2 / 1 | 2 / 1 / 1 |
| all i32 or all f32 | 4 / 2 / 2 | 4 / 2 / 2 | 4 / 2 / 2 |
| i16 and i32 comparisons, i16 select | 4 / 2 / 2 | 3 / 2 / 1 | 4 / 2 / 2 |
| two i16 comparisons, i32 select | 4 / 2 / 2 | 2 / 2 / 2 | 4 / 2 / 2 |

These validate geometry, not latency or factor 1 for all widths. Shared masks
may split groups and require transfers; loaded i1 can use data-vector logic.
These, casts, broadcasts, arithmetic producers, constants/XOR-NOT, cross-block
edges and unsupported widths/consumers retain the previous estimate, including
reference lanes for unresolved i1. The 256-node cap limits analysis work, not
hardware groups. Full fusion, folding, register pressure and cross-group
transfers remain unmodeled; this estimate is not a cost bound.
