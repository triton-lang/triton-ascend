# Arithmetic CCE auxiliary tests

One shared CCE kernel and host cover FP32 ADD/DIV/EXP/LOG and i32 signed/unsigned
remainder. The runner selects one operation and one route; there are no parameter
scans, unrelated operation branches, minimum-sample selection, or trace modes.
Only source files are committed. Timing data and independent Triton validation
belong to the PR evidence.

FP32 uses eight independent chains, SIMD vector64 or SIMT 32 requested warps.
Each chain performs one target per runtime iteration, with normal compiler
optimization (no forced unroll pragma). EXP/LOG use finite 1/3/5-step recurrences;
unchanged runtime inputs reload on every VF call, preventing long overflow or
fixed-input hoisting. DIV and SIMD ADD use 200/400/600 steps; SIMT ADD retains 400/800/1200. Remainder preserves i32 signed
truncating or unsigned semantics, four chains and SIMT 16 requested warps, positive
runtime divisors 13/15/17/19, and 200/400/600 steps. Launch warps are not a measured
instantaneous active/resident count. Input/readback helpers are not memory tests.

The shared host checks all three points against CPU results, including finite
FP32 outputs and exact integer bits; it reports all seven batches, arithmetic
means and same-ELF iteration differences in device SYS_CNT ticks. Its timing
scope includes VF/loop control. A point residual is diagnostic, not proof of
confidence-bounded linearity, emitted target counts, saturation or model accuracy.

**These remainder tests are reduced-numerator auxiliary feedback.** After the
first x%=divisor, inputs are already reduced. They did not produce the final
profile's generic srem/urem rates. Those rates come from the separately validated
Triton i32 complete-lowering training/heldout evidence attached to the PR, including
required integer regeneration and lowering/state costs. Other integer widths reuse
that estimate without independent validation. Never substitute historical mixed
95.5/56 coefficients or describe this test as pure REM/full-range throughput.
The consolidated source has only been statically compiled; no new hardware
calibration or coefficients were produced by simplifying these helpers.

Dependencies: Python3 standard library, g++, CANN9.1.0 CCE/runtime/ascendcl and
bishengir/lib/Template include/lib (dav-c310). Inspect npu-smi and select an idle
physical NPU with no other process; never share or reset someone else's device.
Choose a fresh output directory. The runner never changes a profile.

```sh
python3 run.py --template-root "$TEMPLATE" --device "$DEVICE" \
  --route simt --op add --output /tmp/cce-add-new
python3 run.py --template-root "$TEMPLATE" --device "$DEVICE" \
  --route simd --op div --output /tmp/cce-div-new
python3 run.py --template-root "$TEMPLATE" --device "$DEVICE" \
  --route simt --op exp --output /tmp/cce-exp-new
python3 run.py --template-root "$TEMPLATE" --device "$DEVICE" \
  --route simt --op log --output /tmp/cce-log-new
python3 run.py --template-root "$TEMPLATE" --device "$DEVICE" \
  --route simt --op srem --output /tmp/cce-srem-aux-new
```

Both routes support all six selectors (srem/urem are two signedness variants of
MOD). Add --compile-only to build without launching. Do not blindly convert ticks
using historical 988.9MHz; verify the actual device's counter domain and frequency.
The PR retains before/after absolute-budget and independent heldout proofs;
this small source package does not claim automatic whole-kernel score accuracy.

Static check: all twelve operation/route configurations compile for both CCE and
the shared host. No physical device was launched during this check.

ADD is an auxiliary CCE throughput reference: the retained historical eight-chain,
32-requested-warp, one-target-per-chain SIMT measurement is
104.79086917450967 scalar-op/SYS_CNT tick. The add core reuses its common runtime
base/increment, literal chain offsets and fixed result sum; SIMD uses eight
vector64 vadd chains. This simplified shared source was statically compiled,
not newly timed to seek a peak. Independent Triton accuracy/heldout evidence
covers DIV/EXP/LOG/MOD; it does not establish ADD heldout accuracy. The online
profile keeps the original shared SIMT ADD reference (141); the auxiliary
104.79 result is not adopted in this change.
