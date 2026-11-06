// === #2686 SelectedBlockScanRule (bit 65536) — on-chip budget gate ===
//
// The scan rewrite grows the loop body (vector table read, unconditional K/V
// tiles, scan masks), so the rewritten SANDBOX -- not the gather loop it
// replaces -- is measured against the two queries the resource model already
// exposes to rules:
//   * UB : getSafeUBBudget() vs estimatePeakLiveBytes()  (the row lift's caliber)
//   * L0 : exceedsCubeBudgets()                          (#2841 cube query)
// Zero capacities keep a query disabled, so every other RUN line in this
// directory (no budget options) is unaffected.  An over-budget form is declined
// with a warning on the live function; the pass never fails.
//
// Measured on Inputs/nsa_fwd_const_ns.mlir: the scan rewrite's own form peaks at 41,108 B,
// the joint scan + row lift form at 165,488 B.  The predicate budgets a demand
// figure, not that peak: the static estimate counts each UB tensor once while
// the backend multi-buffers loop-carried UB buffers, so both forms are scaled by
// the buffer replication factor (kUBBufferReplication = 2): the scan rewrite demands
// 82,216 B and the joint form 330,976 B.  The production budget is 209,715 B
// (physical 256 KiB x 80 %), so the scan rewrite's own form fits but the joint form does
// not; section 4 passes a larger explicit capacity to exercise the gate's
// "fits, do not warn" path, while sections 1-3 pin the declining paths.

// --- 1. UB: budget 2048*80/100 = 1638 B < the scan rewrite's 82,216 B demand -------
// The scan rewrite's own checkpoint runs before the row lift is consulted, so its overflow
// declines the rewrite and names both numbers.  The IR stays byte-identical to
// the control.
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=65536 ub-capacity-bytes=2048' -canonicalize -cse -o %t.ub 2> %t.ub.err
// RUN: FileCheck %s --check-prefix=BUDGETWARN < %t.ub.err
// RUN: diff %t.ctl %t.ub

// --- 2. L0: 1024*80/100 = 819 B per buffer, below the >=4 KiB dot tiles ----
// UB is left unspecified (disabled), so only the cube query can reject.
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=65536 mapping-l0a-capacity-bytes=1024 mapping-l0b-capacity-bytes=1024 mapping-l0c-capacity-bytes=1024' -canonicalize -cse -o %t.cube 2> %t.cube.err
// RUN: FileCheck %s --check-prefix=CUBEWARN < %t.cube.err
// RUN: diff %t.ctl %t.cube

// --- 3. joint path under the same tight UB budget -------------------------
// The own-form gate runs before the row lift is consulted, so the production path
// reports the same verdict and rewrites nothing.
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=196608 ub-capacity-bytes=2048' -canonicalize -cse -o %t.joint 2> %t.joint.err
// RUN: FileCheck %s --check-prefix=BUDGETWARN < %t.joint.err
// RUN: diff %t.ctl %t.joint

// --- 4. joint path under budgets wide enough for the row lift's demand model ----
// 1 MiB x 80 % = 838,861 B >= the joint form's 330,976 B demand, so the second
// (post-rule-B) checkpoint passes and the committed form is the lifted loop.
// RUN: echo -n > %t.empty
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=196608 ub-capacity-bytes=1048576 mapping-l0a-capacity-bytes=65536 mapping-l0b-capacity-bytes=65536 mapping-l0c-capacity-bytes=262144' -canonicalize -cse -o %t.jointfit 2> %t.jointfit.err
// RUN: diff %t.empty %t.jointfit.err
// RUN: FileCheck %s --check-prefix=JOINTFIT --implicit-check-not='scf.if' < %t.jointfit

// BUDGETWARN — the UB verdict: which form, both numbers, the decline.
// BUDGETWARN: warning: selected-block-scan: the scan rewrite needs 82216 bytes of UB, budget 1638; declining

// CUBEWARN — the cube verdict (no scalar to quote; the budgets are per-arch).
// CUBEWARN: warning: selected-block-scan: the scan rewrite exceeds the L0A/L0B/L0C cube budgets; declining

// JOINTFIT — committed joint form: 8 lifted rows, 4 carried tensors, 3 dots.
// JOINTFIT-LABEL: hacc.program_grid_transforms
// JOINTFIT-LABEL: tt.func public @parallel_nsa_fwd_kernel
// JOINTFIT: tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<8x16x!tt.ptr<i64>>
// JOINTFIT: %{{.*}}:4 = scf.for {{.*}} iter_args(
// JOINTFIT: tt.dot %{{.*}} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// JOINTFIT-COUNT-2: tt.dot %{{.*}} : tensor<128x64xbf16> * tensor<64x64xbf16> -> tensor<128x64xf32>
