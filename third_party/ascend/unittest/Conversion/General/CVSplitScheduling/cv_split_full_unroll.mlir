// RUN: sed 's/arith.constant 8192 : i32/arith.constant 128 : i32/' %S/cv_split_scheduling_fa.mlir | triton-opt "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=COLLAPSE
// RUN: sed 's/arith.constant 8192 : i32/arith.constant 128 : i32/' %S/cv_split_scheduling_fa.mlir | triton-opt "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=LANES
// RUN: sed 's/arith.constant 8192 : i32/arith.constant 128 : i32/' %S/cv_split_scheduling_fa.mlir | triton-opt --debug-only=cv-split-scheduling "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG
// RUN: sed 's/arith.constant 8192 : i32/arith.constant 128 : i32/' %S/cv_split_scheduling_fa.mlir | triton-opt "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4 promote-fully-unrolled=false" 2>/dev/null | FileCheck %s --check-prefix=KEEPLOOP
//
// The FA candidate's inner loop steps by 32, so an upper bound of 128 gives a
// trip count of exactly 4 -- equal to the unroll factor.  `loopUnrollByFactor`
// ends with `promoteIfSingleIteration`, which in that case erases the loop and
// splices its body into the parent block while still reporting success.  Every
// stage after the unroll is written against a live `scf.for` body, so the pass
// used to continue against an erased loop and abort inside `getBody()`.
//
// Such a candidate must unroll in place behind a single-iteration loop and then
// transform exactly as a non-collapsing candidate of the same factor does.

// DIAG: [cv-split] Pre-check accepted candidate loop
// DIAG: [cv-split] Unroll consumes the whole trip count; keeping a single-iteration loop for the remaining stages
// DIAG: [cv-split] Unrolled by 4
// DIAG: [cv-split] Stage 9 complete

// The transformation commits rather than rolling back.
// COLLAPSE: triton_ascend.cv_split_scheduling.applied = 1 : i32
// COLLAPSE-LABEL: func.func @_attn_fwd

// CUBE scope holds all four unrolled lanes -- four QK and four PV matmuls.
// By default the scaffold loop is promoted away, leaving only the outer
// physical-block loop in the function.
// COLLAPSE: scope.scope
// COLLAPSE-COUNT-8: linalg.matmul
// COLLAPSE: hivm.tcore_type = #hivm.tcore_type<CUBE>

// VECTOR scope follows.
// COLLAPSE: scope.scope
// COLLAPSE: hivm.tcore_type = #hivm.tcore_type<VECTOR>

// Neither scope retains a loop of its own.
// COLLAPSE-NOT: scf.for

// Lane zero keeps the loop's own induction variable; the remaining lanes are
// folded to lower_bound + lane * step.
// LANES-DAG: arith.constant 32 : i32
// LANES-DAG: arith.constant 64 : i32
// LANES-DAG: arith.constant 96 : i32

// Cross-scope synchronization is allocated exactly as for a non-collapsing
// unroll by 4: three transfers per lane, flags 0 through 11.
// LANES-DAG: flag = 0
// LANES-DAG: flag = 11

// With promotion disabled the scaffold survives in both scopes, as one loop
// whose step covers its whole range -- a single iteration carrying all lanes.
// KEEPLOOP: scope.scope
// KEEPLOOP: scf.for
// KEEPLOOP: hivm.tcore_type = #hivm.tcore_type<CUBE>
// KEEPLOOP: scope.scope
// KEEPLOOP: scf.for
// KEEPLOOP: hivm.tcore_type = #hivm.tcore_type<VECTOR>
