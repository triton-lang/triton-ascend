// RUN: sed 's/inter_core_buf_count = 2/inter_core_buf_count = 1/' %S/cv_split_scheduling_fa.mlir | triton-opt "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" | FileCheck %s --check-prefix=DEPTH1
// RUN: sed 's/inter_core_buf_count = 2/inter_core_buf_count = 3/' %S/cv_split_scheduling_fa.mlir | triton-opt "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=REJECT
// RUN: sed '/^  func.func @_attn_fwd/a\    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0' %S/cv_split_scheduling_fa.mlir | triton-opt "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" | FileCheck %s --check-prefix=COLLISION
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" "--add_dynamic_cv_pipeline=compile-on-910-95=true" | FileCheck %s --check-prefix=AUTO-COMMIT
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=3" "--add_dynamic_cv_pipeline=compile-on-910-95=true" | FileCheck %s --check-prefix=AUTO-FALLBACK
//
// This contract proves that CV split consumes DynamicCVPipeline's canonical
// inter-core buffer-count attribute.  Depth one produces one physical buffer
// per transfer lineage.  Unsupported depth three is rejected transactionally
// and leaves the original unscoped loop intact.

// DEPTH1: memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
// DEPTH1-NOT: memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
// DEPTH1: memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
// DEPTH1-NOT: memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
// DEPTH1: memref.alloc() : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
// DEPTH1-NOT: memref.alloc() : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>

// One forward flag per buffer slot per phase, so depth one needs three -- and
// with a single slot per phase no lane reuses another's buffer, so there is
// nothing for the schedule to order and no reverse traffic at all.
// DEPTH1: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
// DEPTH1: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
// DEPTH1-NOT: flag = 3
// DEPTH1-NOT: sync_block{{.*}}<PIPE_V>, <PIPE_FIX>
// DEPTH1-NOT: sync_block{{.*}}<PIPE_MTE1>, <PIPE_MTE3>

// REJECT-NOT: scope.scope
// REJECT: scf.for {{.*}} step %c32_i32

// The shared FlagIdManager must see an existing flag zero and allocate the CV
// schedule above it.  This is a collision-avoidance contract, not merely a
// check that CV split happens to emit a familiar sequence on an empty module.
// Two CUBE->VECTOR phases at depth two and a VECTOR->CUBE phase with one slot
// per lane need eight IDs, so shifting past the occupied zero places them on
// 1..8.
// COLLISION: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
// COLLISION: scope.scope
// COLLISION: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
// COLLISION: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
// COLLISION-NOT: flag = 9

// When CV commits, the existing DCVP wrapper sees the result and leaves the
// two proven CV scopes untouched.
// AUTO-COMMIT: module attributes {{.*}}triton_ascend.cv_split_scheduling.applied = 1 : i32
// AUTO-COMMIT-NOT: triton_ascend.dynamic_cv_pipeline.rc
// AUTO-COMMIT-COUNT-2: scope.scope

// Invalid unroll factor three rejects CV before mutation.  DCVP then runs on
// the original module and succeeds, producing its own two scopes without a CV
// result or fallback error code.
// AUTO-FALLBACK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">}
// AUTO-FALLBACK-NOT: triton_ascend.cv_split_scheduling.applied
// AUTO-FALLBACK-NOT: triton_ascend.dynamic_cv_pipeline.rc
// AUTO-FALLBACK-COUNT-2: scope.scope
