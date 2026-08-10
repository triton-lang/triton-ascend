// RUN: sed 's/inter_core_buf_count = 2/inter_core_buf_count = 1/' %S/cv_split_scheduling_fa.mlir | triton-opt "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" | FileCheck %s --check-prefix=DEPTH1
// RUN: sed 's/inter_core_buf_count = 2/inter_core_buf_count = 3/' %S/cv_split_scheduling_fa.mlir | triton-opt "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=REJECT
//
// This contract proves that CV split consumes DynamicCVPipeline's canonical
// inter-core buffer-count attribute.  Depth one produces one physical buffer
// per transfer lineage.  Unsupported depth three is rejected transactionally
// and leaves the original unscoped loop intact.

// DEPTH1: memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
// DEPTH1-NOT: memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
// DEPTH1: memref.alloc() : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
// DEPTH1-NOT: memref.alloc() : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
// DEPTH1: memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
// DEPTH1-NOT: memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
// DEPTH1: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
// DEPTH1: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 11

// REJECT-NOT: scope.scope
// REJECT: scf.for {{.*}} step %c32_i32
