// RUN: triton-opt %s --split-input-file --protect-tcb-reuse | FileCheck %s
// CHECK-LABEL: func.func @one_main_loop
// CHECK-NOT: sync_block
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @one_main_loop(%lo: index, %hi: index, %step: index,
                   %input: tensor<16x16xf32>, %out: memref<16x16xf32>) {
    scope.scope : () -> () {
      scf.for %query = %lo to %hi step %step {
        %vf = arith.addf %input, %input : tensor<16x16xf32>
        scf.for %kv0 = %lo to %hi step %step {
        } {ssbuffer.main_loop = 0 : i32}
        bufferization.materialize_in_destination %vf in writable %out : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
        bufferization.materialize_in_destination %vf in writable %out : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %buffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      scf.for %query = %lo to %hi step %step {
        scf.for %kv0 = %lo to %hi step %step {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%input : tensor<16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
        } {ssbuffer.main_loop = 0 : i32}
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----
// CHECK-LABEL: func.func @two_main_loops
// CHECK: arith.addf
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 1
// CHECK-NEXT: scf.for
// CHECK-NOT: sync_block
// CHECK: scope.scope
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_FIX>] flag = 1
// CHECK-NEXT: scf.for
// CHECK-NOT: sync_block
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @two_main_loops(%lo: index, %hi: index, %step: index,
                   %input: tensor<16x16xf32>, %out: memref<16x16xf32>) {
    scope.scope : () -> () {
      scf.for %query = %lo to %hi step %step {
        %vf = arith.addf %input, %input : tensor<16x16xf32>
        scf.for %kv0 = %lo to %hi step %step {
        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv1 = %lo to %hi step %step {
        } {ssbuffer.main_loop = 1 : i32}
        bufferization.materialize_in_destination %vf in writable %out : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
        bufferization.materialize_in_destination %vf in writable %out : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %buffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      scf.for %query = %lo to %hi step %step {
        scf.for %kv0 = %lo to %hi step %step {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%input : tensor<16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv1 = %lo to %hi step %step {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%input : tensor<16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
        } {ssbuffer.main_loop = 1 : i32}
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----
// CHECK-LABEL: func.func @three_main_loops
// CHECK-NOT: sync_block
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @three_main_loops(%lo: index, %hi: index, %step: index,
                   %input: tensor<16x16xf32>, %out: memref<16x16xf32>) {
    scope.scope : () -> () {
      scf.for %query = %lo to %hi step %step {
        %vf = arith.addf %input, %input : tensor<16x16xf32>
        scf.for %kv0 = %lo to %hi step %step {
        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv1 = %lo to %hi step %step {
        } {ssbuffer.main_loop = 1 : i32}
        scf.for %kv2 = %lo to %hi step %step {
        } {ssbuffer.main_loop = 2 : i32}
        bufferization.materialize_in_destination %vf in writable %out : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
        bufferization.materialize_in_destination %vf in writable %out : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %buffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      scf.for %query = %lo to %hi step %step {
        scf.for %kv0 = %lo to %hi step %step {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%input : tensor<16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv1 = %lo to %hi step %step {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%input : tensor<16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
        } {ssbuffer.main_loop = 1 : i32}
        scf.for %kv2 = %lo to %hi step %step {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%input : tensor<16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
        } {ssbuffer.main_loop = 2 : i32}
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
