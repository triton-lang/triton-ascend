// RUN: triton-opt %s --split-input-file --protect-vf-tcb-reuse | FileCheck %s
// CHECK-LABEL: func.func @constant_fill_not_vf
// CHECK-NOT: sync_block
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @constant_fill_not_vf(%lo: index, %hi: index, %step: index,
                   %scalar: f32, %input: tensor<16x16xf32>, %out: memref<16x16xf32>) {
    scope.scope : () -> () {
      scf.for %query = %lo to %hi step %step {
        %zero = arith.constant 0.0 : f32
        %vf = linalg.fill ins(%zero : f32) outs(%input : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.for %kv = %lo to %hi step %step {

        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv = %lo to %hi step %step {

        } {ssbuffer.main_loop = 1 : i32}

        bufferization.materialize_in_destination %vf in writable %out : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
        bufferization.materialize_in_destination %vf in writable %out : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %buffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      scf.for %query = %lo to %hi step %step {
        scf.for %kv = %lo to %hi step %step {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%input : tensor<16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv = %lo to %hi step %step {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%input : tensor<16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
        } {ssbuffer.main_loop = 1 : i32}
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----
// CHECK-LABEL: func.func @inline_fill_not_vf
// CHECK-NOT: sync_block
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inline_fill_not_vf(%lo: index, %hi: index, %step: index,
                   %scalar: f32, %input: tensor<16x16xf32>, %out: memref<16x16xf32>) {
    scope.scope : () -> () {
      scf.for %query = %lo to %hi step %step {
        %vf = linalg.fill ins(%scalar : f32) outs(%input : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.for %kv = %lo to %hi step %step {
          %sum = arith.addf %vf, %input : tensor<16x16xf32>

        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv = %lo to %hi step %step {

        } {ssbuffer.main_loop = 1 : i32}

        bufferization.materialize_in_destination %input in writable %out : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
        bufferization.materialize_in_destination %input in writable %out : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %buffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      scf.for %query = %lo to %hi step %step {
        scf.for %kv = %lo to %hi step %step {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%input : tensor<16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv = %lo to %hi step %step {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%input : tensor<16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
        } {ssbuffer.main_loop = 1 : i32}
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----
// CHECK-LABEL: func.func @required_runtime_fill_is_vf
// CHECK: linalg.fill
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK-NEXT: scf.for
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK-NEXT: scf.for
// CHECK-NOT: sync_block
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @required_runtime_fill_is_vf(%lo: index, %hi: index, %step: index,
                   %scalar: f32, %input: tensor<16x16xf32>, %out: memref<16x16xf32>) {
    scope.scope : () -> () {
      scf.for %query = %lo to %hi step %step {
        %vf = linalg.fill ins(%scalar : f32) outs(%input : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.for %kv = %lo to %hi step %step {

        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv = %lo to %hi step %step {

        } {ssbuffer.main_loop = 1 : i32}

        bufferization.materialize_in_destination %vf in writable %out : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
        bufferization.materialize_in_destination %vf in writable %out : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %buffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      scf.for %query = %lo to %hi step %step {
        scf.for %kv = %lo to %hi step %step {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%input : tensor<16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv = %lo to %hi step %step {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%input : tensor<16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
        } {ssbuffer.main_loop = 1 : i32}
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
