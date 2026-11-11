// RUN: triton-opt %s --split-input-file --protect-vf-tcb-reuse | FileCheck %s
// CHECK-LABEL: func.func @vf_before_main
// CHECK: arith.addf
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK-NEXT: scf.for
// CHECK: scope.scope
// CHECK: scf.for
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK-NEXT: scf.for
// CHECK: hivm.hir.fixpipe
// CHECK-NOT: sync_block
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vf_before_main(%lo: index, %hi: index, %step: index,
                   %input: tensor<16x16xf32>, %out: memref<16x16xf32>) {
    scope.scope : () -> () {
      scf.for %query = %lo to %hi step %step {
        %vf = arith.addf %input, %input : tensor<16x16xf32>
        scf.for %kv = %lo to %hi step %step {

        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv = %lo to %hi step %step {

        } {ssbuffer.main_loop = 1 : i32}

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
// CHECK-LABEL: func.func @scalar_only
// CHECK-NOT: sync_block
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @scalar_only(%lo: index, %hi: index, %step: index,
                   %input: tensor<16x16xf32>, %out: memref<16x16xf32>) {
    scope.scope : () -> () {
      scf.for %query = %lo to %hi step %step {
        %scalar = arith.addi %lo, %hi : index
        scf.for %kv = %lo to %hi step %step {

        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv = %lo to %hi step %step {

        } {ssbuffer.main_loop = 1 : i32}
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
// CHECK-LABEL: func.func @vf_inside_main
// CHECK-NOT: sync_block
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vf_inside_main(%lo: index, %hi: index, %step: index,
                   %input: tensor<16x16xf32>, %out: memref<16x16xf32>) {
    scope.scope : () -> () {
      scf.for %query = %lo to %hi step %step {

        scf.for %kv = %lo to %hi step %step {
          %vf = arith.addf %input, %input : tensor<16x16xf32>
        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv = %lo to %hi step %step {
          %vf = arith.addf %input, %input : tensor<16x16xf32>
        } {ssbuffer.main_loop = 1 : i32}


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
// CHECK-LABEL: func.func @vf_after_main
// CHECK-NOT: sync_block
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vf_after_main(%lo: index, %hi: index, %step: index,
                   %input: tensor<16x16xf32>, %out: memref<16x16xf32>) {
    scope.scope : () -> () {
      scf.for %query = %lo to %hi step %step {

        scf.for %kv = %lo to %hi step %step {

        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv = %lo to %hi step %step {

        } {ssbuffer.main_loop = 1 : i32}
        %vf = arith.addf %input, %input : tensor<16x16xf32>

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
// CHECK-LABEL: func.func @store_without_vf
// CHECK-NOT: sync_block
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @store_without_vf(%lo: index, %hi: index, %step: index,
                   %input: tensor<16x16xf32>, %out: memref<16x16xf32>) {
    scope.scope : () -> () {
      scf.for %query = %lo to %hi step %step {

        scf.for %kv = %lo to %hi step %step {

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
// CHECK-LABEL: func.func @vf_serial_main
// CHECK: arith.addf
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK-NEXT: scf.for
// CHECK-NOT: sync_block
// CHECK: scope.scope
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK-NEXT: scf.for
// CHECK-NOT: sync_block
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vf_serial_main(%lo: index, %hi: index, %step: index,
                   %input: tensor<16x16xf32>, %out: memref<16x16xf32>) {
    scope.scope : () -> () {
      scf.for %query = %lo to %hi step %step {
        %vf = arith.addf %input, %input : tensor<16x16xf32>
        scf.for %kv = %lo to %hi step %step {

        } {ssbuffer.main_loop = 0 : i32}
        scf.for %secondKv = %lo to %hi step %step {
        } {ssbuffer.main_loop = 1 : i32}
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %buffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      scf.for %query = %lo to %hi step %step {
        scf.for %kv = %lo to %hi step %step {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%input : tensor<16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
        } {ssbuffer.main_loop = 0 : i32}
        scf.for %secondKv = %lo to %hi step %step {
        } {ssbuffer.main_loop = 1 : i32}
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----
// CHECK-LABEL: func.func @vf_nested
// CHECK: arith.addf
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK-NEXT: scf.for
// CHECK-NOT: sync_block
// CHECK: scope.scope
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK-NEXT: scf.for
// CHECK-NOT: sync_block
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vf_nested(%lo: index, %hi: index, %step: index,
                   %input: tensor<16x16xf32>, %out: memref<16x16xf32>) {
    scope.scope : () -> () {
      scf.for %batch = %lo to %hi step %step {
      scf.for %query = %lo to %hi step %step {
        %vf = arith.addf %input, %input : tensor<16x16xf32>
        scf.for %kv = %lo to %hi step %step {

        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv = %lo to %hi step %step {

        } {ssbuffer.main_loop = 1 : i32}
      }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %buffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      scf.for %batch = %lo to %hi step %step {
      scf.for %query = %lo to %hi step %step {
        scf.for %kv = %lo to %hi step %step {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%input : tensor<16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv = %lo to %hi step %step {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%input : tensor<16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
        } {ssbuffer.main_loop = 1 : i32}
      }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----
// CHECK-LABEL: func.func @vf_zero_trip
// CHECK: arith.addf
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK-NEXT: scf.for
// CHECK-NOT: sync_block
// CHECK: scope.scope
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK-NEXT: scf.for
// CHECK-NOT: sync_block
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vf_zero_trip(%lo: index, %hi: index, %step: index,
                   %input: tensor<16x16xf32>, %out: memref<16x16xf32>) {
    scope.scope : () -> () {
      scf.for %query = %lo to %lo step %step {
        %vf = arith.addf %input, %input : tensor<16x16xf32>
        scf.for %kv = %lo to %hi step %step {

        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv = %lo to %hi step %step {

        } {ssbuffer.main_loop = 1 : i32}

      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %buffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      scf.for %query = %lo to %lo step %step {
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
