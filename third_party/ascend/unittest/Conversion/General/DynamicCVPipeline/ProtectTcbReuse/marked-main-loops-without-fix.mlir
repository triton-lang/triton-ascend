// The main_loop contract suffices; an explicit FIX op is not required.
// RUN: triton-opt %s --protect-tcb-reuse | FileCheck %s
// CHECK-LABEL: func.func @marked_main_loops_without_fix
// CHECK: arith.addf
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 1
// CHECK-NEXT: scf.for
// CHECK: bufferization.materialize_in_destination
// CHECK: scope.scope
// CHECK: scf.for
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_FIX>] flag = 1
// CHECK-NEXT: scf.for
// CHECK-NOT: sync_block
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @marked_main_loops_without_fix(%lo: index, %hi: index, %step: index,
                   %input: tensor<16x16xf32>, %out: memref<16x16xf32>) {
    scope.scope : () -> () {
      scf.for %query = %lo to %hi step %step {
        %vf = arith.addf %input, %input : tensor<16x16xf32>
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
        } {ssbuffer.main_loop = 0 : i32}
        scf.for %kv = %lo to %hi step %step {
        } {ssbuffer.main_loop = 1 : i32}
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
