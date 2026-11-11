// RUN: triton-opt --plan-compute-block %s | FileCheck %s

// A packed load is one VECTOR operation even when its body contains untagged
// scalar ops and a padding if. The enclosing if must share one vector block
// with its body instead of inspecting the packed load's internal operations.
module {
  // CHECK-LABEL: func.func @vector_if_with_packed_load(
  // CHECK: scf.if %arg0 {
  // CHECK: %[[LOAD:.*]] = scope.scope : () -> tensor<64xf16> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64xf16>
  // CHECK: scf.if %arg1 {
  // CHECK: linalg.fill ins({{.*}}) outs(%[[ALLOC]] : memref<64xf16>)
  // CHECK: scope.return
  // CHECK-NEXT: } {ssbuffer.block_id = [[BLOCK:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR", ssbuffer.load}
  // CHECK: %[[SUM:.*]] = arith.addf %[[LOAD]], %[[LOAD]] {ssbuffer.block_id = [[BLOCK]] : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: bufferization.materialize_in_destination %[[SUM]] in writable %arg3 {ssbuffer.block_id = [[BLOCK]] : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK-NEXT: } {ssbuffer.block_id = [[BLOCK]] : i32}
  func.func @vector_if_with_packed_load(%cond: i1, %pad: i1, %gm: memref<64xf16>, %out: memref<64xf16>) {
    %zero = arith.constant 0.0 : f16
    scf.if %cond {
      %load = scope.scope : () -> tensor<64xf16> {
        %alloc = memref.alloc() : memref<64xf16>
        scf.if %pad {
          linalg.fill ins(%zero : f16) outs(%alloc : memref<64xf16>)
        }
        memref.copy %gm, %alloc : memref<64xf16> to memref<64xf16>
        %t = bufferization.to_tensor %alloc restrict writable : memref<64xf16> to tensor<64xf16>
        scope.return %t : tensor<64xf16>
      } {ssbuffer.core_type = "VECTOR", ssbuffer.load}
      %sum = arith.addf %load, %load : tensor<64xf16>
      bufferization.materialize_in_destination %sum in writable %out : (tensor<64xf16>, memref<64xf16>) -> ()
    }
    return
  }
}
