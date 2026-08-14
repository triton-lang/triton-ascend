// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @simt_scope
// CHECK-SAME: parallel_mode = "mix_simd_simt"
tt.func public @simt_scope(%arg0: !tt.ptr<f32>) {
  scope.scope : () -> () {
    %c0 = arith.constant 0 : i32
    %p = tt.addptr %arg0, %c0 : !tt.ptr<f32>, i32
    %c = arith.constant 1.0 : f32
    tt.store %p, %c : !tt.ptr<f32>
    scope.return
  } {vector_type = "simt"}
  tt.return
}
