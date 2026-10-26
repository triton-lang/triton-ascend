// RUN: triton-opt %s --triton-to-unstructure --triton-to-linalg="named-ops=True" | FileCheck %s
// A multi-result, effectful call feeds both loop state and a data consumer.
// scf.yield has no results but is not a metadata-only consumer.
// CHECK-LABEL: func.func @custom_loop_yield
// CHECK: scf.for
// CHECK: %[[CALL:.*]]:2 = hivm.hir.custom
// CHECK-NOT: hivm.hir.custom
// CHECK: scf.yield
// CHECK-NOT: hivm.hir.custom
// CHECK: return
module {
  tt.func public @custom_loop_yield(%dst: !tt.ptr<f32>) {
    %zero = arith.constant dense<0.0> : tensor<64xf32>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %base = tt.splat %dst : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%m = %zero) -> tensor<64xf32> {
      %r:2 = hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "state_step"} "state_step" ins(%m : tensor<64xf32>) outs(%zero, %zero : tensor<64xf32>, tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>)
      tt.store %ptr, %r#0 : tensor<64x!tt.ptr<f32>>
      scf.yield %r#1 : tensor<64xf32>
    }
    tt.store %ptr, %result : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
