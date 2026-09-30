// RUN: triton-opt %s --triton-to-unstructure="compile-on-910-95=false compile-mode=simd_simt_template" --verify-each | FileCheck %s

// CHECK-LABEL: tt.func public @kernel(
// CHECK: %[[PTR:.*]] = tt.call @helper(
// CHECK: tt.splat %[[PTR]] : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK: tt.load
// CHECK-LABEL: tt.func private @helper(
// CHECK: tt.return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @kernel(%ptr: !tt.ptr<f32>, %out: !tt.ptr<f32>) attributes {noinline = false} {
    %ret = tt.call @helper(%ptr) : (!tt.ptr<f32>) -> !tt.ptr<f32>
    %splat = tt.splat %ret : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %value = tt.load %splat : tensor<32x!tt.ptr<f32>>
    %out_vec = tt.splat %out : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    tt.store %out_vec, %value : tensor<32x!tt.ptr<f32>>
    tt.return
  }
  tt.func private @helper(%arg: !tt.ptr<f32>) -> !tt.ptr<f32> attributes {noinline = true} {
    tt.return %arg : !tt.ptr<f32>
  }
}
