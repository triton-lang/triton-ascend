// RUN: triton-opt '--add_dynamic_cv_pipeline=compile-on-910-95=True' --split-input-file %s | FileCheck %s

// CHECK-LABEL: module attributes {triton_ascend.dynamic_cv_pipeline.rc = 2 : i32}
module {
  // CHECK-LABEL: func.func @fallback_pure_vv(
  // CHECK-NOT: ssbuffer.
  // CHECK: arith.addf %arg0, %arg1 : tensor<16xf32>
  // CHECK-NOT: ssbuffer.
  // CHECK: arith.mulf {{%.*}}, %arg0 : tensor<16xf32>
  // CHECK-NOT: ssbuffer.
  // CHECK: return {{%.*}} : tensor<16xf32>
  func.func @fallback_pure_vv(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %add = arith.addf %arg0, %arg1 : tensor<16xf32>
    %mul = arith.mulf %add, %arg0 : tensor<16xf32>
    return %mul : tensor<16xf32>
  }
}

// -----

// CHECK-LABEL: module attributes {triton_ascend.dynamic_cv_pipeline.rc = 2 : i32}
module {
  // CHECK-LABEL: func.func @fallback_pure_cc(
  // CHECK-NOT: ssbuffer.
  // CHECK: linalg.matmul ins(%arg0, %arg1 : memref<16x16xf16>, memref<16x16xf16>) outs(%arg2 : memref<16x16xf32>)
  // CHECK-NOT: ssbuffer.
  // CHECK: return
  func.func @fallback_pure_cc(
      %arg0: memref<16x16xf16>,
      %arg1: memref<16x16xf16>,
      %arg2: memref<16x16xf32>) {
    linalg.matmul
      ins(%arg0, %arg1 : memref<16x16xf16>, memref<16x16xf16>)
      outs(%arg2 : memref<16x16xf32>)
    return
  }
}
