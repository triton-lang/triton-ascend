// RUN: triton-opt --plan-compute-block %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @matmul_prelegalize_add_has_attr(
  // CHECK: [[EMPTY:%[0-9]+]] = tensor.empty() {ssbuffer.block_id = [[CUBE:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} : tensor<4x4xf32>
  // CHECK-NEXT: [[ZERO:%[A-Za-z0-9_]+]] = arith.constant {ssbuffer.block_id = [[CUBE]] : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
  // CHECK-NEXT: [[FILL:%[0-9]+]] = linalg.fill {ssbuffer.block_id = [[CUBE]] : i32, ssbuffer.core_type = "CUBE"} ins([[ZERO]] : f32) outs([[EMPTY]] : tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT: [[MM:%[0-9]+]] = linalg.matmul {ssbuffer.block_id = [[CUBE]] : i32, ssbuffer.core_type = "CUBE"} ins(%arg0, %arg1 : tensor<4x4xf16>, tensor<4x4xf16>) outs([[FILL]] : tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT: [[ADD:%[0-9]+]] = arith.addf [[MM]], %arg2 {ssbuffer.add_from_matmul, ssbuffer.block_id = [[VEC:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<4x4xf32>
  // CHECK-NEXT: return [[ADD]] : tensor<4x4xf32>
  func.func @matmul_prelegalize_add_has_attr(
      %a: tensor<4x4xf16>,
      %b: tensor<4x4xf16>,
      %acc: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %mm = linalg.matmul
      ins(%a, %b : tensor<4x4xf16>, tensor<4x4xf16>)
      outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %mm : tensor<4x4xf32>
  }
}
