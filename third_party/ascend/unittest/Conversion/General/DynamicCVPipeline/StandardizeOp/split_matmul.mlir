// RUN: triton-opt -ssbuf-standardize-op-pattern-match %s | FileCheck %s

module {
  // Case 1: Bias is a block argument (no defining op in the current block).
  // According to Rule 2, its value is unknown, so we split it.
  // CHECK-LABEL: func.func @case1_block_arg_bias
  // CHECK-SAME: (%[[A:.*]]: tensor<32x64xf32>, %[[B:.*]]: tensor<64x32xf32>, %[[BIAS:.*]]: tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[ZERO:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM:.*]] = linalg.matmul ins(%[[A]], %[[B]] : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ZERO]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[ADD:.*]] = arith.addf %[[MM]], %[[BIAS]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // CHECK: return %[[ADD]] : tensor<32x32xf32>
  func.func @case1_block_arg_bias(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>, %bias: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %mm = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %mm : tensor<32x32xf32>
  }

  // Case 2: Bias is a constant zero (filled via linalg.fill).
  // According to Rule 3, we bypass the split to avoid redundant additions.
  // CHECK-LABEL: func.func @case2_zero_bias
  // CHECK: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM:.*]] = linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ZERO]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK-NOT: arith.addf
  // CHECK: return %[[MM]]
  func.func @case2_zero_bias(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>) -> tensor<32x32xf32> {
    %cst = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x32xf32>
    %zero = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
    %mm = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%zero : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %mm : tensor<32x32xf32>
  }

  // Case 3: Result of the first matmul (%mm1) is directly used by another matmul (%mm2).
  // According to Rule 1, %mm1 must be split even though its bias is constant zero.
  // %mm2's bias is zero and its result is not used by any other matmul, so %mm2 is not split.
  // CHECK-LABEL: func.func @case3_result_used_by_matmul
  // CHECK: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[EMPTY32:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO32:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY32]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM1_ACC:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[MM1_ZERO:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[MM1_ACC]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM1:.*]] = linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[MM1_ZERO]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[ADD:.*]] = arith.addf %[[MM1]], %[[ZERO32]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // CHECK: %[[EMPTY16:.*]] = tensor.empty() : tensor<32x16xf32>
  // CHECK: %[[ZERO16:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY16]] : tensor<32x16xf32>) -> tensor<32x16xf32>
  // CHECK: %[[MM2:.*]] = linalg.matmul ins(%[[ADD]], %{{.*}} : tensor<32x32xf32>, tensor<32x16xf32>) outs(%[[ZERO16]] : tensor<32x16xf32>) -> tensor<32x16xf32>
  // CHECK-NOT: arith.addf
  // CHECK: return %[[MM2]]
  func.func @case3_result_used_by_matmul(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>, %B2: tensor<32x16xf32>) -> tensor<32x16xf32> {
    %cst = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x32xf32>
    %zero = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
    %mm1 = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%zero : tensor<32x32xf32>) -> tensor<32x32xf32>

    %empty_F = tensor.empty() : tensor<32x16xf32>
    %zero_F = linalg.fill ins(%cst : f32) outs(%empty_F : tensor<32x16xf32>) -> tensor<32x16xf32>
    %mm2 = linalg.matmul ins(%mm1, %B2 : tensor<32x32xf32>, tensor<32x16xf32>) outs(%zero_F : tensor<32x16xf32>) -> tensor<32x16xf32>
    return %mm2 : tensor<32x16xf32>
  }

  // Case 4: Bias is non-zero (filled with a non-zero constant 1.0).
  // It should be split into a zero-initialized matmul followed by an arith.addf.
  // CHECK-LABEL: func.func @case4_nonzero_bias
  // CHECK: %[[C1:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[BIAS:.*]] = linalg.fill ins(%[[C1]] : f32) outs(%{{.*}} : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[ZERO:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM:.*]] = linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ZERO]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[ADD:.*]] = arith.addf %[[MM]], %[[BIAS]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // CHECK: return %[[ADD]]
  func.func @case4_nonzero_bias(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>) -> tensor<32x32xf32> {
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<32x32xf32>
    %bias = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
    %mm = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %mm : tensor<32x32xf32>
  }

  // Case 5: Bias is a 1D-to-2D broadcasted tensor.
  // This non-zero bias definition should be split.
  // CHECK-LABEL: func.func @case5_broadcast_bias
  // CHECK: %[[BIAS:.*]] = linalg.broadcast ins(%{{.*}} : tensor<32xf32>) outs(%{{.*}} : tensor<32x32xf32>) dimensions = [0]
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[ZERO:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM:.*]] = linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ZERO]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[ADD:.*]] = arith.addf %[[MM]], %[[BIAS]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  func.func @case5_broadcast_bias(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>, %bias_1d: tensor<32xf32>) -> tensor<32x32xf32> {
    %empty_bias = tensor.empty() : tensor<32x32xf32>
    %bias = linalg.broadcast ins(%bias_1d : tensor<32xf32>) outs(%empty_bias : tensor<32x32xf32>) dimensions = [0]
    %mm = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %mm : tensor<32x32xf32>
  }

  // Case 6: Integer matmul.
  // Splitting should generate integer zero constant and use arith.addi for accumulation.
  // CHECK-LABEL: func.func @case6_integer_bias
  // CHECK-SAME: (%[[A:.*]]: tensor<32x64xi32>, %[[B:.*]]: tensor<64x32xi32>, %[[BIAS:.*]]: tensor<32x32xi32>) -> tensor<32x32xi32>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xi32>
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[ZERO:.*]] = linalg.fill ins(%[[C0]] : i32) outs(%[[EMPTY]] : tensor<32x32xi32>) -> tensor<32x32xi32>
  // CHECK: %[[MM:.*]] = linalg.matmul ins(%[[A]], %[[B]] : tensor<32x64xi32>, tensor<64x32xi32>) outs(%[[ZERO]] : tensor<32x32xi32>) -> tensor<32x32xi32>
  // CHECK: %[[ADD:.*]] = arith.addi %[[MM]], %[[BIAS]] {ssbuffer.add_from_matmul} : tensor<32x32xi32>
  func.func @case6_integer_bias(%A: tensor<32x64xi32>, %B: tensor<64x32xi32>, %bias: tensor<32x32xi32>) -> tensor<32x32xi32> {
    %mm = linalg.matmul ins(%A, %B : tensor<32x64xi32>, tensor<64x32xi32>) outs(%bias : tensor<32x32xi32>) -> tensor<32x32xi32>
    return %mm : tensor<32x32xi32>
  }

  // Case 7: Dynamic shape dimensions.
  // The split logic should fetch dynamic dimension sizes using tensor.dim.
  // CHECK-LABEL: func.func @case7_dynamic_shape
  // CHECK-SAME: (%[[A:.*]]: tensor<?x?xf32>, %[[B:.*]]: tensor<?x?xf32>, %[[BIAS:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[C0_IDX:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[DIM0:.*]] = tensor.dim %[[BIAS]], %[[C0_IDX]] : tensor<?x?xf32>
  // CHECK-DAG: %[[C1_IDX:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[DIM1:.*]] = tensor.dim %[[BIAS]], %[[C1_IDX]] : tensor<?x?xf32>
  // CHECK: %[[EMPTY:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
  // CHECK: %[[C0_FLT:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[ZERO:.*]] = linalg.fill ins(%[[C0_FLT]] : f32) outs(%[[EMPTY]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[MM:.*]] = linalg.matmul ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ZERO]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[ADD:.*]] = arith.addf %[[MM]], %[[BIAS]] {ssbuffer.add_from_matmul} : tensor<?x?xf32>
  func.func @case7_dynamic_shape(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %bias: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %mm = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%bias : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %mm : tensor<?x?xf32>
  }
}

