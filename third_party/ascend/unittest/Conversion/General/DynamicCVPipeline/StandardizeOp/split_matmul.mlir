// RUN: triton-opt -ssbuf-standardize-op-pattern-match %s | FileCheck %s

module {
  // Case 1: Bias is a block argument (no defining op in the current block).
  // According to Rule 2, its value is unknown, so we split it.
  // CHECK-LABEL: func.func @case1_block_arg_bias
  // CHECK-SAME: (%[[A:.*]]: tensor<32x64xf32>, %[[B:.*]]: tensor<64x32xf32>, %[[BIAS:.*]]: tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK-DAG: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
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
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[EMPTY32:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO32:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY32]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM1:.*]] = linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ZERO32]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK-NOT: arith.addf
  // CHECK: %[[EMPTY16:.*]] = tensor.empty() : tensor<32x16xf32>
  // CHECK: %[[ZERO16:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY16]] : tensor<32x16xf32>) -> tensor<32x16xf32>
  // CHECK: %[[MM2:.*]] = linalg.matmul ins(%[[MM1]], %{{.*}} : tensor<32x32xf32>, tensor<32x16xf32>) outs(%[[ZERO16]] : tensor<32x16xf32>) -> tensor<32x16xf32>
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
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[BIAS_EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[BIAS:.*]] = linalg.fill ins(%[[C1]] : f32) outs(%[[BIAS_EMPTY]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
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
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[BIAS_EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[BIAS:.*]] = linalg.broadcast ins(%{{.*}} : tensor<32xf32>) outs(%[[BIAS_EMPTY]] : tensor<32x32xf32>) dimensions = [0]
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
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
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xi32>
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
  // CHECK-DAG: %[[C0_FLT:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[EMPTY:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
  // CHECK: %[[ZERO:.*]] = linalg.fill ins(%[[C0_FLT]] : f32) outs(%[[EMPTY]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[MM:.*]] = linalg.matmul ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ZERO]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[ADD:.*]] = arith.addf %[[MM]], %[[BIAS]] {ssbuffer.add_from_matmul} : tensor<?x?xf32>
  func.func @case7_dynamic_shape(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %bias: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %mm = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%bias : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %mm : tensor<?x?xf32>
  }

  // Case 8: Matmul inside scf.for with loop-carried bias.
  // The bias is a loop-carried variable with zero initial value.
  // The matmul is marked as loop_carried_l0c and not split (no addf).
  // CHECK-LABEL: func.func @case8_for_loop_carried_bias
  // CHECK: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO_INIT:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[BIAS:.*]] = %[[ZERO_INIT]]) -> (tensor<32x32xf32>)
  // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[BIAS]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK-NOT: arith.addf
  // CHECK: scf.yield %[[MM]] : tensor<32x32xf32>
  func.func @case8_for_loop_carried_bias(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>) -> tensor<32x32xf32> {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x32xf32>
    %zero_init = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
    %result = scf.for %iv = %c0 to %c10 step %c1 iter_args(%bias = %zero_init) -> (tensor<32x32xf32>) {
      %mm = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %mm : tensor<32x32xf32>
    }
    return %result : tensor<32x32xf32>
  }

  // Case 9: Matmul inside scf.if without else branch.
  // The if statement result is not used, so the entire if block is removed.
  // Only the zero initialization remains.
  // CHECK-LABEL: func.func @case9_if_no_else
  // CHECK: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK-NOT: scf.if
  // CHECK-NOT: linalg.matmul
  // CHECK-NOT: arith.addf
  // CHECK: return %[[ZERO]] : tensor<32x32xf32>
  func.func @case9_if_no_else(%cond: i1, %A: tensor<32x64xf32>, %B: tensor<64x32xf32>, %bias: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %cst = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x32xf32>
    %zero = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.if %cond {
      %mm = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield
    }
    return %zero : tensor<32x32xf32>
  }

  // Case 10: Matmul inside scf.if with both then and else branches.
  // Both branches have matmul with bias from function argument, so both are split.
  // CHECK-LABEL: func.func @case10_if_else
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: scf.if %{{.*}} -> (tensor<32x32xf32>) {
  // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO1:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY1]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM1:.*]] = linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ZERO1]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[ADD1:.*]] = arith.addf %[[MM1]], %{{.*}} {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // CHECK: scf.yield %[[ADD1]] : tensor<32x32xf32>
  // CHECK: } else {
  // CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO2:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY2]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM2:.*]] = linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ZERO2]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[ADD2:.*]] = arith.addf %[[MM2]], %{{.*}} {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // CHECK: scf.yield %[[ADD2]] : tensor<32x32xf32>
  // CHECK: }
  func.func @case10_if_else(%cond: i1, %A: tensor<32x64xf32>, %B: tensor<64x32xf32>, %bias: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %result = scf.if %cond -> (tensor<32x32xf32>) {
      %mm = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %mm : tensor<32x32xf32>
    } else {
      %mm = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %mm : tensor<32x32xf32>
    }
    return %result : tensor<32x32xf32>
  }

  // Case 11: Nested scf.for loops with matmul.
  // The matmul is in an inner loop with bias as a loop-carried variable with zero initial value.
  // The inner matmul is marked as loop_carried_l0c and not split (no addf).
  // CHECK-LABEL: func.func @case11_nested_for
  // CHECK: scf.for {{.*}} {
  // CHECK: %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[BIAS:.*]] = %{{.*}}) -> (tensor<32x32xf32>)
  // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[BIAS]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK-NOT: arith.addf
  // CHECK: scf.yield %[[MM]] : tensor<32x32xf32>
  // CHECK: }
  // CHECK: }
  func.func @case11_nested_for(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>) -> tensor<32x32xf32> {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x32xf32>
    %zero_init = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
    
    %outer_result = scf.for %outer_iv = %c0 to %c10 step %c1 iter_args(%outer_acc = %zero_init) -> (tensor<32x32xf32>) {
      %inner_result = scf.for %inner_iv = %c0 to %c10 step %c1 iter_args(%bias = %outer_acc) -> (tensor<32x32xf32>) {
        %mm = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias : tensor<32x32xf32>) -> tensor<32x32xf32>
        scf.yield %mm : tensor<32x32xf32>
      }
      scf.yield %inner_result : tensor<32x32xf32>
    }
    return %outer_result : tensor<32x32xf32>
  }
}

