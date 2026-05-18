// RUN: triton-opt -ssbuf-standardize-op-pattern-match %s | FileCheck %s

module {
  // Case 1: Bias is a block argument (no defining op in the current block).
  // According to Rule 2, its value is unknown, so we split it.
  // CHECK-LABEL: func.func @case1_block_arg_bias
  // CHECK-SAME: (%[[A:.*]]: tensor<32x64xf32>, %[[B:.*]]: tensor<64x32xf32>, %[[BIAS:.*]]: tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%[[A]], %[[B]] : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ZERO]] : tensor<32x32xf32>) -> tensor<32x32xf32>
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
  // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ZERO]] : tensor<32x32xf32>) -> tensor<32x32xf32>
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
  // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO1:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY1]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO_MM1:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY2]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM1:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ZERO_MM1]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[ADD:.*]] = arith.addf %[[MM1]], %[[ZERO1]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<32x16xf32>
  // CHECK: %[[ZERO2:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY3]] : tensor<32x16xf32>) -> tensor<32x16xf32>
  // CHECK: %[[MM2:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%[[ADD]], %{{.*}} : tensor<32x32xf32>, tensor<32x16xf32>) outs(%[[ZERO2]] : tensor<32x16xf32>) -> tensor<32x16xf32>
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
  // CHECK: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[C1:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[BIAS:.*]] = linalg.fill ins(%[[C1]] : f32) outs(%[[EMPTY1]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY2]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ZERO]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[ADD:.*]] = arith.addf %[[MM]], %[[BIAS]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // CHECK: return %[[ADD]]
  func.func @case4_nonzero_bias(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>) -> tensor<32x32xf32> {
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<32x32xf32>
    %bias = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
    %mm = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %mm : tensor<32x32xf32>
  }

  // Case 5: Integer matmul.
  // Splitting should generate integer zero constant and use arith.addi for accumulation.
  // CHECK-LABEL: func.func @case5_integer_bias
  // CHECK-SAME: (%[[A:.*]]: tensor<32x64xi32>, %[[B:.*]]: tensor<64x32xi32>, %[[BIAS:.*]]: tensor<32x32xi32>) -> tensor<32x32xi32>
  // CHECK: %[[C0_I32:.*]] = arith.constant 0 : i32
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xi32>
  // CHECK: %[[ZERO:.*]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[EMPTY]] : tensor<32x32xi32>) -> tensor<32x32xi32>
  // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%[[A]], %[[B]] : tensor<32x64xi32>, tensor<64x32xi32>) outs(%[[ZERO]] : tensor<32x32xi32>) -> tensor<32x32xi32>
  // CHECK: %[[ADD:.*]] = arith.addi %[[MM]], %[[BIAS]] {ssbuffer.add_from_matmul} : tensor<32x32xi32>
  func.func @case5_integer_bias(%A: tensor<32x64xi32>, %B: tensor<64x32xi32>, %bias: tensor<32x32xi32>) -> tensor<32x32xi32> {
    %mm = linalg.matmul ins(%A, %B : tensor<32x64xi32>, tensor<64x32xi32>) outs(%bias : tensor<32x32xi32>) -> tensor<32x32xi32>
    return %mm : tensor<32x32xi32>
  }

  // Case 6: Dynamic shape dimensions.
  // The split logic should fetch dynamic dimension sizes using tensor.dim.
  // CHECK-LABEL: func.func @case6_dynamic_shape
  // CHECK-SAME: (%[[A:.*]]: tensor<?x?xf32>, %[[B:.*]]: tensor<?x?xf32>, %[[BIAS:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[C1_IDX:.*]] = arith.constant 1 : index
  // CHECK: %[[C0_IDX:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[BIAS]], %[[C0_IDX]] : tensor<?x?xf32>
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[BIAS]], %[[C1_IDX]] : tensor<?x?xf32>
  // CHECK: %[[EMPTY:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
  // CHECK: %[[ZERO:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ZERO]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[ADD:.*]] = arith.addf %[[MM]], %[[BIAS]] {ssbuffer.add_from_matmul} : tensor<?x?xf32>
  func.func @case6_dynamic_shape(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %bias: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %mm = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%bias : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %mm : tensor<?x?xf32>
  }

  // Case 7: Matmul inside scf.if with both then and else branches.
  // Both branches have matmul with bias from function argument, so both are split.
  // CHECK-LABEL: func.func @case7_if_else
  // CHECK: scf.if %{{.*}} -> (tensor<32x32xf32>) {
  // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO1:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[EMPTY1]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM1:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ZERO1]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[ADD1:.*]] = arith.addf %[[MM1]], %{{.*}} {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // CHECK: scf.yield %[[ADD1]] : tensor<32x32xf32>
  // CHECK: } else {
  // CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO2:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[EMPTY2]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM2:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ZERO2]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[ADD2:.*]] = arith.addf %[[MM2]], %{{.*}} {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // CHECK: scf.yield %[[ADD2]] : tensor<32x32xf32>
  // CHECK: }
  func.func @case7_if_else(%cond: i1, %A: tensor<32x64xf32>, %B: tensor<64x32xf32>, %bias: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %result = scf.if %cond -> (tensor<32x32xf32>) {
      %mm = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %mm : tensor<32x32xf32>
    } else {
      %mm = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %mm : tensor<32x32xf32>
    }
    return %result : tensor<32x32xf32>
  }

  // Case 8: Nested scf.for loops with matmul.
  // The matmul is in an inner loop with bias as a loop-carried variable with zero initial value.
  // The inner matmul is marked as loop_carried_l0c and not split (no addf).
  // CHECK-LABEL: func.func @case8_nested_for
  // CHECK: scf.for {{.*}} {
  // CHECK: %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[BIAS:.*]] = %{{.*}}) -> (tensor<32x32xf32>) {
  // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[BIAS]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK-NOT: arith.addf
  // CHECK: scf.yield %[[MM]] : tensor<32x32xf32>
  // CHECK: }
  // CHECK: }
  func.func @case8_nested_for(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>) -> tensor<32x32xf32> {
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

  // Case 9: Nested scf.for loops with non-zero initial value.
  // The initial bias is filled with 1.0 (non-zero), so the behavior differs from case8.
  // An empty tensor is created and used as iter_args, and an arith.addf is added after the loop.
  // CHECK-LABEL: func.func @case9_nested_for_nonzero_init
  // CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[NONZERO_INIT:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY1]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO_FILL:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[EMPTY2]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[FOR_RESULT:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[ZERO_FILL]]) -> (tensor<32x32xf32>) {
  // CHECK: %[[INNER_RESULT:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<32x32xf32>) {
  // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%{{.*}} : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK-NOT: arith.addf
  // CHECK: scf.yield %[[MM]] : tensor<32x32xf32>
  // CHECK: }
  // CHECK: scf.yield %[[INNER_RESULT]] : tensor<32x32xf32>
  // CHECK: }
  // CHECK: %[[ADD:.*]] = arith.addf %[[FOR_RESULT]], %[[NONZERO_INIT]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // CHECK: return %[[ADD]] : tensor<32x32xf32>
  func.func @case9_nested_for_nonzero_init(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>) -> tensor<32x32xf32> {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<32x32xf32>
    %nonzero_init = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>

    %outer_result = scf.for %outer_iv = %c0 to %c10 step %c1 iter_args(%outer_acc = %nonzero_init) -> (tensor<32x32xf32>) {
      %inner_result = scf.for %inner_iv = %c0 to %c10 step %c1 iter_args(%bias = %outer_acc) -> (tensor<32x32xf32>) {
        %mm = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias : tensor<32x32xf32>) -> tensor<32x32xf32>
        scf.yield %mm : tensor<32x32xf32>
      }
      scf.yield %inner_result : tensor<32x32xf32>
    }
    return %outer_result : tensor<32x32xf32>
  }

  // Case 10: Cascaded matmul chains with zero and non-zero initialization.
  // Chain 1 (zero init): c -> %1 -> %2 -> %3
  // Chain 2 (non-zero init): d -> %4 -> %5 -> %6
  // Zero-init chain: matmul ops use previous output directly (L0C chain, no split).
  // Non-zero chain: each matmul is split into empty + matmul + addf pattern.
  // CHECK-LABEL: func.func @case10_cascaded_matmul_chains
  // CHECK: %[[CST_ZERO:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[CST_NONZERO:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[C:.*]] = linalg.fill ins(%[[CST_ZERO]] : f32) outs(%[[EMPTY0]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // Chain 1: zero init - L0C chain, no split
  // CHECK: %[[MM1:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[C]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM2:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[MM1]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM3:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[MM2]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK-NOT: arith.addf {{.*}} %[[MM3]]
  // Chain 2: non-zero init - each matmul is split
  // CHECK: %[[EMPTY_D:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[D:.*]] = linalg.fill ins(%[[CST_NONZERO]] : f32) outs(%[[EMPTY_D]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO4:.*]] = linalg.fill ins(%[[CST_ZERO]] : f32) outs(%[[EMPTY4]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM4:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ZERO4]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[ADD4:.*]] = arith.addf %[[MM4]], %[[D]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // CHECK: %[[EMPTY5:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO5:.*]] = linalg.fill ins(%[[CST_ZERO]] : f32) outs(%[[EMPTY5]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM5:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ZERO5]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[ADD5:.*]] = arith.addf %[[MM5]], %[[ADD4]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // CHECK: %[[EMPTY6:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO6:.*]] = linalg.fill ins(%[[CST_ZERO]] : f32) outs(%[[EMPTY6]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[MM6:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ZERO6]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[ADD6:.*]] = arith.addf %[[MM6]], %[[ADD5]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // Final combination
  // CHECK: %[[RESULT:.*]] = arith.addf %[[MM3]], %[[ADD6]] : tensor<32x32xf32>
  // CHECK: return %[[RESULT]] : tensor<32x32xf32>
  func.func @case10_cascaded_matmul_chains(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>) -> tensor<32x32xf32> {
    %cst_zero = arith.constant 0.0 : f32
    %cst_nonzero = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<32x32xf32>

    // Chain 1: zero initialization
    %c = linalg.fill ins(%cst_zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
    %1 = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%c : tensor<32x32xf32>) -> tensor<32x32xf32>
    %2 = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %3 = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>

    // Chain 2: non-zero initialization
    %empty2 = tensor.empty() : tensor<32x32xf32>
    %d = linalg.fill ins(%cst_nonzero : f32) outs(%empty2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %4 = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%d : tensor<32x32xf32>) -> tensor<32x32xf32>
    %5 = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %6 = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>

    // Combine the two chains
    %result = arith.addf %3, %6 : tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }

  // Case 11: Cascaded matmul chains wrapped in separate scf.for loops.
  // Each matmul is in a separate for loop, with bias as iter_args.
  // Chain 1 (zero init): c -> for1(%mm1) -> for2(%mm2) -> for3(%mm3)
  // Chain 2 (non-zero init): d -> for4(%mm4) -> for5(%mm5) -> for6(%mm6)
  // Unlike case8 (nested for loops), separate for loops cause each matmul to be split.
  // Each for loop: empty tensor -> for(matmul) -> addf(previous_result)
  // CHECK-LABEL: func.func @case11_cascaded_for_loops
  // CHECK: %[[CST_ZERO:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[CST_NONZERO:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[C:.*]] = linalg.fill ins(%[[CST_ZERO]] : f32) outs(%[[EMPTY1]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // Chain 1: for loop 1
  // CHECK: %[[EMPTY_FOR1:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO_FOR1:.*]] = linalg.fill ins(%[[CST_ZERO]] : f32) outs(%[[EMPTY_FOR1]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[FOR1:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[ZERO_FOR1]]) -> (tensor<32x32xf32>) {
  // CHECK: %[[MM1:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%{{.*}} : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: scf.yield %[[MM1]] : tensor<32x32xf32>
  // CHECK: }
  // CHECK: %[[ADD1:.*]] = arith.addf %[[FOR1]], %[[C]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // Chain 1: for loop 2
  // CHECK: %[[EMPTY_FOR2:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO_FOR2:.*]] = linalg.fill ins(%[[CST_ZERO]] : f32) outs(%[[EMPTY_FOR2]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[FOR2:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[ZERO_FOR2]]) -> (tensor<32x32xf32>) {
  // CHECK: %[[MM2:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%{{.*}} : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: scf.yield %[[MM2]] : tensor<32x32xf32>
  // CHECK: }
  // CHECK: %[[ADD2:.*]] = arith.addf %[[FOR2]], %[[ADD1]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // Chain 1: for loop 3
  // CHECK: %[[EMPTY_FOR3:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO_FOR3:.*]] = linalg.fill ins(%[[CST_ZERO]] : f32) outs(%[[EMPTY_FOR3]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[FOR3:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[ZERO_FOR3]]) -> (tensor<32x32xf32>) {
  // CHECK: %[[MM3:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%{{.*}} : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: scf.yield %[[MM3]] : tensor<32x32xf32>
  // CHECK: }
  // CHECK: %[[ADD3:.*]] = arith.addf %[[FOR3]], %[[ADD2]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // Chain 2: for loop 4 (non-zero init)
  // CHECK: %[[EMPTY_D:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[D:.*]] = linalg.fill ins(%[[CST_NONZERO]] : f32) outs(%[[EMPTY_D]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[EMPTY_FOR4:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO_FOR4:.*]] = linalg.fill ins(%[[CST_ZERO]] : f32) outs(%[[EMPTY_FOR4]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[FOR4:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[ZERO_FOR4]]) -> (tensor<32x32xf32>) {
  // CHECK: %[[MM4:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%{{.*}} : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: scf.yield %[[MM4]] : tensor<32x32xf32>
  // CHECK: }
  // CHECK: %[[ADD4:.*]] = arith.addf %[[FOR4]], %[[D]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // Chain 2: for loop 5
  // CHECK: %[[EMPTY_FOR5:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO_FOR5:.*]] = linalg.fill ins(%[[CST_ZERO]] : f32) outs(%[[EMPTY_FOR5]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[FOR5:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[ZERO_FOR5]]) -> (tensor<32x32xf32>) {
  // CHECK: %[[MM5:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%{{.*}} : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: scf.yield %[[MM5]] : tensor<32x32xf32>
  // CHECK: }
  // CHECK: %[[ADD5:.*]] = arith.addf %[[FOR5]], %[[ADD4]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // Chain 2: for loop 6
  // CHECK: %[[EMPTY_FOR6:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO_FOR6:.*]] = linalg.fill ins(%[[CST_ZERO]] : f32) outs(%[[EMPTY_FOR6]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[FOR6:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[ZERO_FOR6]]) -> (tensor<32x32xf32>) {
  // CHECK: %[[MM6:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%{{.*}} : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: scf.yield %[[MM6]] : tensor<32x32xf32>
  // CHECK: }
  // CHECK: %[[ADD6:.*]] = arith.addf %[[FOR6]], %[[ADD5]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // Final combination
  // CHECK: %[[RESULT:.*]] = arith.addf %[[ADD3]], %[[ADD6]] : tensor<32x32xf32>
  // CHECK: return %[[RESULT]] : tensor<32x32xf32>
  func.func @case11_cascaded_for_loops(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>) -> tensor<32x32xf32> {
    %c0_idx = arith.constant 0 : index
    %c10_idx = arith.constant 10 : index
    %c1_idx = arith.constant 1 : index

    %cst_zero = arith.constant 0.0 : f32
    %cst_nonzero = arith.constant 1.0 : f32
    %empty1 = tensor.empty() : tensor<32x32xf32>

    // Chain 1: zero initialization, each matmul in a for loop
    %c = linalg.fill ins(%cst_zero : f32) outs(%empty1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %for1_result = scf.for %i1 = %c0_idx to %c10_idx step %c1_idx iter_args(%bias1 = %c) -> (tensor<32x32xf32>) {
      %mm1 = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias1 : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %mm1 : tensor<32x32xf32>
    }
    %for2_result = scf.for %i2 = %c0_idx to %c10_idx step %c1_idx iter_args(%bias2 = %for1_result) -> (tensor<32x32xf32>) {
      %mm2 = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias2 : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %mm2 : tensor<32x32xf32>
    }
    %for3_result = scf.for %i3 = %c0_idx to %c10_idx step %c1_idx iter_args(%bias3 = %for2_result) -> (tensor<32x32xf32>) {
      %mm3 = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias3 : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %mm3 : tensor<32x32xf32>
    }

    // Chain 2: non-zero initialization, each matmul in a for loop
    %empty2 = tensor.empty() : tensor<32x32xf32>
    %d = linalg.fill ins(%cst_nonzero : f32) outs(%empty2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %for4_result = scf.for %i4 = %c0_idx to %c10_idx step %c1_idx iter_args(%bias4 = %d) -> (tensor<32x32xf32>) {
      %mm4 = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias4 : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %mm4 : tensor<32x32xf32>
    }
    %for5_result = scf.for %i5 = %c0_idx to %c10_idx step %c1_idx iter_args(%bias5 = %for4_result) -> (tensor<32x32xf32>) {
      %mm5 = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias5 : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %mm5 : tensor<32x32xf32>
    }
    %for6_result = scf.for %i6 = %c0_idx to %c10_idx step %c1_idx iter_args(%bias6 = %for5_result) -> (tensor<32x32xf32>) {
      %mm6 = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias6 : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %mm6 : tensor<32x32xf32>
    }

    // Combine the two chains
    %result = arith.addf %for3_result, %for6_result : tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }

  // Case 12: scf.if containing scf.for with matmul.
  // The then branch has a for loop with matmul, the else branch yields zero initial value directly.
  // Matmul in the for loop is marked as loop_carried_l0c and not split, maintaining the L0C chain.
  // CHECK-LABEL: func.func @case12_if_for
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[EMPTY]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: scf.if %{{.*}} -> (tensor<32x32xf32>) {
  // CHECK: %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[BIAS:.*]] = %[[ZERO]]) -> (tensor<32x32xf32>) {
  // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[BIAS]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK-NOT: arith.addf
  // CHECK: scf.yield %[[MM]] : tensor<32x32xf32>
  // CHECK: }
  // CHECK: scf.yield %[[FOR]] : tensor<32x32xf32>
  // CHECK: } else {
  // CHECK: scf.yield %[[ZERO]] : tensor<32x32xf32>
  // CHECK: }
  func.func @case12_if_for(%cond: i1, %A: tensor<32x64xf32>, %B: tensor<64x32xf32>) -> tensor<32x32xf32> {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x32xf32>
    %zero_init = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>

    %result = scf.if %cond -> (tensor<32x32xf32>) {
      %for_result = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc = %zero_init) -> (tensor<32x32xf32>) {
        %mm = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
        scf.yield %mm : tensor<32x32xf32>
      }
      scf.yield %for_result : tensor<32x32xf32>
    } else {
      scf.yield %zero_init : tensor<32x32xf32>
    }
    return %result : tensor<32x32xf32>
  }

  // Case 13: scf.for loop where the upper bound is a function argument.
  // Since one of the bounds is not constant, scfMayNotExec returns true.
  // The loop has a non-zero initial value, so it is split.
  // After the split, an scf.if checks if the loop executes (ub > lb).
  // If so, it yields the loop result; otherwise, it fills with zero.
  // CHECK-LABEL: func.func @case13_may_not_exec_arg_bound
  // CHECK-SAME: (%[[UB:.*]]: index, %[[A:.*]]: tensor<32x64xf32>, %[[B:.*]]: tensor<64x32xf32>) -> tensor<32x32xf32>
  // CHECK-DAG: %[[CST_ZERO:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: %[[CST_ONE:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[NONZERO_INIT:.*]] = linalg.fill ins(%[[CST_ONE]] : f32) outs(%[[EMPTY1]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[ZERO_INIT:.*]] = linalg.fill ins(%[[CST_ZERO]] : f32) outs(%[[EMPTY2]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[FOR_RESULT:.*]] = scf.for %{{.*}} = %{{.*}} to %[[UB]] step %{{.*}} iter_args(%[[ITER_BIAS:.*]] = %[[ZERO_INIT]]) -> (tensor<32x32xf32>) {
  // CHECK:   %[[MM:.*]] = linalg.matmul {ssbuffer.loop_carried_l0c} ins(%[[A]], %[[B]] : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[ITER_BIAS]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK:   scf.yield %[[MM]] : tensor<32x32xf32>
  // CHECK: }
  // CHECK: %[[COND:.*]] = arith.cmpi sgt, %[[UB]], %{{.*}} : index
  // CHECK: %[[IF_RESULT:.*]] = scf.if %[[COND]] -> (tensor<32x32xf32>) {
  // CHECK:   scf.yield %[[FOR_RESULT]] : tensor<32x32xf32>
  // CHECK: } else {
  // CHECK:   %[[FILL_ELSE:.*]] = linalg.fill ins(%[[CST_ZERO]] : f32) outs(%[[FOR_RESULT]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK:   scf.yield %[[FILL_ELSE]] : tensor<32x32xf32>
  // CHECK: }
  // CHECK: %[[ADD:.*]] = arith.addf %[[IF_RESULT]], %[[NONZERO_INIT]] {ssbuffer.add_from_matmul} : tensor<32x32xf32>
  // CHECK: return %[[ADD]] : tensor<32x32xf32>
  func.func @case13_may_not_exec_arg_bound(%ub: index, %A: tensor<32x64xf32>, %B: tensor<64x32xf32>) -> tensor<32x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<32x32xf32>
    %nonzero_init = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>

    %result = scf.for %iv = %c0 to %ub step %c1 iter_args(%bias = %nonzero_init) -> (tensor<32x32xf32>) {
      %mm = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%bias : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %mm : tensor<32x32xf32>
    }
    return %result : tensor<32x32xf32>
  }
}

