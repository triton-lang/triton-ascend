// RUN: triton-opt %s --ssbuf-split-matmul --verify-each | FileCheck %s
// RUN: triton-opt %s --ssbuf-standardize-op --verify-each > /dev/null

// A/B edges are omitted from grouping, but the intermediate result is still
// observed as A. Removing its bias across the pair would change multiplication.
// CHECK-LABEL: func.func @intermediate_used_as_a_and_c
// CHECK: %[[RAW0:.*]] = linalg.matmul
// CHECK: %[[SUM0:.*]] = arith.addf %[[RAW0]], %arg2
// CHECK: %[[RAW1:.*]] = linalg.matmul {{.*}}ins(%[[SUM0]], %arg1
// CHECK: %[[SUM1:.*]] = arith.addf %[[RAW1]], %[[SUM0]]
// CHECK: return %[[SUM1]]
func.func @intermediate_used_as_a_and_c(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %x = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  %y = linalg.matmul ins(%x, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%x : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %y : tensor<4x4xf32>
}

// A-only dependency is outside the accumulator chain. Both matmuls split
// independently, and the second one observes the first one's restored result.
// CHECK-LABEL: func.func @a_only_dependency
// CHECK: %[[RAW0:.*]] = linalg.matmul
// CHECK: %[[SUM0:.*]] = arith.addf %[[RAW0]], %arg2
// CHECK: %[[RAW1:.*]] = linalg.matmul {{.*}}ins(%[[SUM0]], %arg1
// CHECK: %[[SUM1:.*]] = arith.addf %[[RAW1]], %arg3
// CHECK: return %[[SUM1]]
func.func @a_only_dependency(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>, %d: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %x = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  %y = linalg.matmul ins(%x, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%d : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %y : tensor<4x4xf32>
}

// A loop-carried Phi is observed as A as well as C. Keep the bias addition
// inside the loop; the selected C-only graph must not hide this IR use.
// CHECK-LABEL: func.func @loop_accumulator_used_as_a
// CHECK: %[[LOOP:.*]] = scf.for {{.*}}iter_args(%[[ACC:.*]] = %arg1)
// CHECK: %[[RAW:.*]] = linalg.matmul {{.*}}ins(%[[ACC]], %arg0
// CHECK: %[[SUM:.*]] = arith.addf %[[RAW]], %[[ACC]]
// CHECK: scf.yield %[[SUM]]
// CHECK: return %[[LOOP]]
func.func @loop_accumulator_used_as_a(%b: tensor<4x4xf32>, %c: tensor<4x4xf32>, %n: index) -> tensor<4x4xf32> {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %r = scf.for %i = %lb to %n step %step iter_args(%acc = %c) -> tensor<4x4xf32> {
    %m = linalg.matmul ins(%acc, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.yield %m : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}

// A single entry Phi can merge two distinct external values. The pass must
// check Value uniqueness as well as node uniqueness and split at the matmul.
// CHECK-LABEL: func.func @single_entry_two_initial_values
// CHECK: %[[BIAS:.*]] = scf.if
// CHECK: scf.yield %arg2
// CHECK: } else {
// CHECK: scf.yield %arg3
// CHECK: %[[ZERO:.*]] = linalg.fill
// CHECK: %[[MM:.*]] = linalg.matmul {{.*}}outs(%[[ZERO]]
// CHECK: %[[SUM:.*]] = arith.addf %[[MM]], %[[BIAS]]
// CHECK: return %[[SUM]]
func.func @single_entry_two_initial_values(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>, %d: tensor<4x4xf32>, %go: i1) -> tensor<4x4xf32> {
  %bias = scf.if %go -> tensor<4x4xf32> {
    scf.yield %c : tensor<4x4xf32>
  } else {
    scf.yield %d : tensor<4x4xf32>
  }
  %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%bias : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %m : tensor<4x4xf32>
}

// Sharing a Condition does not make the three matmuls strongly connected.
// The first two are singleton SCCs; only the last matmul has a C backedge.
// CHECK-LABEL: func.func @shared_condition_independent_accumulators
// CHECK: %[[LOOP:.*]] = scf.for
// CHECK: scf.if
// CHECK: %[[MM0:.*]] = linalg.matmul
// CHECK-NOT: memref.store
// CHECK: %[[MM1:.*]] = linalg.matmul
// CHECK-NOT: memref.store
// CHECK: arith.addf %[[MM0]], %[[MM1]]
// CHECK: linalg.matmul {{.*}}ssbuffer.coupled_matmul_and_store
// CHECK: memref.store
// CHECK: memref.load
// CHECK: %[[GUARD:.*]] = scf.if
// CHECK: } else {
// CHECK: scf.yield %[[LOOP]]
// CHECK: arith.addf %[[GUARD]], %arg2
func.func @shared_condition_independent_accumulators(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>, %go: i1, %n: index) -> tensor<4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %r = scf.for %i = %lb to %n step %step iter_args(%acc = %c) -> tensor<4x4xf32> {
    %x = scf.if %go -> tensor<4x4xf32> {
      %m0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
      %m1 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
      %sum = arith.addf %m0, %m1 : tensor<4x4xf32>
      %m2 = linalg.matmul ins(%sum, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
      scf.yield %m2 : tensor<4x4xf32>
    } else {
      scf.yield %acc : tensor<4x4xf32>
    }
    scf.yield %x : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}
