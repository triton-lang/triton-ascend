// RUN: triton-opt %s --ssbuf-split-matmul --verify-each | FileCheck %s
// RUN: triton-opt %s --ssbuf-standardize-op --verify-each > /dev/null

// The same C entering two different component-side Phi nodes is two entries.
// Reject the maximal component for residency and preserve both original paths.
// CHECK-LABEL: func.func @reset_to_initial_c
// CHECK: %[[LOOP:.*]] = scf.for {{.*}}iter_args(%[[ACC:.*]] = %arg2)
// CHECK: scf.if
// CHECK: %[[ZERO:.*]] = linalg.fill
// CHECK: %[[MM:.*]] = linalg.matmul {{.*}}outs(%[[ZERO]]
// CHECK: %[[SUM:.*]] = arith.addf %[[MM]], %[[ACC]]
// CHECK: scf.yield %[[SUM]]
// CHECK: } else {
// CHECK: scf.yield %arg2
// CHECK-NOT: memref.load
// CHECK: return %[[LOOP]]
func.func @reset_to_initial_c(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>, %go: i1, %n: index) -> tensor<4x4xf32> {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %r = scf.for %i = %lb to %n step %step iter_args(%acc = %c) -> tensor<4x4xf32> {
    %x = scf.if %go -> tensor<4x4xf32> {
      %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
      scf.yield %m : tensor<4x4xf32>
    } else {
      scf.yield %c : tensor<4x4xf32>
    }
    scf.yield %x : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}

// The predicate Phi is not strongly connected to the tensor accumulator.
// Only the C ring belongs to this SCC; its before index 0 maps to result 1.
// CHECK-LABEL: func.func @while_permuted_result
// CHECK: %[[LOOP:.*]]:2 = scf.while
// CHECK: scf.condition
// CHECK: linalg.matmul
// CHECK: memref.store
// CHECK: memref.load
// CHECK: %[[GUARD:.*]] = scf.if
// CHECK: } else {
// CHECK: scf.yield %[[LOOP]]#1
// CHECK: arith.addf %[[GUARD]], %arg2

func.func @while_permuted_result(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>, %go: i1) -> tensor<4x4xf32> {
  %r:2 = scf.while (%acc = %c, %p = %go) : (tensor<4x4xf32>, i1) -> (i1, tensor<4x4xf32>) {
    scf.condition(%p) %p, %acc : i1, tensor<4x4xf32>
  } do {
  ^bb0(%again: i1, %arg: tensor<4x4xf32>):
    %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%arg : tensor<4x4xf32>) -> tensor<4x4xf32>
    %stop = arith.constant false
    scf.yield %m, %stop : tensor<4x4xf32>, i1
  }
  return %r#1 : tensor<4x4xf32>
}

// One before Phi, but two after arguments/results. The external predicate
// contributes only a Control edge, so it is not a second entry. The tensor
// output must be while result 1, independent of its before argument index 0.
// CHECK-LABEL: func.func @while_single_entry_permuted_result
// CHECK: %[[ZERO:.*]] = linalg.fill
// CHECK: %[[LOOP:.*]]:2 = scf.while (%{{.*}} = %[[ZERO]])
// CHECK: scf.condition
// CHECK: linalg.matmul
// CHECK: memref.load
// CHECK: %[[GUARD:.*]] = scf.if
// CHECK: scf.yield %[[ZERO]]
// CHECK: } else {
// CHECK: scf.yield %[[LOOP]]#1
// CHECK: arith.addf %[[GUARD]], %arg2
func.func @while_single_entry_permuted_result(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>, %go: i1) -> tensor<4x4xf32> {
  %r:2 = scf.while (%acc = %c) : (tensor<4x4xf32>) -> (i1, tensor<4x4xf32>) {
    scf.condition(%go) %go, %acc : i1, tensor<4x4xf32>
  } do {
  ^bb0(%again: i1, %arg: tensor<4x4xf32>):
    %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%arg : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.yield %m : tensor<4x4xf32>
  }
  return %r#1 : tensor<4x4xf32>
}
