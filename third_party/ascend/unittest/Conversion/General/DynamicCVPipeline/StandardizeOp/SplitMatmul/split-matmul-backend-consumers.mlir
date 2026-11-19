// RUN: triton-opt %s --ssbuffer-split-matmul --verify-each | FileCheck %s
// RUN: triton-opt %s --ssbuf-standardize-op --verify-each > /dev/null

// Supply the MNE handoff normally emitted by PatternMatchRewritePass so the
// standalone split pass runs without changing its production entry condition.

func.func private @observe(tensor<4x4xf32>)

// CHECK-LABEL: func.func @internal_observer(
// CHECK: scf.for {{.*}} iter_args(%[[ACC:[a-zA-Z0-9_]+]] = %arg2)
// CHECK: call @observe(%[[ACC]])
// CHECK: %[[MM:[a-zA-Z0-9_]+]] = linalg.matmul
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addf %[[MM]], %[[ACC]]
// CHECK: scf.yield %[[SUM]]
func.func @internal_observer(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c) -> (tensor<4x4xf32>) {
    func.call @observe(%acc) : (tensor<4x4xf32>) -> ()
    %m = linalg.matmul {ssbuffer.may_not_exec} ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.yield %m : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @accumulator_as_b(
// CHECK: scf.for {{.*}} iter_args(%[[ACC:[a-zA-Z0-9_]+]] = %arg2)
// CHECK: %[[MM:[a-zA-Z0-9_]+]] = linalg.matmul
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addf %[[MM]], %[[ACC]]
// CHECK: scf.yield %[[SUM]]
func.func @accumulator_as_b(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c) -> (tensor<4x4xf32>) {
    %m = linalg.matmul {ssbuffer.may_not_exec} ins(%a, %acc : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.yield %m : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}

// Final output consumers do not observe an intermediate accumulator.
// CHECK-LABEL: func.func @final_output_consumers(
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = linalg.fill
// CHECK: %[[LOOP:[a-zA-Z0-9_]+]] = scf.for {{.*}} iter_args(%[[ACC:[a-zA-Z0-9_]+]] = %[[ZERO]])
// CHECK: linalg.matmul {{.*}}outs(%[[ACC]]
// CHECK-NOT: arith.addf
// CHECK: scf.yield
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addf %[[LOOP]], %arg2
// CHECK: call @observe(%[[SUM]])
// CHECK: return %[[SUM]]
func.func @final_output_consumers(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c) -> (tensor<4x4xf32>) {
    %m = linalg.matmul {ssbuffer.may_not_exec} ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.yield %m : tensor<4x4xf32>
  }
  func.call @observe(%r) : (tensor<4x4xf32>) -> ()
  return %r : tensor<4x4xf32>
}

// The observed outer argument is not a direct matmul C operand.
// CHECK-LABEL: func.func @outer_accumulator_observer(
// CHECK: scf.for {{.*}} iter_args(%[[OUTER:[a-zA-Z0-9_]+]] = %arg2)
// CHECK: call @observe(%[[OUTER]])
// CHECK: scf.for {{.*}} iter_args(%[[INNER:[a-zA-Z0-9_]+]] = %[[OUTER]])
// CHECK: %[[MM:[a-zA-Z0-9_]+]] = linalg.matmul
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addf %[[MM]], %[[INNER]]
// CHECK: scf.yield %[[SUM]]
func.func @outer_accumulator_observer(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%outer = %c) -> (tensor<4x4xf32>) {
    func.call @observe(%outer) : (tensor<4x4xf32>) -> ()
    %inner_result = scf.for %j = %c0 to %c4 step %c1 iter_args(%inner = %outer) -> (tensor<4x4xf32>) {
      %m = linalg.matmul {ssbuffer.may_not_exec} ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%inner : tensor<4x4xf32>) -> tensor<4x4xf32>
      scf.yield %m : tensor<4x4xf32>
    }
    scf.yield %inner_result : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}
