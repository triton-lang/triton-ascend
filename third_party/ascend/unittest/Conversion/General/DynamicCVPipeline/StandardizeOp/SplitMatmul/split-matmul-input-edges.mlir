// RUN: triton-opt %s --ssbuffer-split-matmul --verify-each | FileCheck %s
// RUN: triton-opt %s --ssbuf-standardize-op --verify-each > /dev/null

// Supply the MNE handoff normally emitted by PatternMatchRewritePass so the
// standalone split pass runs without changing its production entry condition.

// A/B edges are omitted from grouping, but the intermediate result is still
// observed as A. Removing its bias across the pair would change multiplication.
// CHECK-LABEL: func.func @intermediate_used_as_a_and_c
// CHECK: %[[RAW0:.*]] = linalg.matmul
// CHECK: %[[SUM0:.*]] = arith.addf %[[RAW0]], %arg2
// CHECK: %[[RAW1:.*]] = linalg.matmul {{.*}}ins(%[[SUM0]], %arg1
// CHECK: %[[SUM1:.*]] = arith.addf %[[RAW1]], %[[SUM0]]
// CHECK: return %[[SUM1]]
func.func @intermediate_used_as_a_and_c(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %x = linalg.matmul {ssbuffer.may_not_exec} ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  %y = linalg.matmul {ssbuffer.may_not_exec} ins(%x, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%x : tensor<4x4xf32>) -> tensor<4x4xf32>
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
  %x = linalg.matmul {ssbuffer.may_not_exec} ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  %y = linalg.matmul {ssbuffer.may_not_exec} ins(%x, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%d : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %y : tensor<4x4xf32>
}

// An accumulator observed as A must retain the original loop recurrence.
// Split only the matmul C operand and restore its bias inside the loop.
// CHECK-LABEL: func.func @loop_accumulator_used_as_a
// CHECK: %[[LOOP:[a-zA-Z0-9_]+]] = scf.for {{.*}}iter_args(%[[ACC:[a-zA-Z0-9_]+]] = %arg1)
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = linalg.fill
// CHECK: %[[RAW:[a-zA-Z0-9_]+]] = linalg.matmul {{.*}}ins(%[[ACC]], %arg0
// CHECK-SAME: outs(%[[ZERO]]
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addf %[[RAW]], %[[ACC]]
// CHECK: scf.yield %[[SUM]]
// CHECK-NOT: memref.load
// CHECK-NOT: arith.addf
// CHECK: return %[[LOOP]]

func.func @loop_accumulator_used_as_a(%b: tensor<4x4xf32>, %c: tensor<4x4xf32>, %n: index) -> tensor<4x4xf32> {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %r = scf.for %i = %lb to %n step %step iter_args(%acc = %c) -> tensor<4x4xf32> {
    %m = linalg.matmul {ssbuffer.may_not_exec} ins(%acc, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
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
  %m = linalg.matmul {ssbuffer.may_not_exec} ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%bias : tensor<4x4xf32>) -> tensor<4x4xf32>
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
      %m0 = linalg.matmul {ssbuffer.may_not_exec} ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
      %m1 = linalg.matmul {ssbuffer.may_not_exec} ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
      %sum = arith.addf %m0, %m1 : tensor<4x4xf32>
      %m2 = linalg.matmul {ssbuffer.may_not_exec} ins(%sum, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
      scf.yield %m2 : tensor<4x4xf32>
    } else {
      scf.yield %acc : tensor<4x4xf32>
    }
    scf.yield %x : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}

// The RHS and accumulator alias. Splitting must replace only the accumulator.
// CHECK-LABEL: func.func @split_matmul_with_aliased_rhs_and_accumulator(
// CHECK-SAME: %[[A:[^:]+]]: tensor<32x32xf32>, %[[B:[^:]+]]: tensor<32x32xf32>
// CHECK: %[[ZERO:.*]] = linalg.fill
// CHECK: %[[MATMUL:.*]] = linalg.matmul {{.*}}ins(%[[A]], %[[B]] : tensor<32x32xf32>, tensor<32x32xf32>) outs(%[[ZERO]] : tensor<32x32xf32>)
// CHECK: %[[RESULT:.*]] = arith.addf %[[MATMUL]], %[[B]] {{.*}}ssbuffer.add_from_matmul
// CHECK: return %[[RESULT]] : tensor<32x32xf32>
func.func @split_matmul_with_aliased_rhs_and_accumulator(
    %a: tensor<32x32xf32>, %b: tensor<32x32xf32>)
    -> tensor<32x32xf32> {
  %matmul = linalg.matmul {ssbuffer.may_not_exec, input_precision = "ieee"}
      ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>)
      outs(%b : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %matmul : tensor<32x32xf32>
}

// A/B producers must not add extra entries to the accumulator SCC. The bias
// is split around the complete loop, rather than once per matmul iteration.
// CHECK-LABEL: func.func @filter_ab_edges
// CHECK: %[[C:.*]] = tensor.cast %arg2
// CHECK: %[[ZERO:.*]] = linalg.fill
// CHECK: %[[LOOP:.*]] = scf.for {{.*}}iter_args(%[[ACC:.*]] = %[[ZERO]])
// CHECK: linalg.matmul {{.*}}outs(%[[ACC]]
// CHECK-NOT: arith.addf
// CHECK: scf.yield
// CHECK: %[[SUM:.*]] = arith.addf %[[LOOP]], %[[C]]
// CHECK: return %[[SUM]]
func.func @filter_ab_edges(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %a0 = tensor.cast %a : tensor<4x4xf32> to tensor<4x4xf32>
  %b0 = tensor.cast %b : tensor<4x4xf32> to tensor<4x4xf32>
  %c0 = tensor.cast %c : tensor<4x4xf32> to tensor<4x4xf32>
  %lb = arith.constant 0 : index
  %ub = arith.constant 2 : index
  %step = arith.constant 1 : index
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %c0) -> tensor<4x4xf32> {
    %m = linalg.matmul {ssbuffer.may_not_exec} ins(%a0, %b0 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.yield %m : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}

// Filtering an operand site must not remove another use of the same SSA value.
// With B == C, only the accumulator operand is replaced by zero.
// CHECK-LABEL: func.func @aliased_b_and_c
// CHECK: %[[B:.*]] = tensor.cast %arg1
// CHECK: %[[ZERO:.*]] = linalg.fill
// CHECK: %[[MM:.*]] = linalg.matmul {{.*}}ins(%arg0, %[[B]] {{.*}}outs(%[[ZERO]]
// CHECK: %[[SUM:.*]] = arith.addf %[[MM]], %[[B]]
// CHECK: return %[[SUM]]
func.func @aliased_b_and_c(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %b0 = tensor.cast %b : tensor<4x4xf32> to tensor<4x4xf32>
  %m = linalg.matmul {ssbuffer.may_not_exec} ins(%a, %b0 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%b0 : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %m : tensor<4x4xf32>
}
