// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s --check-prefixes=CHECK,CANON
// RUN: triton-opt --canonicalize --triton-to-linalg --split-input-file %s | FileCheck %s --check-prefix=CANON

// A changing tensor must keep the current iteration's predicate, rather than
// deriving a slice length from the initial range.
// CHECK-LABEL: func.func @changing_for
// CHECK: scf.for {{.*}} iter_args(%[[OFFSETS:[a-zA-Z0-9_]+]] =
// CHECK: %[[MASK:.*]] = linalg.generic {{.*}} ins(%[[OFFSETS]],
// CHECK: arith.cmpi slt,
// CHECK: linalg.generic {{.*}} ins(%[[MASK]],
// CHECK: arith.select {{.*}} : i32
func.func @changing_for(%n: i32, %steps: index, %yes: tensor<32xi32>, %no: tensor<32xi32>) -> tensor<32xi32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %stride = arith.constant dense<32> : tensor<32xi32>
  %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %limit = tt.splat %n : i32 -> tensor<32xi32>
  %result:2 = scf.for %i = %zero to %steps step %one iter_args(%offsets = %range, %acc = %no) -> (tensor<32xi32>, tensor<32xi32>) {
    %mask = arith.cmpi slt, %offsets, %limit : tensor<32xi32>
    %selected = arith.select %mask, %yes, %acc : tensor<32xi1>, tensor<32xi32>
    %next = arith.addi %offsets, %stride : tensor<32xi32>
    scf.yield %next, %selected : tensor<32xi32>, tensor<32xi32>
  }
  return %result#1 : tensor<32xi32>
}

// -----

// Canonicalization removes the unchanged carrier before select analysis, so
// the rectangular optimization needs no invariance analysis in MaskState.
// TritonToLinalg also canonicalizes internally; both RUN lines retain this path.
// CANON-LABEL: func.func @invariant_for
// CANON-NOT: arith.select
// CANON: tensor.extract_slice
// CANON: tensor.insert_slice
// CANON-NOT: arith.select
// CANON: return
func.func @invariant_for(%n: i32, %steps: index, %yes: tensor<32xi32>, %no: tensor<32xi32>) -> tensor<32xi32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %limit = tt.splat %n : i32 -> tensor<32xi32>
  %result:2 = scf.for %i = %zero to %steps step %one iter_args(%offsets = %range, %acc = %no) -> (tensor<32xi32>, tensor<32xi32>) {
    %mask = arith.cmpi slt, %offsets, %limit : tensor<32xi32>
    %selected = arith.select %mask, %yes, %acc : tensor<32xi1>, tensor<32xi32>
    scf.yield %offsets, %selected : tensor<32xi32>, tensor<32xi32>
  }
  return %result#1 : tensor<32xi32>
}

// -----

// The inner carrier is unchanged, but its init is a changing outer carrier.
// CHECK-LABEL: func.func @nested_for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.cmpi slt,
// CHECK: arith.select {{.*}} : i32
func.func @nested_for(%n: i32, %steps: index, %yes: tensor<32xi32>, %no: tensor<32xi32>) -> tensor<32xi32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %stride = arith.constant dense<32> : tensor<32xi32>
  %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %limit = tt.splat %n : i32 -> tensor<32xi32>
  %result:2 = scf.for %i = %zero to %steps step %one iter_args(%offsets = %range, %acc = %no) -> (tensor<32xi32>, tensor<32xi32>) {
    %inner:2 = scf.for %j = %zero to %steps step %one iter_args(%lanes = %offsets, %value = %acc) -> (tensor<32xi32>, tensor<32xi32>) {
      %mask = arith.cmpi slt, %lanes, %limit : tensor<32xi32>
      %selected = arith.select %mask, %yes, %value : tensor<32xi1>, tensor<32xi32>
      scf.yield %lanes, %selected : tensor<32xi32>, tensor<32xi32>
    }
    %next = arith.addi %offsets, %stride : tensor<32xi32>
    scf.yield %next, %inner#1 : tensor<32xi32>, tensor<32xi32>
  }
  return %result#1 : tensor<32xi32>
}

// -----

// A while before-region argument also changes on the backedge. The forwarded
// after-region argument is not directly tied to the loop init either.
// CHECK-LABEL: func.func @changing_while
// CHECK: scf.while
// CHECK: arith.cmpi slt,
// CHECK: arith.select {{.*}} : i32
// CHECK: scf.condition
// CHECK: arith.cmpi slt,
// CHECK: arith.select {{.*}} : i32
func.func @changing_while(%n: i32, %steps: i32, %yes: tensor<32xi32>, %no: tensor<32xi32>) -> tensor<32xi32> {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %stride = arith.constant dense<32> : tensor<32xi32>
  %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %limit = tt.splat %n : i32 -> tensor<32xi32>
  %result:3 = scf.while (%i = %zero, %offsets = %range, %acc = %no) : (i32, tensor<32xi32>, tensor<32xi32>) -> (i32, tensor<32xi32>, tensor<32xi32>) {
    %mask = arith.cmpi slt, %offsets, %limit : tensor<32xi32>
    %selected = arith.select %mask, %yes, %acc : tensor<32xi1>, tensor<32xi32>
    %next = arith.addi %offsets, %stride : tensor<32xi32>
    %continue = arith.cmpi slt, %i, %steps : i32
    scf.condition(%continue) %i, %next, %selected : i32, tensor<32xi32>, tensor<32xi32>
  } do {
  ^bb0(%i: i32, %offsets: tensor<32xi32>, %acc: tensor<32xi32>):
    %mask = arith.cmpi slt, %offsets, %limit : tensor<32xi32>
    %selected = arith.select %mask, %acc, %no : tensor<32xi1>, tensor<32xi32>
    %next = arith.addi %i, %one : i32
    scf.yield %next, %offsets, %selected : i32, tensor<32xi32>, tensor<32xi32>
  }
  return %result#2 : tensor<32xi32>
}
