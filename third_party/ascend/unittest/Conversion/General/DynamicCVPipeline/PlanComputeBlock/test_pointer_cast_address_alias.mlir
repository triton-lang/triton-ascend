// RUN: triton-opt --debug-only=memory-effects-tracker --plan-compute-block %s 2>&1 | FileCheck %s

module {
  // The second store only depends on its own allocation: [0, 16) and [16, 32)
  // are adjacent but do not overlap.
  // CHECK-LABEL: Analyzing op func.func @constant_disjoint_pointer_casts
  // CHECK: Analyzing op %[[DISJOINT_RHS:[A-Za-z0-9_]+]] = hivm.hir.pointer_cast(%c16_i64)
  // CHECK: Analyzing op memref.store {{.*}}, %[[DISJOINT_RHS]][{{.*}}]
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] %[[DISJOINT_RHS]] = hivm.hir.pointer_cast(%c16_i64)
  func.func @constant_disjoint_pointer_casts() {
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c0 = arith.constant 0 : index
    %value = arith.constant 1 : i32
    %lhs = hivm.hir.pointer_cast(%c0_i64) : memref<4xi32, #hivm.address_space<ub>>
    %rhs = hivm.hir.pointer_cast(%c16_i64) : memref<4xi32, #hivm.address_space<ub>>
    memref.store %value, %lhs[%c0] : memref<4xi32, #hivm.address_space<ub>>
    memref.store %value, %rhs[%c0] : memref<4xi32, #hivm.address_space<ub>>
    return
  }

  // [0, 16) and [8, 24) overlap, so the second store depends on the first.
  // CHECK-LABEL: Analyzing op func.func @constant_overlapping_pointer_casts
  // CHECK: Analyzing op memref.store %c1_i32, %[[OVERLAP_LHS:[A-Za-z0-9_]+]][%c0]
  // CHECK: Analyzing op memref.store %c1_i32, %{{[A-Za-z0-9_]+}}[%c0]
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] memref.store %c1_i32, %[[OVERLAP_LHS]][%c0]
  func.func @constant_overlapping_pointer_casts() {
    %c0_i64 = arith.constant 0 : i64
    %c8_i64 = arith.constant 8 : i64
    %c0 = arith.constant 0 : index
    %value = arith.constant 1 : i32
    %lhs = hivm.hir.pointer_cast(%c0_i64) : memref<4xi32, #hivm.address_space<ub>>
    %rhs = hivm.hir.pointer_cast(%c8_i64) : memref<4xi32, #hivm.address_space<ub>>
    memref.store %value, %lhs[%c0] : memref<4xi32, #hivm.address_space<ub>>
    memref.store %value, %rhs[%c0] : memref<4xi32, #hivm.address_space<ub>>
    return
  }

  // A non-arith.constant address conservatively aliases the constant range.
  // CHECK-LABEL: Analyzing op func.func @variable_pointer_cast_may_alias
  // CHECK: Analyzing op memref.store %c1_i32, %[[VARIABLE_LHS:[A-Za-z0-9_]+]][%c0]
  // CHECK: Analyzing op memref.store %c1_i32, %{{[A-Za-z0-9_]+}}[%c0]
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] memref.store %c1_i32, %[[VARIABLE_LHS]][%c0]
  func.func @variable_pointer_cast_may_alias(%address: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %value = arith.constant 1 : i32
    %lhs = hivm.hir.pointer_cast(%c0_i64) : memref<4xi32, #hivm.address_space<ub>>
    %rhs = hivm.hir.pointer_cast(%address) : memref<4xi32, #hivm.address_space<ub>>
    memref.store %value, %lhs[%c0] : memref<4xi32, #hivm.address_space<ub>>
    memref.store %value, %rhs[%c0] : memref<4xi32, #hivm.address_space<ub>>
    return
  }

  // memref<8xi1> occupies ceil(8 / 8) = 1 byte. The byte addresses 0 and 1
  // therefore map to the adjacent, non-overlapping ranges [0, 1) and [1, 2).
  // CHECK-LABEL: Analyzing op func.func @total_bit_size_rounds_up_to_bytes
  // CHECK: Analyzing op %[[BIT_RHS:[A-Za-z0-9_]+]] = hivm.hir.pointer_cast(%c1_i64)
  // CHECK: Analyzing op memref.store {{.*}}, %[[BIT_RHS]][{{.*}}]
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] %[[BIT_RHS]] = hivm.hir.pointer_cast(%c1_i64)
  func.func @total_bit_size_rounds_up_to_bytes() {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %value = arith.constant true
    %lhs = hivm.hir.pointer_cast(%c0_i64) : memref<8xi1, #hivm.address_space<ub>>
    %rhs = hivm.hir.pointer_cast(%c1_i64) : memref<8xi1, #hivm.address_space<ub>>
    memref.store %value, %lhs[%c0] : memref<8xi1, #hivm.address_space<ub>>
    memref.store %value, %rhs[%c0] : memref<8xi1, #hivm.address_space<ub>>
    return
  }

}
