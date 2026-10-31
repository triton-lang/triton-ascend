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

  // The subview slot must not introduce dependencies between disjoint sources.
  // CHECK-LABEL: Analyzing op func.func @disjoint_pointer_cast_through_subview
  // CHECK: Analyzing op %[[VIEW_DISJOINT_RHS:[A-Za-z0-9_]+]] = hivm.hir.pointer_cast(%c16_i64)
  // CHECK: Analyzing op memref.store {{.*}}, %[[VIEW_DISJOINT_RHS]][{{.*}}]
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] %[[VIEW_DISJOINT_RHS]] = hivm.hir.pointer_cast(%c16_i64)
  // CHECK-NEXT: [memory-effects-tracker] Analyzing op func.return
  func.func @disjoint_pointer_cast_through_subview() {
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c0 = arith.constant 0 : index
    %value = arith.constant 1 : i32
    %lhs = hivm.hir.pointer_cast(%c0_i64) : memref<4xi32, #hivm.address_space<ub>>
    %rhs = hivm.hir.pointer_cast(%c16_i64) : memref<4xi32, #hivm.address_space<ub>>
    %view = memref.subview %lhs[0] [4] [1]
      : memref<4xi32, #hivm.address_space<ub>> to memref<4xi32, strided<[1]>, #hivm.address_space<ub>>
    memref.store %value, %view[%c0] : memref<4xi32, strided<[1]>, #hivm.address_space<ub>>
    memref.store %value, %rhs[%c0] : memref<4xi32, #hivm.address_space<ub>>
    return
  }

  // Overlapping source ranges retain the dependency through a subview slot.
  // CHECK-LABEL: Analyzing op func.func @overlapping_pointer_cast_through_subview
  // CHECK: Analyzing op memref.store %c1_i32, %[[VIEW_OVERLAP_LHS:[A-Za-z0-9_]+]][%c0]
  // CHECK: Analyzing op memref.store %c1_i32, %{{[A-Za-z0-9_]+}}[%c0]
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] memref.store %c1_i32, %[[VIEW_OVERLAP_LHS]][%c0]
  func.func @overlapping_pointer_cast_through_subview() {
    %c0_i64 = arith.constant 0 : i64
    %c8_i64 = arith.constant 8 : i64
    %c0 = arith.constant 0 : index
    %value = arith.constant 1 : i32
    %lhs = hivm.hir.pointer_cast(%c0_i64) : memref<4xi32, #hivm.address_space<ub>>
    %rhs = hivm.hir.pointer_cast(%c8_i64) : memref<4xi32, #hivm.address_space<ub>>
    %view = memref.subview %lhs[0] [4] [1]
      : memref<4xi32, #hivm.address_space<ub>> to memref<4xi32, strided<[1]>, #hivm.address_space<ub>>
    memref.store %value, %view[%c0] : memref<4xi32, strided<[1]>, #hivm.address_space<ub>>
    memref.store %value, %rhs[%c0] : memref<4xi32, #hivm.address_space<ub>>
    return
  }

  // Different views of one function argument must retain the write dependency.
  // CHECK-LABEL: Analyzing op func.func @same_argument_overlapping_subviews
  // CHECK: Analyzing op memref.store %c1_i32, %[[ARG_VIEW_LHS:[A-Za-z0-9_]+]][%c0]
  // CHECK: Analyzing op memref.store %c1_i32, %{{[A-Za-z0-9_]+}}[%c0]
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] memref.store %c1_i32, %[[ARG_VIEW_LHS]][%c0]
  func.func @same_argument_overlapping_subviews(%arg: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %value = arith.constant 1 : i32
    %lhs = memref.subview %arg[0] [4] [1]
      : memref<8xi32> to memref<4xi32, strided<[1]>>
    %rhs = memref.subview %arg[2] [4] [1]
      : memref<8xi32> to memref<4xi32, strided<[1], offset: 2>>
    memref.store %value, %lhs[%c0] : memref<4xi32, strided<[1]>>
    memref.store %value, %rhs[%c0] : memref<4xi32, strided<[1], offset: 2>>
    return
  }

  // A view preserves its source state without becoming a memory producer.
  // CHECK-LABEL: Analyzing op func.func @overlapping_pointer_cast_view_slot
  // CHECK: Analyzing op %[[SLOT_LHS:[A-Za-z0-9_]+]] = hivm.hir.pointer_cast(%c0_i64)
  // CHECK: Analyzing op %[[SLOT_RHS:[A-Za-z0-9_]+]] = hivm.hir.pointer_cast(%c8_i64)
  // CHECK: Analyzing op %{{[A-Za-z0-9_]+}} = memref.subview
  // CHECK: Analyzing op memref.store
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] %[[SLOT_RHS]] = hivm.hir.pointer_cast(%c8_i64)
  // CHECK-NEXT: [memory-effects-tracker] %[[SLOT_LHS]] = hivm.hir.pointer_cast(%c0_i64)
  // CHECK-NEXT: [memory-effects-tracker] Analyzing op func.return
  func.func @overlapping_pointer_cast_view_slot() {
    %c0_i64 = arith.constant 0 : i64
    %c8_i64 = arith.constant 8 : i64
    %c0 = arith.constant 0 : index
    %value = arith.constant 1 : i32
    %lhs = hivm.hir.pointer_cast(%c0_i64) : memref<4xi32, #hivm.address_space<ub>>
    %rhs = hivm.hir.pointer_cast(%c8_i64) : memref<4xi32, #hivm.address_space<ub>>
    %view = memref.subview %lhs[0] [4] [1]
      : memref<4xi32, #hivm.address_space<ub>> to memref<4xi32, strided<[1]>, #hivm.address_space<ub>>
    memref.store %value, %rhs[%c0] : memref<4xi32, #hivm.address_space<ub>>
    return
  }

  // A view of an unwritten argument must not create a memory dependency.
  // CHECK-LABEL: Analyzing op func.func @same_argument_view_slot
  // CHECK: Analyzing op %[[ARG_SLOT_VIEW:[A-Za-z0-9_]+]] = memref.subview
  // CHECK: Analyzing op memref.store
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] Analyzing op func.return
  func.func @same_argument_view_slot(%arg: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %value = arith.constant 1 : i32
    %view = memref.subview %arg[0] [4] [1]
      : memref<8xi32> to memref<4xi32, strided<[1]>>
    memref.store %value, %arg[%c0] : memref<8xi32>
    return
  }

}
