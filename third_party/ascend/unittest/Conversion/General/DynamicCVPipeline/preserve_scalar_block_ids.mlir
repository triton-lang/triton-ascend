// RUN: triton-opt %s --preserve-control-attrs-canonicalize --verify-each | FileCheck %s

// A pointer view's inclusive reach becomes its capacity by adding one.
// Combining the constants must preserve the block ID checked by CloneOps.
// CHECK-LABEL: func.func @pointer_capacity
// CHECK-SAME: %[[OFFSET:.*]]: index, %[[ADDRESS:.*]]: i64
// CHECK: %[[EXTENT:.*]] = arith.constant 248032 : index
// CHECK: %[[END:.*]] = arith.addi %[[OFFSET]], %[[EXTENT]] {ssbuffer.block_id = 19 : i32} : index
// CHECK: %[[CAPACITY:.*]] = arith.maxsi %[[END]], %{{.*}} {ssbuffer.block_id = 19 : i32} : index
// CHECK: %[[POINTER:.*]] = hivm.hir.pointer_cast(%[[ADDRESS]]) [%[[CAPACITY]]] {ssbuffer.block_id = 19 : i32} : memref<?xi8>
// CHECK: return %[[POINTER]] : memref<?xi8>
func.func @pointer_capacity(%offset: index, %address: i64) -> memref<?xi8> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c248031 = arith.constant 248031 : index
  %last = arith.addi %offset, %c248031 {ssbuffer.block_id = 19 : i32} : index
  %end = arith.addi %last, %c1 {ssbuffer.block_id = 19 : i32} : index
  %capacity = arith.maxsi %end, %c0 {ssbuffer.block_id = 19 : i32} : index
  %pointer = hivm.hir.pointer_cast(%address) [%capacity] {ssbuffer.block_id = 19 : i32} : memref<?xi8>
  return %pointer : memref<?xi8>
}

// Folding to an existing producer must retain that producer's block ID.
// CHECK-LABEL: func.func @existing_scalar_keeps_its_block
// CHECK: %[[PRODUCT:.*]] = arith.muli %{{.*}}, %{{.*}} {ssbuffer.block_id = 7 : i32} : index
// CHECK-NOT: ssbuffer.block_id = 19
// CHECK-NOT: arith.addi
// CHECK: return %[[PRODUCT]] : index
func.func @existing_scalar_keeps_its_block(%lhs: index, %rhs: index) -> index {
  %c0 = arith.constant 0 : index
  %product = arith.muli %lhs, %rhs {ssbuffer.block_id = 7 : i32} : index
  %identity = arith.addi %product, %c0 {ssbuffer.block_id = 19 : i32} : index
  return %identity : index
}

// An existing producer outside the scheduled blocks must not acquire the
// block ID of a folded consumer either.
// CHECK-LABEL: func.func @existing_unassigned_scalar_stays_unassigned
// CHECK: %[[PRODUCT:.*]] = arith.muli %{{[^ ,]+}}, %{{[^ ,]+}} : index
// CHECK-NOT: ssbuffer.block_id
// CHECK-NOT: arith.addi
// CHECK: return %[[PRODUCT]] : index
func.func @existing_unassigned_scalar_stays_unassigned(%lhs: index, %rhs: index) -> index {
  %c0 = arith.constant 0 : index
  %product = arith.muli %lhs, %rhs : index
  %identity = arith.addi %product, %c0 {ssbuffer.block_id = 19 : i32} : index
  return %identity : index
}
