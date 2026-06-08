// RUN: triton-opt --discrete-load-store %s | FileCheck %s

// ============================================================================
// Test Case: @test_discrete_load
// ============================================================================
// This test verifies that for a discrete LOAD (GM -> local buffer), the block IDs
// of the local allocation, the scf.for loop (with ExtractedLoadOrStore attribute),
// and the operations inside the loop are unified to the block ID of the local
// buffer's external users.
//
// - Initial state:
//   - %alloc has block_id = 4
//   - scf.for loop and its internal ops have block_id = 3
//   - bufferization.to_tensor (external user of %alloc) has block_id = 5
// - Expected unified block_id: 5
// ============================================================================
// CHECK-LABEL: func.func @test_discrete_load
func.func @test_discrete_load(%arg0: memref<?xf16>, %lb: index, %ub: index, %step: index) {
  // CHECK: [[ALLOC:%.*]] = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"}
  %alloc = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x32xf16>

  // CHECK: scf.for [[LOOP_IV:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK:   %{{.*}} = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [1, 32], strides: [32, 1] {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK:   %{{.*}} = memref.subview [[ALLOC]][[[LOOP_IV]], 0] [1, 32] [1, 1] {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK:   memref.copy %{{.*}}, %{{.*}} {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: } {ExtractedLoadOrStore, ssbuffer.block_id = 5 : i32}
  scf.for %arg1 = %lb to %ub step %step {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 32], strides: [32, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<1x32xf16, strided<[32, 1], offset: ?>>
    %subview = memref.subview %alloc[%arg1, 0] [1, 32] [1, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x32xf16> to memref<1x32xf16, strided<[32, 1], offset: ?>>
    memref.copy %reinterpret_cast, %subview {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : memref<1x32xf16, strided<[32, 1], offset: ?>> to memref<1x32xf16, strided<[32, 1], offset: ?>>
  } {ExtractedLoadOrStore, ssbuffer.block_id = 3 : i32}

  // CHECK: %{{.*}} = bufferization.to_tensor [[ALLOC]] restrict writable {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"}
  %to_tensor = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x32xf16>
  return
}

// ============================================================================
// Test Case: @test_discrete_store
// ============================================================================
// This test verifies that for a discrete STORE (local buffer -> GM), the block IDs
// of the scf.for loop (with ExtractedLoadOrStore attribute) and the operations
// inside the loop are unified to the block ID of the local buffer (%alloc) itself.
//
// - Initial state:
//   - %alloc has block_id = 2
//   - scf.for loop and its internal ops have block_id = 3
// - Expected unified block_id: 2
// ============================================================================
// CHECK-LABEL: func.func @test_discrete_store
func.func @test_discrete_store(%arg0: memref<?xf16>, %lb: index, %ub: index, %step: index) {
  // CHECK: [[ALLOC:%.*]] = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"}
  %alloc = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x32xf16>

  // CHECK: scf.for [[LOOP_IV:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK:   %{{.*}} = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [1, 32], strides: [32, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK:   %{{.*}} = memref.subview [[ALLOC]][[[LOOP_IV]], 0] [1, 32] [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK:   memref.copy %{{.*}}, %{{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: } {ExtractedLoadOrStore, ssbuffer.block_id = 2 : i32}
  scf.for %arg1 = %lb to %ub step %step {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 32], strides: [32, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<1x32xf16, strided<[32, 1], offset: ?>>
    %subview = memref.subview %alloc[%arg1, 0] [1, 32] [1, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x32xf16> to memref<1x32xf16, strided<[32, 1], offset: ?>>
    memref.copy %subview, %reinterpret_cast {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : memref<1x32xf16, strided<[32, 1], offset: ?>> to memref<1x32xf16, strided<[32, 1], offset: ?>>
  } {ExtractedLoadOrStore, ssbuffer.block_id = 3 : i32}

  return
}

