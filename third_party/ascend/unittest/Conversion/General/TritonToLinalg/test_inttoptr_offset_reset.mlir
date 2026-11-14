// RUN: triton-opt "--triton-to-linalg=global-kernel=false named-ops=True" --split-input-file %s | FileCheck %s

// Test the positive static offset representation selected when a scalar
// pointer reaches the layout-sensitive memref.copy boundary.
//
// When a scalar tt.addptr adds a constant offset to a tt.int_to_ptr result,
// the conversion retains that offset in the memref view and expands the raw
// pointer capacity to cover both the leading offset and the eight-element
// extent. The copy must consume that exact offset-bearing view.

// CHECK-LABEL: func.func @test_inttoptr_offset_reset
// CHECK: %[[ADDRESS:.*]] = memref.load {{.*}} : memref<?xi64>
// CHECK: %[[CAPACITY:.*]] = arith.constant 12 : index
// CHECK: %[[POINTER:.*]] = hivm.hir.pointer_cast(%[[ADDRESS]]) [%[[CAPACITY]]] : memref<?xi32>
// CHECK: %[[OFFSET_VIEW:.*]] = memref.reinterpret_cast %[[POINTER]] to offset: [4], sizes: [8], strides: [1]
// CHECK-SAME: to memref<8xi32, strided<[1], offset: 4>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<8xi32>
// CHECK: memref.copy %[[OFFSET_VIEW]], %[[ALLOC]] : memref<8xi32, strided<[1], offset: 4>> to memref<8xi32>

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @test_inttoptr_offset_reset(%addr_ptr: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    // Load a raw i64 address from memory, then convert to pointer.
    // This creates a hivm.pointer_cast during conversion.
    %raw_addr = tt.load %addr_ptr : !tt.ptr<i64>
    %ptr = tt.int_to_ptr %raw_addr : i64 -> !tt.ptr<i32>
    // Scalar addptr with a constant offset — the offset ends up in the
    // ReinterpretCastOp result type's strided layout.
    %offset_ptr = tt.addptr %ptr, %c4_i32 : !tt.ptr<i32>, i32
    // Splat to tensor and load via indirect addressing.
    %ptrs = tt.splat %offset_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %offsets = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %load_ptrs = tt.addptr %ptrs, %offsets : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %data = tt.load %load_ptrs : tensor<8x!tt.ptr<i32>>
    // Store result.
    %out_ptrs = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %out_addptr = tt.addptr %out_ptrs, %offsets : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %out_addptr, %data : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// The dynamic variant also retains the original raw address and applies the
// displacement exactly once, in the view consumed by memref.copy.

// CHECK-LABEL: func.func @test_inttoptr_dyn_offset_reset
// CHECK: %[[ADDRESS:.*]] = memref.load {{.*}} : memref<?xi64>
// CHECK: %[[OFFSET:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK: %[[POINTER:.*]] = hivm.hir.pointer_cast(%[[ADDRESS]]) [%{{.*}}] : memref<?xi32>
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[POINTER]] to offset: [%[[OFFSET]]], sizes: [8], strides: [1]
// CHECK-SAME: to memref<8xi32, strided<[1], offset: ?>>
// CHECK: memref.copy %[[VIEW]], {{.*}} : memref<8xi32, strided<[1], offset: ?>> to memref<8xi32>

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @test_inttoptr_dyn_offset_reset(%addr_ptr: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %dyn_offset: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %raw_addr = tt.load %addr_ptr : !tt.ptr<i64>
    %ptr = tt.int_to_ptr %raw_addr : i64 -> !tt.ptr<i32>
    // Scalar addptr with a dynamic (runtime) offset.
    %offset_ptr = tt.addptr %ptr, %dyn_offset : !tt.ptr<i32>, i32
    %ptrs = tt.splat %offset_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %offsets = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %load_ptrs = tt.addptr %ptrs, %offsets : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %data = tt.load %load_ptrs : tensor<8x!tt.ptr<i32>>
    %out_ptrs = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %out_addptr = tt.addptr %out_ptrs, %offsets : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %out_addptr, %data : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}
