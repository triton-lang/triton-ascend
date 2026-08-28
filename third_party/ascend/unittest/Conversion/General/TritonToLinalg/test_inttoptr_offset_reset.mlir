// RUN: triton-opt "--triton-to-linalg=global-kernel=false named-ops=True" --split-input-file %s | FileCheck %s

// Test that ReinterpretCastOp's result type has offset: 0 after the
// "Calculate size of PointerCastOp precisely" optimization.
//
// When a scalar tt.addptr adds a constant offset to a tt.int_to_ptr result,
// the conversion creates a hivm.pointer_cast + memref.reinterpret_cast with
// the original offset in the result type. The optimization then bakes the
// offset into the base address (realAddr = addr + offset * elemSize) and
// creates a new ReinterpretCastOp with offset 0. The result type's strided
// layout must also have offset 0, otherwise the op offset and type offset
// are inconsistent.

// CHECK-LABEL: func.func @test_inttoptr_offset_reset
// The optimized ReinterpretCastOp must have offset 0 in the result type.
// CHECK: memref.reinterpret_cast {{.*}} strided<[1]>
// CHECK-NOT: memref.reinterpret_cast {{.*}} strided<[1], offset: 4>

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

// Dynamic offset variant: the scalar tt.addptr uses a runtime value as
// offset, so the original ReinterpretCastOp has a dynamic offset (?)
// in its result type. After the optimization bakes the offset into the
// base address, the new ReinterpretCastOp still have a dynamic offset
// in the result type.

// CHECK-LABEL: func.func @test_inttoptr_dyn_offset_reset
// CHECK: memref.reinterpret_cast {{.*}} strided<[1], offset: ?>

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
