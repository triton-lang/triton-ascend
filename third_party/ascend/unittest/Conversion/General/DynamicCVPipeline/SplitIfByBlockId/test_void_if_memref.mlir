// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // Void if with cross-group memref dependency: two block_id groups in else,
  // one produces a memref.alloc, the other consumes it.
  // The placeholder for the memref slot in the passthrough branch MUST use
  // tensor.empty + bufferization.to_memref (not memref.alloca), so downstream
  // traceback doesn't stop on the wrong alloc.

  // CHECK-LABEL: func.func @test_void_if_memref_placeholder
  // CHECK: tensor.empty
  // CHECK: bufferization.to_memref
  // Placeholder branch must NOT use memref.alloca
  // CHECK-NOT: memref.alloca

  func.func @test_void_if_memref_placeholder(%cond: i1) {
    scf.if %cond {
    } else {
      %alloc = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<16xi32>
      memref.dealloc %alloc {ssbuffer.block_id = 9 : i32} : memref<16xi32>
    }
    return
  }

  // ============================================================================
  // Void if with dynamic-dim memref cross-group dependency.
  // Block 5 produces a memref<?xi32, strided<[1]>> (dynamic dim),
  // block 3 consumes it. Placeholder must substitute dynamic dims with 1
  // when creating tensor.empty (tensor::EmptyOp requires static shapes).
  // ============================================================================

  // CHECK-LABEL: func.func @test_dynamic_memref_placeholder
  // block_id=5 split if: yields the dynamic memref
  // CHECK: scf.if
  // CHECK-SAME: -> (memref<?xi32, strided<[1]>>)
  // CHECK: memref.reinterpret_cast {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // block_id=3 void if: consumes it
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK: memref.reinterpret_cast {{.*}} {ssbuffer.block_id = 3 : i32}

  func.func @test_dynamic_memref_placeholder(%cond: i1) {
    %alloc = memref.alloc() {ssbuffer.block_id = 13 : i32} : memref<16xi32>
    scf.if %cond {
    } else {
      %r = memref.reinterpret_cast %alloc to offset: [0], sizes: [8], strides: [1]
          {ssbuffer.block_id = 5 : i32} : memref<16xi32> to memref<?xi32, strided<[1]>>
      %r2 = memref.reinterpret_cast %r to offset: [0], sizes: [8], strides: [1]
          {ssbuffer.block_id = 3 : i32} : memref<?xi32, strided<[1]>> to memref<8xi32, strided<[1]>>
    }
    return
  }

  // ============================================================================
  // Static memref placeholder tensor.empty MUST carry s sbuffer.block_id = -1
  // so downstream passes can identify it as a placeholder (not a real alloc).
  // ============================================================================

  // CHECK-LABEL: func.func @test_placeholder_block_id_attr
  // CHECK: tensor.empty() {ssbuffer.block_id = -1 : i32}
  // CHECK: bufferization.to_memref

  func.func @test_placeholder_block_id_attr(%cond: i1) {
    scf.if %cond {
    } else {
      %alloc = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<16xi32>
      memref.dealloc %alloc {ssbuffer.block_id = 9 : i32} : memref<16xi32>
    }
    return
  }
}
