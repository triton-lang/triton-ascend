// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // Void if with cross-group memref dependency: two block_id groups in else,
  // one produces a memref.alloc, the other consumes it.
  // The else branch creates a local memref.alloc placeholder (place-and-use)
  // with block_id = -1. No memref.alloca should appear.

  // CHECK-LABEL: func.func @test_void_if_memref_placeholder
  // CHECK: scf.for
  // block_id=8 result if:
  // CHECK: scf.if
  // CHECK: memref.alloc() {{.*}}ssbuffer.block_id = 8 : i32
  // else: local memref.alloc placeholder (place-and-use), block_id = -1
  // CHECK: else
  // CHECK: memref.alloc() {{.*}}ssbuffer.block_id = -1 : i32
  // block_id=9 void if:
  // CHECK: scf.if
  // Placeholder branch must NOT use memref.alloca
  // CHECK-NOT: memref.alloca

  func.func @test_void_if_memref_placeholder(%cond: i1) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {
      scf.if %cond {
      } else {
        %alloc = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<16xi32>
        memref.dealloc %alloc {ssbuffer.block_id = 9 : i32} : memref<16xi32>
      }
    } {ssbuffer.main_loop = 0 : i64}
    return
  }

  // ============================================================================
  // Void if with dynamic-dim memref cross-group dependency.
  // Block 5 produces a memref<?xi32, strided<[1]>> (dynamic dim),
  // block 3 consumes it. Placeholder must substitute dynamic dims with 1
  // when creating tensor.empty (tensor::EmptyOp requires static shapes).
  // ============================================================================

  // CHECK-LABEL: func.func @test_dynamic_memref_placeholder
  // for inside main_loop:
  // CHECK: scf.for
  // reinterpret_cast is yielded through the yield chain (then branch),
  // and cloned locally in the else branch with block_id=-1 (place-and-use).
  // block_id=5 result if: reinterpret_cast in then, local clone in else:
  // CHECK: scf.if
  // CHECK: memref.reinterpret_cast {{.*}} {ssbuffer.block_id = 5 : i32}
  // else block: local reinterpret_cast cloned with block_id=-1:
  // CHECK: else
  // CHECK: memref.reinterpret_cast {{.*}} {ssbuffer.block_id = -1 : i32}
  // block_id=3 void if: reinterpret_cast from yield-chain result, block_id=3:
  // CHECK: scf.if
  // CHECK: memref.reinterpret_cast {{.*}} {ssbuffer.block_id = 3 : i32}

  func.func @test_dynamic_memref_placeholder(%cond: i1) {
    %alloc = memref.alloc() {ssbuffer.block_id = 13 : i32} : memref<16xi32>
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {
      scf.if %cond {
      } else {
        %r = memref.reinterpret_cast %alloc to offset: [0], sizes: [8], strides: [1]
            {ssbuffer.block_id = 5 : i32} : memref<16xi32> to memref<?xi32, strided<[1]>>
        %r2 = memref.reinterpret_cast %r to offset: [0], sizes: [8], strides: [1]
            {ssbuffer.block_id = 3 : i32} : memref<?xi32, strided<[1]>> to memref<8xi32, strided<[1]>>
      }
    } {ssbuffer.main_loop = 0 : i64}
    return
  }

  // ============================================================================
  // Static memref placeholder MUST carry s sbuffer.block_id = -1
  // so downstream passes can identify it as a placeholder (not a real alloc).
  // ============================================================================

  // CHECK-LABEL: func.func @test_placeholder_block_id_attr
  // CHECK: scf.for
  // block_id=8 result if:
  // CHECK: scf.if
  // CHECK: memref.alloc() {{.*}}ssbuffer.block_id = 8 : i32
  // else: local memref.alloc placeholder (place-and-use), block_id = -1
  // CHECK: else
  // CHECK: memref.alloc() {{.*}}ssbuffer.block_id = -1 : i32
  // block_id=9 void if:
  // CHECK: scf.if

  func.func @test_placeholder_block_id_attr(%cond: i1) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {
      scf.if %cond {
      } else {
        %alloc = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<16xi32>
        memref.dealloc %alloc {ssbuffer.block_id = 9 : i32} : memref<16xi32>
      }
    } {ssbuffer.main_loop = 0 : i64}
    return
  }

  // ============================================================================
  // Void if consuming GM memref (function arg) with 5 groups and
  // intermediate arith ops. Verifies place-and-use for arg-sourced memref
  // placeholders: the first split-if creates local reinterpret_cast
  // placeholders with block_id=-1, and subsequent split-ifs pass
  // through via the yield chain.
  // ============================================================================

  // CHECK-LABEL: func.func @test_void_if_gm_arg_multi_group
  // for inside main_loop:
  // CHECK: scf.for
  // negated condition (else-side split):
  // CHECK: arith.xori
  // block_id=5 result if: reinterpret_cast in then, local clone in else:
  // CHECK: scf.if
  // CHECK: memref.reinterpret_cast {{.*}} {ssbuffer.block_id = 5 : i32}
  // else: local clone, block_id=-1 (place-and-use):
  // CHECK: else
  // CHECK: memref.reinterpret_cast {{.*}} {ssbuffer.block_id = -1 : i32}
  // block_id=6 result if:
  // CHECK: scf.if
  // CHECK: memref.reinterpret_cast {{.*}} {ssbuffer.block_id = 6 : i32}
  // else: yield chain, passes through previous split-if results:
  // CHECK: else
  // CHECK: scf.yield
  // block_id=7 result if (memref.load):
  // CHECK: scf.if
  // CHECK: memref.load {{.*}} {ssbuffer.block_id = 7 : i32}
  // else: yield chain:
  // CHECK: else
  // CHECK: scf.yield
  // block_id=8 result if (arith.mulf):
  // CHECK: scf.if
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 8 : i32}
  // else: yield chain:
  // CHECK: else
  // CHECK: scf.yield
  // block_id=9 void if (arith.addf):
  // CHECK: scf.if
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 9 : i32}
  // No memref.alloca:
  // CHECK-NOT: memref.alloca

  func.func @test_void_if_gm_arg_multi_group(%cond: i1, %arg0: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {
      scf.if %cond {
      } else {
        %r1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1]
            {ssbuffer.block_id = 5 : i32} : memref<?xf32> to memref<?xf32, strided<[1]>>
        %r2 = memref.reinterpret_cast %r1 to offset: [0], sizes: [4], strides: [2]
            {ssbuffer.block_id = 6 : i32} : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[2]>>
        %val = memref.load %r2[%c0] {ssbuffer.block_id = 7 : i32} : memref<?xf32, strided<[2]>>
        %tmp = arith.mulf %val, %val {ssbuffer.block_id = 8 : i32} : f32
        %result = arith.addf %tmp, %val {ssbuffer.block_id = 9 : i32} : f32
      }
    } {ssbuffer.main_loop = 0 : i64}
    return
  }
}
