// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ============================================================================
  // GM provenance for placeholder in void-if split across block_id groups.
  // block_id=3 produces a strided memref from a function argument (GM).
  // The placeholder in the passthrough branch MUST use the same function
  // argument (memref.reinterpret_cast %arg1), NOT memref.alloca (UB).
  // The else block creates its own local reinterpret_cast (place-and-use)
  // instead of referencing the pre-created placeholder from outside.
  // The then branch keeps yielding the reinterpret_cast through the yield chain.
  // ============================================================================

  // CHECK-LABEL: func.func @test_gm_placeholder
  // for inside main_loop:
  // CHECK: scf.for
  // block_id=3 result if: reinterpret_cast in then, local clone in else:
  // CHECK: scf.if
  // CHECK: memref.reinterpret_cast %arg1 {{.*}} {ssbuffer.block_id = 3 : i32}
  // else block: local reinterpret_cast cloned inside else block (block_id=-1):
  // CHECK: else
  // CHECK: memref.reinterpret_cast %arg1 {{.*}} {ssbuffer.block_id = -1 : i32}
  // block_id=4 void if: reinterpret_cast from yield-chain result, keeps block_id=4:
  // CHECK: scf.if
  // CHECK: memref.reinterpret_cast {{.*}} {ssbuffer.block_id = 4 : i32}
  // No memref.alloca placeholder anywhere:
  // CHECK-NOT: memref.alloca

  func.func @test_gm_placeholder(%cond: i1, %arg0: memref<?xf32>) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {
      scf.if %cond {
      } else {
        %r = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1]
            {ssbuffer.block_id = 3 : i32} : memref<?xf32> to memref<?xf32, strided<[1]>>
        %r2 = memref.reinterpret_cast %r to offset: [0], sizes: [8], strides: [1]
            {ssbuffer.block_id = 4 : i32} : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
      }
    } {ssbuffer.main_loop = 0 : i64}
    return
  }

  // ============================================================================
  // Nested if: outer void if contains inner scf.if with block_id groups.
  // Inner if produces a strided memref from a function argument (GM).
  // The inner if is then itself split. Placeholders at BOTH levels
  // (outer split-if chain and inner split-if chain) MUST use the function
  // argument, NOT memref.alloca.
  // The then branch keeps yielding reinterpret_cast through the yield chain;
  // the else branch creates local reinterpret_cast from function arg (place-and-use).
  // ============================================================================

  // CHECK-LABEL: func.func @test_nested_gm_placeholder
  // for inside main_loop:
  // CHECK: scf.for
  // No memref.alloca placeholder anywhere:
  // CHECK-NOT: memref.alloca
  // reinterpret_cast cloned at both outer and inner levels, preserving %arg2 provenance.
  // First outer split-if wraps inner if with block_id=7:
  // CHECK: scf.if
  // CHECK: scf.if %arg1
  // CHECK: memref.reinterpret_cast %arg2 {{.*}} {ssbuffer.block_id = 7 : i32}
  // else block of first outer split-if: local reinterpret_cast from %arg2:
  // CHECK: else
  // CHECK: memref.reinterpret_cast %arg2 {{.*}} {ssbuffer.block_id = -1 : i32}
  // Second outer split-if: inner if with block_id=8:
  // CHECK: scf.if
  // CHECK: scf.if %arg1
  // CHECK: memref.reinterpret_cast {{.*}} {ssbuffer.block_id = 8 : i32}

  func.func @test_nested_gm_placeholder(%cond1: i1, %cond2: i1, %arg0: memref<?xf32>) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {
      scf.if %cond1 {
      } else {
        scf.if %cond2 {
          %r = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1]
              {ssbuffer.block_id = 7 : i32} : memref<?xf32> to memref<?xf32, strided<[1]>>
          %r2 = memref.reinterpret_cast %r to offset: [0], sizes: [8], strides: [1]
              {ssbuffer.block_id = 8 : i32} : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
        }
      }
    } {ssbuffer.main_loop = 0 : i64}
    return
  }
}
