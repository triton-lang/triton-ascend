// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ==========================================================================
  // Placeholder tests — verify correct placeholder creation for special types
  // (memref from function args, dynamic-shape memrefs, etc.)
  //
  // Key rules:
  // - Strided memrefs from function args (GM) must reuse the arg in placeholders
  //   (via memref.reinterpret_cast), never memref.alloca
  // - Placeholders are created locally in the branch that needs them ("即插即用")
  // - All placeholders carry ssbuffer.block_id = -1
  // ==========================================================================

  // --------------------------------------------------------------------------
  // GM provenance: placeholder in void-if split preserving function arg
  // block_id=3 produces a strided memref from %arg1 (GM).
  // The placeholder in the else branch must also use %arg1, NOT alloca.
  // --------------------------------------------------------------------------

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

  // --------------------------------------------------------------------------
  // Nested GM placeholder: outer void if contains inner scf.if with groups
  // Inner if produces strided memref from %arg2 (GM).
  // Placeholders at BOTH levels must use the function arg, NOT alloca.
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_nested_gm_placeholder
  // for inside main_loop:
  // CHECK: scf.for
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

  // --------------------------------------------------------------------------
  // Void if memref placeholder: cross-group memref.alloc dependency
  // block_id=8 produces a memref.alloc, block_id=9 consumes it (dealloc)
  // Else branch creates local memref.alloc placeholder with block_id=-1
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_memref_placeholder_basic
  // CHECK: scf.for
  // block_id=8 result if:
  // CHECK: scf.if
  // CHECK: memref.alloc() {{.*}}ssbuffer.block_id = 8 : i32
  // else: local memref.alloc placeholder (place-and-use), block_id = -1
  // CHECK: else
  // CHECK: memref.alloc() {{.*}}ssbuffer.block_id = -1 : i32
  // block_id=9 void if:
  // CHECK: scf.if
  // CHECK-NOT: memref.alloca

  func.func @test_memref_placeholder_basic(%cond: i1) {
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

  // --------------------------------------------------------------------------
  // Dynamic-dim memref placeholder: block 5 produces memref<?xi32>,
  // block 3 consumes it. Placeholder must substitute dynamic dims with 1
  // when creating tensor.empty (tensor::EmptyOp requires static shapes).
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_dynamic_memref_placeholder
  // for inside main_loop:
  // CHECK: scf.for
  // block_id=5 result if: reinterpret_cast in then, local clone in else:
  // CHECK: scf.if
  // CHECK: memref.reinterpret_cast {{.*}} {ssbuffer.block_id = 5 : i32}
  // else: local clone with block_id=-1 (place-and-use):
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

  // --------------------------------------------------------------------------
  // Placeholder block_id attribute: static memref placeholders must carry
  // ssbuffer.block_id = -1 so downstream passes can identify them.
  // --------------------------------------------------------------------------

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

  // --------------------------------------------------------------------------
  // Void if consuming GM memref (function arg) with 5 groups and
  // intermediate arith ops. Verifies place-and-use for arg-sourced memref
  // placeholders across a longer split-if chain.
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_gm_arg_multi_group
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
  // block_id=7 result if (memref.load):
  // CHECK: scf.if
  // CHECK: memref.load {{.*}} {ssbuffer.block_id = 7 : i32}
  // block_id=8 result if (arith.mulf):
  // CHECK: scf.if
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 8 : i32}
  // block_id=9 void if (arith.addf):
  // CHECK: scf.if
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 9 : i32}
  // CHECK-NOT: memref.alloca

  func.func @test_gm_arg_multi_group(%cond: i1, %arg0: memref<?xf32>) {
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

  // --------------------------------------------------------------------------
  // Placeholder dedup: non-last group with multiple same-type cross-group values
  // G0 (bid=10) produces %a (index) and %b (index) — both index type.
  // The else block must create only ONE arith.constant 0 : index and reuse it
  // for both yield slots, instead of creating a separate constant per slot.
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_placeholder_dedup
  // CHECK: scf.for
  // G0 produces 2 index values:
  // CHECK: [[G0:%.*]]:2 = scf.if {{.*}} -> (index, index)
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield
  // else: exactly ONE placeholder constant 0 : index, reused for both slots
  // CHECK: else
  // CHECK: [[PH:%.*]] = arith.constant {ssbuffer.block_id = -1 : i32} 0 : index
  // CHECK-NOT: arith.constant {ssbuffer.block_id = -1 : i32} 0 : index
  // CHECK: scf.yield [[PH]], [[PH]] : index, index
  // G1 (bid=11): last if with original yield type, references G0#0 and G0#1
  // CHECK: [[G1:%.*]] = scf.if {{.*}} -> (tensor<16xf32>)
  // CHECK: arith.index_cast [[G0]]#0 {ssbuffer.block_id = 11 : i32} : index to i32
  // CHECK: arith.index_cast [[G0]]#1 {ssbuffer.block_id = 11 : i32} : index to i32

  func.func @test_placeholder_dedup(%cond: i1) {
    %cst = arith.constant {ssbuffer.block_id = 14 : i32} 0.0 : f32
    %t0 = tensor.empty() {ssbuffer.block_id = 14 : i32} : tensor<16xf32>
    %fallback = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%cst : f32) outs(%t0 : tensor<16xf32>) -> tensor<16xf32>
    %c0 = arith.constant {ssbuffer.block_id = 14 : i32} 0 : index
    %c4 = arith.constant {ssbuffer.block_id = 14 : i32} 4 : index
    %c64 = arith.constant {ssbuffer.block_id = 14 : i32} 64 : index
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {
      %result = scf.if %cond -> (tensor<16xf32>) {
        // G0 (bid=10): produces two index values of the same type
        %a = arith.addi %c0, %c4 {ssbuffer.block_id = 10 : i32} : index
        %b = arith.addi %c0, %c64 {ssbuffer.block_id = 10 : i32} : index
        // G1 (bid=11): consumes both, produces the original yield
        %cast_a = arith.index_cast %a {ssbuffer.block_id = 11 : i32} : index to i32
        %cast_b = arith.index_cast %b {ssbuffer.block_id = 11 : i32} : index to i32
        %sum = arith.addi %cast_a, %cast_b {ssbuffer.block_id = 11 : i32} : i32
        %val = arith.sitofp %sum {ssbuffer.block_id = 11 : i32} : i32 to f32
        %r = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<16xf32>
        %result = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%val : f32) outs(%r : tensor<16xf32>) -> tensor<16xf32>
        scf.yield %result : tensor<16xf32>
      } else {
        scf.yield %fallback : tensor<16xf32>
      }
    } {ssbuffer.main_loop = 0 : i64}
    return
  }
}
