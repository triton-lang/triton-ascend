// RUN: triton-opt %s --split-input-file --ub-usage-opt --verify-each | FileCheck %s --enable-var-scope

// Sink the large tensor boundary through a reduction inside if regions.
// The loop variants also exercise the enclosing for argument/result mapping.

// CHECK-LABEL: func.func @if_regions(
func.func @if_regions(%cond: i1, %n: index, %input: tensor<64x64xf32>, %init: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.if %cond -> tensor<64xf32> {
    %big = math.exp %input {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    // CHECK: linalg.reduce {{.*}}ssbuffer.block_id = 10 : i32
    %small = linalg.reduce ins(%big : tensor<64x64xf32>) outs(%init : tensor<64xf32>) dimensions = [1] {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"}
    (%x: f32, %y: f32) {
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    }
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %next = arith.addf %small, %init {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    scf.yield %next : tensor<64xf32>
  } else {
    %big = math.exp %input {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    // CHECK: linalg.reduce {{.*}}ssbuffer.block_id = 30 : i32
    %small = linalg.reduce ins(%big : tensor<64x64xf32>) outs(%init : tensor<64xf32>) dimensions = [1] {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"}
    (%x: f32, %y: f32) {
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    }
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %next = arith.addf %small, %init {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    scf.yield %next : tensor<64xf32>
  }
  return %result : tensor<64xf32>
}

// -----

// CHECK-LABEL: func.func @for_if_backedge(
func.func @for_if_backedge(%cond: i1, %n: index, %input: tensor<64x64xf32>, %init: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> tensor<64xf32> {
    %result = scf.if %cond -> tensor<64xf32> {
      %big = math.exp %input {ssbuffer.block_id = 110 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      // CHECK: linalg.reduce {{.*}}ssbuffer.block_id = 110 : i32
      %small = linalg.reduce ins(%big : tensor<64x64xf32>) outs(%init : tensor<64xf32>) dimensions = [1] {ssbuffer.block_id = 111 : i32, ssbuffer.core_type = "VECTOR"}
      (%x: f32, %y: f32) {
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      }
      // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 111 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
      %next = arith.addf %small, %acc {ssbuffer.block_id = 111 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
      scf.yield %next : tensor<64xf32>
    } else {
      scf.yield %acc : tensor<64xf32>
    }
    scf.yield %result : tensor<64xf32>
  }
  return %loop : tensor<64xf32>
}

// -----

// CHECK-LABEL: func.func @for_if_else_backedge(
func.func @for_if_else_backedge(%cond: i1, %n: index, %input: tensor<64x64xf32>, %init: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> tensor<64xf32> {
    %result = scf.if %cond -> tensor<64xf32> {
      %big = math.exp %input {ssbuffer.block_id = 210 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      // CHECK: linalg.reduce {{.*}}ssbuffer.block_id = 210 : i32
      %small = linalg.reduce ins(%big : tensor<64x64xf32>) outs(%init : tensor<64xf32>) dimensions = [1] {ssbuffer.block_id = 211 : i32, ssbuffer.core_type = "VECTOR"}
      (%x: f32, %y: f32) {
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      }
      // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 211 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
      %next = arith.addf %small, %acc {ssbuffer.block_id = 211 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
      scf.yield %next : tensor<64xf32>
    } else {
      %big = math.exp %input {ssbuffer.block_id = 230 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      // CHECK: linalg.reduce {{.*}}ssbuffer.block_id = 230 : i32
      %small = linalg.reduce ins(%big : tensor<64x64xf32>) outs(%init : tensor<64xf32>) dimensions = [1] {ssbuffer.block_id = 231 : i32, ssbuffer.core_type = "VECTOR"}
      (%x: f32, %y: f32) {
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      }
      // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 230 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
      %next = arith.addf %small, %acc {ssbuffer.block_id = 231 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
      scf.yield %next : tensor<64xf32>
    }
    scf.yield %result : tensor<64xf32>
  }
  return %loop : tensor<64xf32>
}

// -----

// Without the backedge, both cuts cost 256 bytes; move only the reduction.
// This pairs with the duplicate-edge case below to make the weight observable.
// CHECK-LABEL: func.func @for_if_small_boundary_control(
func.func @for_if_small_boundary_control(%cond: i1, %n: index, %input: tensor<4x8x8xf32>, %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> tensor<8x8xf32> {
    %result = scf.if %cond -> tensor<8x8xf32> {
      %big = math.exp %input {ssbuffer.block_id = 310 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4x8x8xf32>
      // CHECK: linalg.reduce {{.*}}ssbuffer.block_id = 310 : i32
      %small = linalg.reduce ins(%big : tensor<4x8x8xf32>) outs(%init : tensor<8x8xf32>) dimensions = [0] {ssbuffer.block_id = 311 : i32, ssbuffer.core_type = "VECTOR"}
      (%x: f32, %y: f32) {
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      }
      // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 311 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %consumer = arith.addf %small, %init {ssbuffer.block_id = 311 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      // CHECK: linalg.matmul {ssbuffer.block_id = 312 : i32, ssbuffer.core_type = "CUBE"}
      %cube = linalg.matmul {ssbuffer.block_id = 312 : i32, ssbuffer.core_type = "CUBE"} ins(%consumer, %init : tensor<8x8xf32>, tensor<8x8xf32>) outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
      scf.yield %small : tensor<8x8xf32>
    } else {
      scf.yield %acc : tensor<8x8xf32>
    }
    scf.yield %result : tensor<8x8xf32>
  }
  return %loop : tensor<8x8xf32>
}

// -----

// The same producer/consumer pair has an SSA edge and a double-buffered backedge.
// Yield adds no edge. Cutting after addf costs 256 bytes instead of 512
// after reduce, so both operations move before the CUBE boundary.
// CHECK-LABEL: func.func @for_if_duplicate_ssa_backedge(
func.func @for_if_duplicate_ssa_backedge(%cond: i1, %n: index, %input: tensor<4x8x8xf32>, %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> tensor<8x8xf32> {
    %result = scf.if %cond -> tensor<8x8xf32> {
      %big = math.exp %input {ssbuffer.block_id = 410 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4x8x8xf32>
      // CHECK: linalg.reduce {{.*}}ssbuffer.block_id = 410 : i32
      %small = linalg.reduce ins(%big : tensor<4x8x8xf32>) outs(%init : tensor<8x8xf32>) dimensions = [0] {ssbuffer.block_id = 411 : i32, ssbuffer.core_type = "VECTOR"}
      (%x: f32, %y: f32) {
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      }
      // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 410 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %consumer = arith.addf %small, %acc {ssbuffer.block_id = 411 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      // CHECK: linalg.matmul {ssbuffer.block_id = 412 : i32, ssbuffer.core_type = "CUBE"}
      %cube = linalg.matmul {ssbuffer.block_id = 412 : i32, ssbuffer.core_type = "CUBE"} ins(%consumer, %init : tensor<8x8xf32>, tensor<8x8xf32>) outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
      scf.yield %small : tensor<8x8xf32>
    } else {
      scf.yield %acc : tensor<8x8xf32>
    }
    scf.yield %result : tensor<8x8xf32>
  }
  return %loop : tensor<8x8xf32>
}

// -----

// The 512-byte input equals the backedge cost after reduce. The 256-byte
// CUBE boundary after addf is smaller, so move both operations. A stale
// 256-byte SSA weight would incorrectly choose a cut before addf.
// CHECK-LABEL: func.func @for_if_duplicate_backedge_cut_choice(
func.func @for_if_duplicate_backedge_cut_choice(%cond: i1, %n: index, %input: tensor<2x8x8xf32>, %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> tensor<8x8xf32> {
    %result = scf.if %cond -> tensor<8x8xf32> {
      %big = math.exp %input {ssbuffer.block_id = 510 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x8x8xf32>
      // CHECK: linalg.reduce {{.*}}ssbuffer.block_id = 510 : i32
      %small = linalg.reduce ins(%big : tensor<2x8x8xf32>) outs(%init : tensor<8x8xf32>) dimensions = [0] {ssbuffer.block_id = 511 : i32, ssbuffer.core_type = "VECTOR"}
      (%x: f32, %y: f32) {
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      }
      // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 510 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %consumer = arith.addf %small, %acc {ssbuffer.block_id = 511 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      // CHECK: linalg.matmul {ssbuffer.block_id = 512 : i32, ssbuffer.core_type = "CUBE"}
      %cube = linalg.matmul {ssbuffer.block_id = 512 : i32, ssbuffer.core_type = "CUBE"} ins(%consumer, %init : tensor<8x8xf32>, tensor<8x8xf32>) outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
      scf.yield %small : tensor<8x8xf32>
    } else {
      scf.yield %acc : tensor<8x8xf32>
    }
    scf.yield %result : tensor<8x8xf32>
  }
  return %loop : tensor<8x8xf32>
}

// -----

// The 512-byte input equals the backedge cost after reduce. The 256-byte
// CUBE boundary after addf is smaller, so move both operations. A stale
// 256-byte SSA weight would incorrectly choose a cut before addf.
// CHECK-LABEL: func.func @for_if_duplicate_backedge_arg_first(
func.func @for_if_duplicate_backedge_arg_first(%cond: i1, %n: index, %input: tensor<2x8x8xf32>, %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> tensor<8x8xf32> {
    %result = scf.if %cond -> tensor<8x8xf32> {
      %big = math.exp %input {ssbuffer.block_id = 610 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x8x8xf32>
      // CHECK: linalg.reduce {{.*}}ssbuffer.block_id = 610 : i32
      %small = linalg.reduce ins(%big : tensor<2x8x8xf32>) outs(%init : tensor<8x8xf32>) dimensions = [0] {ssbuffer.block_id = 611 : i32, ssbuffer.core_type = "VECTOR"}
      (%x: f32, %y: f32) {
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      }
      // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 610 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %consumer = arith.addf %acc, %small {ssbuffer.block_id = 611 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      // CHECK: linalg.matmul {ssbuffer.block_id = 612 : i32, ssbuffer.core_type = "CUBE"}
      %cube = linalg.matmul {ssbuffer.block_id = 612 : i32, ssbuffer.core_type = "CUBE"} ins(%consumer, %init : tensor<8x8xf32>, tensor<8x8xf32>) outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
      scf.yield %small : tensor<8x8xf32>
    } else {
      scf.yield %acc : tensor<8x8xf32>
    }
    scf.yield %result : tensor<8x8xf32>
  }
  return %loop : tensor<8x8xf32>
}
