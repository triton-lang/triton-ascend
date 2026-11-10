// RUN: triton-opt --triton-control-flow-opt --split-input-file %s | FileCheck %s

// The condition is loop-carried but not returned. The nested pointer loop must
// still be decomposed without inventing a result for the outer while.
module {
  tt.func public @while_without_results(%base: !tt.ptr<f32>, %flag: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %false = arith.constant false
    %offset = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %ptrs = tt.addptr %splat, %offset : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %step = arith.constant dense<16> : tensor<16xi32>
    scf.while (%condition = %flag) : (i1) -> () {
      scf.condition(%condition)
    } do {
      %end = scf.for %i = %c0 to %c2 step %c1 iter_args(%p = %ptrs) -> tensor<16x!tt.ptr<f32>> : i32 {
        %value = tt.load %p : tensor<16x!tt.ptr<f32>>
        tt.store %p, %value : tensor<16x!tt.ptr<f32>>
        %next = tt.addptr %p, %step : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        scf.yield %next : tensor<16x!tt.ptr<f32>>
      }
      scf.yield %false : i1
    }
    tt.return
  }
}

// CHECK-LABEL: tt.func public @while_without_results
// CHECK: scf.while (%[[COND:.*]] = %{{.*}}) : (i1) -> () {
// CHECK: scf.condition(%[[COND]])
// CHECK: } do {
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (i32)
// CHECK: tt.load
// CHECK: tt.store
// CHECK: scf.yield {{.*}} : i32
// CHECK: scf.yield {{.*}} : i1
// CHECK: tt.return

// -----

// The after region has more arguments than the before region. Decompose a
// block-pointer loop in the before region and preserve both forwarded values.
module {
  tt.func public @while_more_results(%base: !tt.ptr<f32>, %flag: i1, %count: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %ptr = tt.make_tensor_ptr %base, [%c16_i64], [%c1_i64], [%c0] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
    %result:2 = scf.while (%before = %count) : (i32) -> (i1, i32) {
      %end = scf.for %i = %c0 to %c2 step %c1 iter_args(%p = %ptr) -> !tt.ptr<tensor<16xf32>> : i32 {
        %value = tt.load %p : !tt.ptr<tensor<16xf32>>
        tt.store %p, %value : !tt.ptr<tensor<16xf32>>
        %next = tt.advance %p, [%c1] : !tt.ptr<tensor<16xf32>>
        scf.yield %next : !tt.ptr<tensor<16xf32>>
      }
      %positive = arith.cmpi sgt, %before, %c0 : i32
      scf.condition(%positive) %flag, %before : i1, i32
    } do {
    ^bb0(%after_flag: i1, %after_count: i32):
      %next = arith.subi %after_count, %c1 : i32
      %selected = arith.select %after_flag, %next, %c0 : i32
      scf.yield %selected : i32
    }
    tt.return %result#1 : i32
  }
}

// CHECK-LABEL: tt.func public @while_more_results
// CHECK: %[[RESULT:.*]]:2 = scf.while (%[[BEFORE:.*]] = %{{.*}}) : (i32) -> (i1, i32) {
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (i32)
// CHECK: tt.load
// CHECK: tt.store
// CHECK: scf.condition(%{{.*}}) %{{.*}}, %[[BEFORE]] : i1, i32
// CHECK: } do {
// CHECK: ^bb0(%[[FLAG:.*]]: i1, %[[COUNT:.*]]: i32):
// CHECK: %[[NEXT:.*]] = arith.subi %[[COUNT]], %{{.*}} : i32
// CHECK: %[[SELECTED:.*]] = arith.select %[[FLAG]], %[[NEXT]], %{{.*}} : i32
// CHECK: scf.yield %[[SELECTED]] : i32
// CHECK: tt.return %[[RESULT]]#1 : i32

// -----

// Equal arity does not imply equal types. The loop backedge consumes i32,
// while the body and users of the outer result consume f32.
module {
  tt.func public @while_different_types(%base: !tt.ptr<f32>, %count: i32) -> f32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %offset = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %ptrs = tt.addptr %splat, %offset : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %step = arith.constant dense<16> : tensor<16xi32>
    %result = scf.while (%before = %count) : (i32) -> f32 {
      %positive = arith.cmpi sgt, %before, %c0 : i32
      %forwarded = arith.sitofp %before : i32 to f32
      scf.condition(%positive) %forwarded : f32
    } do {
    ^bb0(%after: f32):
      %end = scf.for %i = %c0 to %c2 step %c1 iter_args(%p = %ptrs) -> tensor<16x!tt.ptr<f32>> : i32 {
        %value = tt.load %p : tensor<16x!tt.ptr<f32>>
        tt.store %p, %value : tensor<16x!tt.ptr<f32>>
        %next = tt.addptr %p, %step : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        scf.yield %next : tensor<16x!tt.ptr<f32>>
      }
      %integer = arith.fptosi %after : f32 to i32
      %next = arith.subi %integer, %c1 : i32
      scf.yield %next : i32
    }
    tt.return %result : f32
  }
}

// CHECK-LABEL: tt.func public @while_different_types
// CHECK: %[[RESULT:.*]] = scf.while (%[[BEFORE:.*]] = %{{.*}}) : (i32) -> f32 {
// CHECK: %[[FORWARDED:.*]] = arith.sitofp %[[BEFORE]] : i32 to f32
// CHECK: scf.condition(%{{.*}}) %[[FORWARDED]] : f32
// CHECK: } do {
// CHECK: ^bb0(%[[AFTER:.*]]: f32):
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (i32)
// CHECK: tt.load
// CHECK: tt.store
// CHECK: %[[INTEGER:.*]] = arith.fptosi %[[AFTER]] : f32 to i32
// CHECK: %[[NEXT:.*]] = arith.subi %[[INTEGER]], %{{.*}} : i32
// CHECK: scf.yield %[[NEXT]] : i32
// CHECK: tt.return %[[RESULT]] : f32
