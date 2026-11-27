// RUN: triton-opt --add_multi_buffer_inner_scope --split-input-file %s | FileCheck %s

// Two independent consumers must never expose the shared ring as writable
// tensor storage. Each selected slot is copied into a private allocation,
// including the else-yield consumer of a split conditional.
// CHECK-LABEL: func.func @outer_two_slots
// CHECK: %[[A:.*]] = memref.alloc() {ssbuffer.block_id = 2 : i32} : memref<32xf32>
// CHECK-NEXT: hivm.hir.copy ins(%{{.*}} : memref<32xf32>) outs(%[[A]] : memref<32xf32>) {ssbuffer.block_id = 2 : i32}
// CHECK-NEXT: bufferization.to_tensor %[[A]] restrict writable
// CHECK: %[[B:.*]] = memref.alloc() {ssbuffer.block_id = 3 : i32} : memref<32xf32>
// CHECK-NEXT: hivm.hir.copy ins(%{{.*}} : memref<32xf32>) outs(%[[B]] : memref<32xf32>) {ssbuffer.block_id = 3 : i32}
// CHECK-NEXT: bufferization.to_tensor %[[B]] restrict writable
// CHECK: %[[C:.*]] = memref.alloc() {ssbuffer.block_id = 4 : i32} : memref<32xf32>
// CHECK-NEXT: hivm.hir.copy ins(%{{.*}} : memref<32xf32>) outs(%[[C]] : memref<32xf32>) {ssbuffer.block_id = 4 : i32}
// CHECK-NEXT: bufferization.to_tensor %[[C]] restrict writable
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.intra_buf_count = 2 : i32} {
  func.func @outer_two_slots(%input: tensor<32xf32>, %condition: i1) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %end = arith.constant 8 : i32
    scope.scope : () -> () {
      scf.for %i = %zero to %end step %one : i32 {
        %m = arith.addf %input, %input {ssbuffer.block_id = 1 : i32} : tensor<32xf32>
        %alpha = math.exp %m {ssbuffer.block_id = 2 : i32} : tensor<32xf32>
        %later = arith.subf %m, %m {ssbuffer.block_id = 3 : i32} : tensor<32xf32>
        %carry = scf.if %condition -> tensor<32xf32> {
          scf.yield %input : tensor<32xf32>
        } else {
          scf.yield %m : tensor<32xf32>
        } {ssbuffer.block_id = 4 : i32}
      } {ssbuffer.main_loop = 1 : i64, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}

// -----

// The one-slot fast path must protect both consumers too.
// CHECK-LABEL: func.func @outer_one_slot
// CHECK: %[[D:.*]] = memref.alloc() {ssbuffer.block_id = 2 : i32} : memref<32xf32>
// CHECK-NEXT: hivm.hir.copy ins(%{{.*}} : memref<32xf32>) outs(%[[D]] : memref<32xf32>) {ssbuffer.block_id = 2 : i32}
// CHECK-NEXT: bufferization.to_tensor %[[D]] restrict writable
// CHECK: %[[E:.*]] = memref.alloc() {ssbuffer.block_id = 3 : i32} : memref<32xf32>
// CHECK-NEXT: hivm.hir.copy ins(%{{.*}} : memref<32xf32>) outs(%[[E]] : memref<32xf32>) {ssbuffer.block_id = 3 : i32}
// CHECK-NEXT: bufferization.to_tensor %[[E]] restrict writable
// CHECK: %[[F:.*]] = memref.alloc() {ssbuffer.block_id = 4 : i32} : memref<32xf32>
// CHECK-NEXT: hivm.hir.copy ins(%{{.*}} : memref<32xf32>) outs(%[[F]] : memref<32xf32>) {ssbuffer.block_id = 4 : i32}
// CHECK-NEXT: bufferization.to_tensor %[[F]] restrict writable
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.intra_buf_count = 1 : i32} {
  func.func @outer_one_slot(%input: tensor<32xf32>, %condition: i1) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %end = arith.constant 8 : i32
    scope.scope : () -> () {
      scf.for %i = %zero to %end step %one : i32 {
        %m = arith.addf %input, %input {ssbuffer.block_id = 1 : i32} : tensor<32xf32>
        %alpha = math.exp %m {ssbuffer.block_id = 2 : i32} : tensor<32xf32>
        %later = arith.subf %m, %m {ssbuffer.block_id = 3 : i32} : tensor<32xf32>
        %carry = scf.if %condition -> tensor<32xf32> {
          scf.yield %input : tensor<32xf32>
        } else {
          scf.yield %m : tensor<32xf32>
        } {ssbuffer.block_id = 4 : i32}
      } {ssbuffer.main_loop = 1 : i64, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}

// -----

// Existing implicitly selected mainloops keep their previous representation.
// CHECK-LABEL: func.func @legacy_main_loop
// CHECK-NOT: memref.alloc() {ssbuffer.block_id
// CHECK: bufferization.to_tensor %{{.*}} restrict writable
// CHECK-NOT: memref.alloc() {ssbuffer.block_id
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.intra_buf_count = 1 : i32} {
  func.func @legacy_main_loop(%input: tensor<32xf32>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %end = arith.constant 8 : i32
    scope.scope : () -> () {
      scf.for %i = %zero to %end step %one : i32 {
        %m = arith.addf %input, %input {ssbuffer.block_id = 1 : i32} : tensor<32xf32>
        %alpha = math.exp %m {ssbuffer.block_id = 2 : i32} : tensor<32xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
