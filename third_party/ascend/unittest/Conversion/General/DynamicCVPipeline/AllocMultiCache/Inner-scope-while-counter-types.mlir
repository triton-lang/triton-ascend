// RUN: triton-opt --split-input-file --add_multi_buffer_inner_scope %s | FileCheck %s

// The pass appends an i32 iteration counter independently of the user while
// induction variable. Check that its increment and yielded value stay i32,
// while index, i32 and i64 user induction variables retain their own types.

// CHECK-LABEL: func.func @inner_counter_index
// CHECK: scf.while {{.*}} : (tensor<128xf32>, index, i32) -> (tensor<128xf32>, index, i32)
// CHECK: ^bb0(%{{[^:]+}}: tensor<128xf32>, %[[IV_INDEX:[^:]+]]: index, %[[COUNT_INDEX:[^:]+]]: i32):
// CHECK: %[[ONE_INDEX:[^ ]+]] = arith.constant {{.*}} 1 : i32
// CHECK-NEXT: %[[NEXT_COUNT_INDEX:[^ ]+]] = arith.addi %[[COUNT_INDEX]], %[[ONE_INDEX]] {{.*}}ssbuffer.iterCounter{{.*}} : i32
// CHECK-NEXT: %[[NEXT_IV_INDEX:[^ ]+]] = arith.addi %[[IV_INDEX]], %{{[^ ]+}} : index
// CHECK-NEXT: scf.yield %{{[^,]+}}, %[[NEXT_IV_INDEX]], %[[NEXT_COUNT_INDEX]] : tensor<128xf32>, index, i32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.intra_buf_count = 2 : i32} {
  func.func @inner_counter_index() {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %cst_zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128xf32>
    %carry = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<128xf32>) -> tensor<128xf32>
    scope.scope : () -> () {
      %result:2 = scf.while (%arg0 = %carry, %arg1 = %c0)
          : (tensor<128xf32>, index) -> (tensor<128xf32>, index) {
        %cmp = arith.cmpi slt, %arg1, %c10 {ssbuffer.block_id = 16 : i32} : index
        scf.condition(%cmp) %arg0, %arg1 : tensor<128xf32>, index
      } do {
      ^bb0(%arg0: tensor<128xf32>, %arg1: index):
        %alloc = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<128xf32>
        %prod = bufferization.to_tensor %alloc {ssbuffer.block_id = 7 : i32} : memref<128xf32> to tensor<128xf32>
        %consumed = arith.addf %prod, %prod {ssbuffer.block_id = 10 : i32} : tensor<128xf32>
        %next = arith.addi %arg1, %c1 : index
        scf.yield %consumed, %next : tensor<128xf32>, index
      } attributes {ssbuffer.main_loop = 1 : i64, ssbuffer.block_id = 23 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

}

// -----

// CHECK-LABEL: func.func @inner_counter_i32
// CHECK: scf.while {{.*}} : (tensor<128xf32>, i32, i32) -> (tensor<128xf32>, i32, i32)
// CHECK: ^bb0(%{{[^:]+}}: tensor<128xf32>, %[[IV_I32:[^:]+]]: i32, %[[COUNT_I32:[^:]+]]: i32):
// CHECK: %[[ONE_I32:[^ ]+]] = arith.constant {{.*}} 1 : i32
// CHECK-NEXT: %[[NEXT_COUNT_I32:[^ ]+]] = arith.addi %[[COUNT_I32]], %[[ONE_I32]] {{.*}}ssbuffer.iterCounter{{.*}} : i32
// CHECK-NEXT: %[[NEXT_IV_I32:[^ ]+]] = arith.addi %[[IV_I32]], %{{[^ ]+}} : i32
// CHECK-NEXT: scf.yield %{{[^,]+}}, %[[NEXT_IV_I32]], %[[NEXT_COUNT_I32]] : tensor<128xf32>, i32, i32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.intra_buf_count = 2 : i32} {
  func.func @inner_counter_i32() {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c1 = arith.constant 1 : i32
    %cst_zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128xf32>
    %carry = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<128xf32>) -> tensor<128xf32>
    scope.scope : () -> () {
      %result:2 = scf.while (%arg0 = %carry, %arg1 = %c0)
          : (tensor<128xf32>, i32) -> (tensor<128xf32>, i32) {
        %cmp = arith.cmpi slt, %arg1, %c10 {ssbuffer.block_id = 16 : i32} : i32
        scf.condition(%cmp) %arg0, %arg1 : tensor<128xf32>, i32
      } do {
      ^bb0(%arg0: tensor<128xf32>, %arg1: i32):
        %alloc = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<128xf32>
        %prod = bufferization.to_tensor %alloc {ssbuffer.block_id = 7 : i32} : memref<128xf32> to tensor<128xf32>
        %consumed = arith.addf %prod, %prod {ssbuffer.block_id = 10 : i32} : tensor<128xf32>
        %next = arith.addi %arg1, %c1 : i32
        scf.yield %consumed, %next : tensor<128xf32>, i32
      } attributes {ssbuffer.main_loop = 1 : i64, ssbuffer.block_id = 23 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

}

// -----

// CHECK-LABEL: func.func @inner_counter_i64
// CHECK: scf.while {{.*}} : (tensor<128xf32>, i64, i32) -> (tensor<128xf32>, i64, i32)
// CHECK: ^bb0(%{{[^:]+}}: tensor<128xf32>, %[[IV_I64:[^:]+]]: i64, %[[COUNT_I64:[^:]+]]: i32):
// CHECK: %[[ONE_I64:[^ ]+]] = arith.constant {{.*}} 1 : i32
// CHECK-NEXT: %[[NEXT_COUNT_I64:[^ ]+]] = arith.addi %[[COUNT_I64]], %[[ONE_I64]] {{.*}}ssbuffer.iterCounter{{.*}} : i32
// CHECK-NEXT: %[[NEXT_IV_I64:[^ ]+]] = arith.addi %[[IV_I64]], %{{[^ ]+}} : i64
// CHECK-NEXT: scf.yield %{{[^,]+}}, %[[NEXT_IV_I64]], %[[NEXT_COUNT_I64]] : tensor<128xf32>, i64, i32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.intra_buf_count = 2 : i32} {
  func.func @inner_counter_i64() {
    %c0 = arith.constant 0 : i64
    %c10 = arith.constant 10 : i64
    %c1 = arith.constant 1 : i64
    %cst_zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128xf32>
    %carry = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<128xf32>) -> tensor<128xf32>
    scope.scope : () -> () {
      %result:2 = scf.while (%arg0 = %carry, %arg1 = %c0)
          : (tensor<128xf32>, i64) -> (tensor<128xf32>, i64) {
        %cmp = arith.cmpi slt, %arg1, %c10 {ssbuffer.block_id = 16 : i32} : i64
        scf.condition(%cmp) %arg0, %arg1 : tensor<128xf32>, i64
      } do {
      ^bb0(%arg0: tensor<128xf32>, %arg1: i64):
        %alloc = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<128xf32>
        %prod = bufferization.to_tensor %alloc {ssbuffer.block_id = 7 : i32} : memref<128xf32> to tensor<128xf32>
        %consumed = arith.addf %prod, %prod {ssbuffer.block_id = 10 : i32} : tensor<128xf32>
        %next = arith.addi %arg1, %c1 : i64
        scf.yield %consumed, %next : tensor<128xf32>, i64
      } attributes {ssbuffer.main_loop = 1 : i64, ssbuffer.block_id = 23 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

}
