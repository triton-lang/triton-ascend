// RUN: triton-opt --split-input-file --merge-cube-block %s | FileCheck %s

// Only the second matmul feeds the vector block. Trace its right multiplicand
// through both cube blocks to the producer, which keeps block ID 7.
// CHECK-LABEL: func.func @merge_chained_matmuls(
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 7 : i32
// CHECK: arith.negf {{.*}}ssbuffer.block_id = 7 : i32
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 7 : i32
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 7 : i32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @merge_chained_matmuls(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %init: tensor<16x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %product = linalg.matmul {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %neg = arith.negf %product {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x16xf32>
      %rhs = arith.addf %b, %neg {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x16xf32>
      %result = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %rhs : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %out = arith.negf %result {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf32>
    }
    return
  }
}

// -----

// The only path from the first matmul to the second crosses a vector block.
// CHECK-LABEL: func.func @keep_matmuls_with_vector_intermediate(
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 7 : i32
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 4 : i32
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 2 : i32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @keep_matmuls_with_vector_intermediate(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %init: tensor<16x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %product = linalg.matmul {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %rhs = arith.addf %b, %product {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf32>
      %result = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %rhs : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
    }
    return
  }
}

// -----

// The product feeds the next matmul directly and also reaches its accumulator
// through block 4. Merging 7 and 2 would create a cycle with block 4.
// CHECK-LABEL: func.func @keep_chained_matmuls_if_merge_creates_cycle(
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 7 : i32
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 4 : i32
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 2 : i32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @keep_chained_matmuls_if_merge_creates_cycle(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %init: tensor<16x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %product = linalg.matmul {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %acc = arith.addf %product, %init {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf32>
      %result = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%product, %b : tensor<16x16xf32>, tensor<16x16xf32>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    }
    return
  }
}
