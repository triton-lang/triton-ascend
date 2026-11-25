// RUN: triton-opt --op-classifier %s | FileCheck %s

module {
  // A vector load must prevent whole-loop CUBE loader colouring.
  // CHECK-LABEL: func.func @for_vector_load(
  // CHECK: %[[LOOP:.*]] = scf.for
  // CHECK: %[[PAGE:.*]] = scope.scope : () -> tensor<4x8xf16> {
  // CHECK: } {ssbuffer.core_type = "VECTOR", ssbuffer.load}
  // CHECK: %[[NEXT:.*]] = tensor.insert_slice %[[PAGE]] into {{.*}} {ssbuffer.core_type = "VECTOR"}
  // CHECK: scf.yield {ssbuffer.core_type = "VECTOR"} %[[NEXT]]
  // CHECK-NEXT: } {ssbuffer.core_type = "VECTOR"}
  // CHECK: linalg.matmul {ssbuffer.core_type = "CUBE"} ins(%[[LOOP]],
  func.func @for_vector_load(%gm: memref<4x8xf16>, %rhs: tensor<8x8xf16>, %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %empty = tensor.empty() : tensor<8x8xf16>
    %loaded = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %empty) -> tensor<8x8xf16> {
      %page = scope.scope : () -> tensor<4x8xf16> {
        %alloc = memref.alloc() : memref<4x8xf16>
        memref.copy %gm, %alloc : memref<4x8xf16> to memref<4x8xf16>
        %t = bufferization.to_tensor %alloc restrict writable : memref<4x8xf16> to tensor<4x8xf16>
        scope.return %t : tensor<4x8xf16>
      } {ssbuffer.core_type = "VECTOR", ssbuffer.load}
      %offset = arith.muli %i, %c4 : index
      %next = tensor.insert_slice %page into %acc[%offset, 0] [4, 8] [1, 1] : tensor<4x8xf16> into tensor<8x8xf16>
      scf.yield %next : tensor<8x8xf16>
    }
    %mm = linalg.matmul ins(%loaded, %rhs : tensor<8x8xf16>, tensor<8x8xf16>) outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %mm : tensor<8x8xf32>
  }

  // A cube load must allow whole-loop CUBE loader colouring.
  // CHECK-LABEL: func.func @for_cube_load(
  // CHECK: %[[LOOP:.*]] = scf.for
  // CHECK: %[[PAGE:.*]] = scope.scope : () -> tensor<4x8xf16> {
  // CHECK: } {ssbuffer.core_type = "CUBE", ssbuffer.load}
  // CHECK: %[[NEXT:.*]] = tensor.insert_slice %[[PAGE]] into {{.*}} {ssbuffer.core_type = "CUBE"}
  // CHECK: scf.yield {ssbuffer.core_type = "CUBE"} %[[NEXT]]
  // CHECK-NEXT: } {ssbuffer.core_type = "CUBE"}
  // CHECK: linalg.matmul {ssbuffer.core_type = "CUBE"} ins(%[[LOOP]],
  func.func @for_cube_load(%gm: memref<4x8xf16>, %rhs: tensor<8x8xf16>, %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %empty = tensor.empty() : tensor<8x8xf16>
    %loaded = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %empty) -> tensor<8x8xf16> {
      %page = scope.scope : () -> tensor<4x8xf16> {
        %alloc = memref.alloc() : memref<4x8xf16>
        memref.copy %gm, %alloc : memref<4x8xf16> to memref<4x8xf16>
        %t = bufferization.to_tensor %alloc restrict writable : memref<4x8xf16> to tensor<4x8xf16>
        scope.return %t : tensor<4x8xf16>
      } {ssbuffer.core_type = "CUBE", ssbuffer.load}
      %offset = arith.muli %i, %c4 : index
      %next = tensor.insert_slice %page into %acc[%offset, 0] [4, 8] [1, 1] : tensor<4x8xf16> into tensor<8x8xf16>
      scf.yield %next : tensor<8x8xf16>
    }
    %mm = linalg.matmul ins(%loaded, %rhs : tensor<8x8xf16>, tensor<8x8xf16>) outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %mm : tensor<8x8xf32>
  }

  // A unclassified load must allow whole-loop CUBE loader colouring.
  // CHECK-LABEL: func.func @for_unclassified_load(
  // CHECK: %[[LOOP:.*]] = scf.for
  // CHECK: %[[PAGE:.*]] = scope.scope : () -> tensor<4x8xf16> {
  // CHECK: } {ssbuffer.core_type = "CUBE", ssbuffer.load}
  // CHECK: %[[NEXT:.*]] = tensor.insert_slice %[[PAGE]] into {{.*}} {ssbuffer.core_type = "CUBE"}
  // CHECK: scf.yield {ssbuffer.core_type = "CUBE"} %[[NEXT]]
  // CHECK-NEXT: } {ssbuffer.core_type = "CUBE"}
  // CHECK: linalg.matmul {ssbuffer.core_type = "CUBE"} ins(%[[LOOP]],
  func.func @for_unclassified_load(%gm: memref<4x8xf16>, %rhs: tensor<8x8xf16>, %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %empty = tensor.empty() : tensor<8x8xf16>
    %loaded = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %empty) -> tensor<8x8xf16> {
      %page = scope.scope : () -> tensor<4x8xf16> {
        %alloc = memref.alloc() : memref<4x8xf16>
        memref.copy %gm, %alloc : memref<4x8xf16> to memref<4x8xf16>
        %t = bufferization.to_tensor %alloc restrict writable : memref<4x8xf16> to tensor<4x8xf16>
        scope.return %t : tensor<4x8xf16>
      } {ssbuffer.load}
      %offset = arith.muli %i, %c4 : index
      %next = tensor.insert_slice %page into %acc[%offset, 0] [4, 8] [1, 1] : tensor<4x8xf16> into tensor<8x8xf16>
      scf.yield %next : tensor<8x8xf16>
    }
    %mm = linalg.matmul ins(%loaded, %rhs : tensor<8x8xf16>, tensor<8x8xf16>) outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %mm : tensor<8x8xf32>
  }

  // The shared loader eligibility check must enforce the same restriction for while.
  // CHECK-LABEL: func.func @while_vector_load(
  // CHECK: %[[LOOP:.*]] = scf.while
  // CHECK: %[[PAGE:.*]] = scope.scope : () -> tensor<4x8xf16> {
  // CHECK: } {ssbuffer.core_type = "VECTOR", ssbuffer.load}
  // CHECK: %[[NEXT:.*]] = tensor.insert_slice %[[PAGE]] into {{.*}} {ssbuffer.core_type = "VECTOR"}
  // CHECK: scf.yield {ssbuffer.core_type = "VECTOR"} %[[NEXT]]
  // CHECK-NEXT: } attributes {ssbuffer.core_type = "VECTOR"}
  // CHECK: linalg.matmul {ssbuffer.core_type = "CUBE"} ins(%[[LOOP]],
  func.func @while_vector_load(%cond: i1, %gm: memref<4x8xf16>, %rhs: tensor<8x8xf16>, %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %empty = tensor.empty() : tensor<8x8xf16>
    %loaded = scf.while (%acc = %empty) : (tensor<8x8xf16>) -> tensor<8x8xf16> {
      scf.condition(%cond) %acc : tensor<8x8xf16>
    } do {
    ^bb0(%acc: tensor<8x8xf16>):
      %page = scope.scope : () -> tensor<4x8xf16> {
        %alloc = memref.alloc() : memref<4x8xf16>
        memref.copy %gm, %alloc : memref<4x8xf16> to memref<4x8xf16>
        %t = bufferization.to_tensor %alloc restrict writable : memref<4x8xf16> to tensor<4x8xf16>
        scope.return %t : tensor<4x8xf16>
      } {ssbuffer.core_type = "VECTOR", ssbuffer.load}
      %next = tensor.insert_slice %page into %acc[0, 0] [4, 8] [1, 1] : tensor<4x8xf16> into tensor<8x8xf16>
      scf.yield %next : tensor<8x8xf16>
    }
    %mm = linalg.matmul ins(%loaded, %rhs : tensor<8x8xf16>, tensor<8x8xf16>) outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %mm : tensor<8x8xf32>
  }
}
