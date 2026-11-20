// RUN: triton-opt --debug-only=memory-effects-tracker --plan-compute-block %s 2>&1 | FileCheck %s

module {
  // Two views and two reads of one argument must not introduce memory writers.
  // CHECK-LABEL: Analyzing op func.func @view_reads_have_no_memory_producer
  // CHECK: Analyzing op %{{[A-Za-z0-9_]+}} = memref.load
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] Analyzing op %{{[A-Za-z0-9_]+}} = memref.reinterpret_cast
  // CHECK: Analyzing op %{{[A-Za-z0-9_]+}} = memref.load
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] Analyzing op func.return
  func.func @view_reads_have_no_memory_producer(%arg: memref<?xi32>, %offset: index) -> (i32, i32) {
    %c0 = arith.constant 0 : index
    %lhs = memref.reinterpret_cast %arg to offset: [0], sizes: [4], strides: [1]
      : memref<?xi32> to memref<4xi32, strided<[1]>>
    %a = memref.load %lhs[%c0] : memref<4xi32, strided<[1]>>
    %rhs = memref.reinterpret_cast %arg to offset: [%offset], sizes: [4], strides: [1]
      : memref<?xi32> to memref<4xi32, strided<[1], offset: ?>>
    %b = memref.load %rhs[%c0] : memref<4xi32, strided<[1], offset: ?>>
    return %a, %b : i32, i32
  }

  // Defining a new view after a write must preserve the real writer (RAW).
  // CHECK-LABEL: Analyzing op func.func @view_preserves_raw
  // CHECK: Analyzing op memref.store %{{[A-Za-z0-9_]+}}, %[[RAW_VIEW:[A-Za-z0-9_]+]][
  // CHECK: Analyzing op %{{[A-Za-z0-9_]+}} = memref.load
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] memref.store {{.*}}, %[[RAW_VIEW]][
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] memref.store {{.*}}, %[[RAW_VIEW]][
  // CHECK-NEXT: [memory-effects-tracker] Analyzing op func.return
  func.func @view_preserves_raw(%arg: memref<8xi32>, %value: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %lhs = memref.subview %arg[0] [4] [1]
      : memref<8xi32> to memref<4xi32, strided<[1]>>
    memref.store %value, %lhs[%c2] : memref<4xi32, strided<[1]>>
    %rhs = memref.subview %arg[2] [4] [1]
      : memref<8xi32> to memref<4xi32, strided<[1], offset: 2>>
    %v = memref.load %rhs[%c0] : memref<4xi32, strided<[1], offset: 2>>
    return %v : i32
  }

  // Defining a view after a read must preserve the reader before a write (WAR).
  // CHECK-LABEL: Analyzing op func.func @view_preserves_war
  // CHECK: Analyzing op %[[WAR_READ:[A-Za-z0-9_]+]] = memref.load
  // CHECK: Analyzing op memref.store
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] %[[WAR_READ]] = memref.load
  // CHECK-NEXT: [memory-effects-tracker] Analyzing op func.return
  func.func @view_preserves_war(%arg: memref<8xi32>, %value: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %lhs = memref.subview %arg[0] [4] [1]
      : memref<8xi32> to memref<4xi32, strided<[1]>>
    %v = memref.load %lhs[%c2] : memref<4xi32, strided<[1]>>
    %rhs = memref.subview %arg[2] [4] [1]
      : memref<8xi32> to memref<4xi32, strided<[1], offset: 2>>
    memref.store %value, %rhs[%c0] : memref<4xi32, strided<[1], offset: 2>>
    return %v : i32
  }

  // Defining a view between writes must not replace the first writer (WAW).
  // CHECK-LABEL: Analyzing op func.func @view_preserves_waw
  // CHECK: Analyzing op memref.store %{{[A-Za-z0-9_]+}}, %[[WAW_VIEW:[A-Za-z0-9_]+]][
  // CHECK: Analyzing op memref.store
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] memref.store {{.*}}, %[[WAW_VIEW]][
  // CHECK-NEXT: [memory-effects-tracker] Analyzing op func.return
  func.func @view_preserves_waw(%arg: memref<8xi32>, %value: i32) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %lhs = memref.subview %arg[0] [4] [1]
      : memref<8xi32> to memref<4xi32, strided<[1]>>
    memref.store %value, %lhs[%c2] : memref<4xi32, strided<[1]>>
    %rhs = memref.subview %arg[2] [4] [1]
      : memref<8xi32> to memref<4xi32, strided<[1], offset: 2>>
    memref.store %value, %rhs[%c0] : memref<4xi32, strided<[1], offset: 2>>
    return
  }
}
