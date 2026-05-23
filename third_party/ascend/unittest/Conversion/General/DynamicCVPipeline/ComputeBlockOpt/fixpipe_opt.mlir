// RUN: triton-opt --fixpipe-opt %s | FileCheck %s

module {
  // Test 1: Basic fixpipe pattern - should match and set core_type to CUBE
  // CHECK-LABEL: func @test_basic_fixpipe_pattern
  func.func @test_basic_fixpipe_pattern(%arg0: memref<128x64xf16> {tt.divisibility = 16 : i32}, %A: tensor<64x64xf16>, %B: tensor<64x64xf16>, %init: tensor<64x64xf32>) {
    // CHECK: linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"}
    %matmul = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<64x64xf16>, tensor<64x64xf16>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"}
    %trunc = arith.truncf %matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"}
    %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"}
    %subview = memref.subview %arg0[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16> to memref<64x64xf16, strided<[64, 1]>>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"}
    bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1]>>) -> ()
    return
  }

  // Test 2: Nested subview chain from global memory - should match
  // CHECK-LABEL: func @test_nested_subview_from_global
  func.func @test_nested_subview_from_global(%arg0: memref<256x64xf16> {tt.divisibility = 16 : i32}, %A: tensor<64x64xf16>, %B: tensor<64x64xf16>, %init: tensor<64x64xf32>) {
    // CHECK: linalg.matmul {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"}
    %matmul = linalg.matmul {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<64x64xf16>, tensor<64x64xf16>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"}
    %trunc = arith.truncf %matmul {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"}
    %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"}
    %subview1 = memref.subview %arg0[0, 0] [128, 64] [1, 1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : memref<256x64xf16> to memref<128x64xf16, strided<[64, 1]>>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"}
    %subview2 = memref.subview %subview1[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1]>> to memref<64x64xf16, strided<[64, 1]>>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"}
    bufferization.materialize_in_destination %extract in writable %subview2 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1]>>) -> ()
    return
  }

  // Test 3: Reinterpret_cast chain from global memory - should match
  // CHECK-LABEL: func @test_reinterpret_cast_from_global
  func.func @test_reinterpret_cast_from_global(%arg0: memref<?xf16> {tt.divisibility = 16 : i32}, %A: tensor<64x64xf16>, %B: tensor<64x64xf16>, %init: tensor<64x64xf32>) {
    %c0 = arith.constant 0 : index
    // CHECK: linalg.matmul {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "CUBE"}
    %matmul = linalg.matmul {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<64x64xf16>, tensor<64x64xf16>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "CUBE"}
    %trunc = arith.truncf %matmul {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "CUBE"}
    %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: memref.reinterpret_cast %{{.*}} {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "CUBE"}
    %reinterpret = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "CUBE"}
    %subview = memref.subview %reinterpret[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "CUBE"}
    bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    return
  }

  // Test 4: Subview from scf.for iter arg (global memory) - should match
  // CHECK-LABEL: func @test_subview_from_for_iter_arg
  func.func @test_subview_from_for_iter_arg(%arg0: memref<128x64xf16> {tt.divisibility = 16 : i32}, %A: tensor<64x64xf16>, %B: tensor<64x64xf16>, %init: tensor<64x64xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    
    %result = scf.for %iv = %c0 to %c10 step %c1 iter_args(%memref = %arg0) -> (memref<128x64xf16>) {
      // CHECK: linalg.matmul {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "CUBE"}
      %matmul = linalg.matmul {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<64x64xf16>, tensor<64x64xf16>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
      // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"}
      %trunc = arith.truncf %matmul {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
      // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"}
      %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
      // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"}
      %subview = memref.subview %memref[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16> to memref<64x64xf16, strided<[64, 1]>>
      // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"}
      bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1]>>) -> ()
      
      scf.yield %memref : memref<128x64xf16>
    }
    
    return
  }

  // Test 5: No truncf op - should NOT match
  // CHECK-LABEL: func @test_no_truncf
  func.func @test_no_truncf(%arg0: memref<128x64xf16> {tt.divisibility = 16 : i32}, %A: tensor<64x64xf16>, %B: tensor<64x64xf16>, %init: tensor<64x64xf16>) {
    // CHECK: linalg.matmul {ssbuffer.block_id = 41 : i32, ssbuffer.core_type = "CUBE"}
    %matmul = linalg.matmul {ssbuffer.block_id = 41 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<64x64xf16>, tensor<64x64xf16>) outs(%init : tensor<64x64xf16>) -> tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 43 : i32, ssbuffer.core_type = "VECTOR"}
    %extract = tensor.extract_slice %matmul[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 43 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 44 : i32, ssbuffer.core_type = "VECTOR"}
    %subview = memref.subview %arg0[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 44 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16> to memref<64x64xf16, strided<[64, 1]>>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 45 : i32, ssbuffer.core_type = "VECTOR"}
    bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 45 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1]>>) -> ()
    return
  }

  // Test 6: Subview from alloc (not global memory) - should NOT match
  // CHECK-LABEL: func @test_subview_from_alloc
  func.func @test_subview_from_alloc(%A: tensor<64x64xf16>, %B: tensor<64x64xf16>, %init: tensor<64x64xf32>) {
    // CHECK: linalg.matmul {ssbuffer.block_id = 51 : i32, ssbuffer.core_type = "CUBE"}
    %matmul = linalg.matmul {ssbuffer.block_id = 51 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<64x64xf16>, tensor<64x64xf16>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 52 : i32, ssbuffer.core_type = "VECTOR"}
    %trunc = arith.truncf %matmul {ssbuffer.block_id = 52 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 53 : i32, ssbuffer.core_type = "VECTOR"}
    %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 53 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: memref.alloc() {ssbuffer.block_id = 54 : i32, ssbuffer.core_type = "VECTOR"}
    %alloc = memref.alloc() {ssbuffer.block_id = 54 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 55 : i32, ssbuffer.core_type = "VECTOR"}
    %subview = memref.subview %alloc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 55 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16> to memref<64x64xf16, strided<[64, 1]>>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 55 : i32, ssbuffer.core_type = "VECTOR"}
    bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 55 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1]>>) -> ()
    return
  }

  // Test 7: No extract_slice op - should NOT match
  // CHECK-LABEL: func @test_no_extract_slice
  func.func @test_no_extract_slice(%arg0: memref<128x64xf16> {tt.divisibility = 16 : i32}, %A: tensor<64x64xf16>, %B: tensor<64x64xf16>, %init: tensor<64x64xf32>) {
    // CHECK: linalg.matmul {ssbuffer.block_id = 61 : i32, ssbuffer.core_type = "CUBE"}
    %matmul = linalg.matmul {ssbuffer.block_id = 61 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<64x64xf16>, tensor<64x64xf16>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 62 : i32, ssbuffer.core_type = "VECTOR"}
    %trunc = arith.truncf %matmul {ssbuffer.block_id = 62 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 64 : i32, ssbuffer.core_type = "VECTOR"}
    %subview = memref.subview %arg0[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 64 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16> to memref<64x64xf16, strided<[64, 1]>>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 65 : i32, ssbuffer.core_type = "VECTOR"}
    bufferization.materialize_in_destination %trunc in writable %subview {ssbuffer.block_id = 65 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1]>>) -> ()
    return
  }

  // Test 8: No materialize_in_destination op - should NOT match
  // CHECK-LABEL: func @test_no_materialize
  func.func @test_no_materialize(%arg0: memref<128x64xf16> {tt.divisibility = 16 : i32}, %A: tensor<64x64xf16>, %B: tensor<64x64xf16>, %init: tensor<64x64xf32>) {
    // CHECK: linalg.matmul {ssbuffer.block_id = 71 : i32, ssbuffer.core_type = "CUBE"}
    %matmul = linalg.matmul {ssbuffer.block_id = 71 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<64x64xf16>, tensor<64x64xf16>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 72 : i32, ssbuffer.core_type = "VECTOR"}
    %trunc = arith.truncf %matmul {ssbuffer.block_id = 72 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 73 : i32, ssbuffer.core_type = "VECTOR"}
    %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 73 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    return
  }

  // Test 9: Destination is function argument (not from subview) - should NOT match
  // CHECK-LABEL: func @test_dest_not_subview
  func.func @test_dest_not_subview(%A: tensor<64x64xf16>, %B: tensor<64x64xf16>, %init: tensor<64x64xf32>, %dest: memref<64x64xf16, strided<[64, 1]>>) {
    // CHECK: linalg.matmul {ssbuffer.block_id = 81 : i32, ssbuffer.core_type = "CUBE"}
    %matmul = linalg.matmul {ssbuffer.block_id = 81 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<64x64xf16>, tensor<64x64xf16>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 82 : i32, ssbuffer.core_type = "CUBE"}
    %trunc = arith.truncf %matmul {ssbuffer.block_id = 82 : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 83 : i32, ssbuffer.core_type = "VECTOR"}
    %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 83 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 85 : i32}
    bufferization.materialize_in_destination %extract in writable %dest {ssbuffer.block_id = 85 : i32} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1]>>) -> ()
    return
  }

  // Test 10: Not linalg.matmul (using arithmetic ops) - should NOT match
  // CHECK-LABEL: func @test_not_matmul
  func.func @test_not_matmul(%arg0: memref<128x64xf16> {tt.divisibility = 16 : i32}, %A: tensor<64x64xf16>, %B: tensor<64x64xf16>, %init: tensor<64x64xf32>) {
    // CHECK: arith.extf %{{.*}} {ssbuffer.block_id = 91 : i32, ssbuffer.core_type = "VECTOR"}
    %extA = arith.extf %A {ssbuffer.block_id = 91 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf32>
    // CHECK: arith.extf %{{.*}} {ssbuffer.block_id = 91 : i32, ssbuffer.core_type = "VECTOR"}
    %extB = arith.extf %B {ssbuffer.block_id = 91 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf32>
    // CHECK: arith.addf %{{.*}} {ssbuffer.block_id = 91 : i32, ssbuffer.core_type = "VECTOR"}
    %add = arith.addf %extA, %extB {ssbuffer.block_id = 91 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 91 : i32, ssbuffer.core_type = "VECTOR"}
    %trunc = arith.truncf %add {ssbuffer.block_id = 91 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"}
    %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "CUBE"}
    %subview = memref.subview %arg0[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16> to memref<64x64xf16, strided<[64, 1]>>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 97 : i32}
    bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 97 : i32} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1]>>) -> ()
    return
  }

  // Test 11: Cyclic fixpipe chains - should NOT merge due to cycle
  // CHECK-LABEL: func @test_cyclic_fixpipe
  func.func @test_cyclic_fixpipe(%arg0: memref<128x64xf16> {tt.divisibility = 16 : i32}, %arg1: memref<?xf16> {tt.divisibility = 16 : i32}, %A: tensor<64x64xf16>, %B: tensor<64x64xf16>, %init: tensor<64x64xf32>, %init2: tensor<64x64xf32>) {
    %c0 = arith.constant 0 : index
    // CHECK: linalg.matmul {ssbuffer.block_id = 100 : i32, ssbuffer.core_type = "CUBE"}
    %matmul = linalg.matmul {ssbuffer.block_id = 100 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<64x64xf16>, tensor<64x64xf16>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK: arith.addf %{{.*}} {ssbuffer.block_id = 101 : i32, ssbuffer.core_type = "VECTOR"}
    %extA = arith.extf %A {ssbuffer.block_id = 101 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf32>
    %extB = arith.extf %B {ssbuffer.block_id = 101 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf32>
    %add = arith.addf %extA, %extB {ssbuffer.block_id = 101 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    // CHECK: memref.reinterpret_cast %{{.*}} {ssbuffer.block_id = 101 : i32, ssbuffer.core_type = "VECTOR"}
    %reinterpret = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 101 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 101 : i32, ssbuffer.core_type = "VECTOR"}
    %subview1 = memref.subview %reinterpret[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 101 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 101 : i32, ssbuffer.core_type = "VECTOR"}
    %trunc1 = arith.truncf %add {ssbuffer.block_id = 101 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 102 : i32, ssbuffer.core_type = "VECTOR"}
    %extract1 = tensor.extract_slice %trunc1[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 102 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 102 : i32, ssbuffer.core_type = "VECTOR"}
    bufferization.materialize_in_destination %extract1 in writable %subview1 {ssbuffer.block_id = 102 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 103 : i32, ssbuffer.core_type = "VECTOR"}
    %trunc2 = arith.truncf %matmul {ssbuffer.block_id = 103 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 104 : i32, ssbuffer.core_type = "VECTOR"}
    %extract2 = tensor.extract_slice %trunc2[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 104 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 103 : i32, ssbuffer.core_type = "VECTOR"}
    %subview2 = memref.subview %reinterpret[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 103 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 104 : i32, ssbuffer.core_type = "VECTOR"}
    bufferization.materialize_in_destination %extract2 in writable %subview2 {ssbuffer.block_id = 104 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    return
  }
}