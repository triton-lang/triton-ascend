// RUN: triton-opt --allow-unregistered-dialect --merge-small-block %s | FileCheck %s

// Test merge-small-block pass: block 32 is a small VECTOR block (0 compute ops)
// that produces %91 and %92, consumed only by downstream VECTOR blocks 33 and
// 35. After the pass, all ops with ssbuffer.block_id = 32 must be merged into
// block 33 (chosen via IROrderStrategy as the earliest downstream block).

// CHECK-NOT: ssbuffer.block_id = 32 : i32
// CHECK: memref.reinterpret_cast {{.*}} {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
// CHECK: memref.alloc() {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : memref<128xf32>
// CHECK: memref.copy {{.*}} {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : memref<128xf32, strided<[1], offset: ?>> to memref<128xf32>
// CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : memref<128xf32> to tensor<128xf32>
// CHECK: memref.reinterpret_cast {{.*}} {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
// CHECK: memref.alloc() {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : memref<128xf32>
// CHECK: memref.copy {{.*}} {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : memref<128xf32, strided<[1], offset: ?>> to memref<128xf32>
// CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : memref<128xf32> to tensor<128xf32>

module {
  func.func @test_merge_small_block(
      %arg9: memref<?xf32>,
      %4: tensor<128x128xf32>,
      %10: tensor<128x128xf32>,
      %98: tensor<128x64xbf16>,
      %transposed: tensor<64x128xbf16>,
      %18: tensor<128x128xf32>,
      %105: tensor<128x64xbf16>,
      %arg21: tensor<128x64xf32>,
      %transposed_11: tensor<64x128xbf16>) {
    %90 = arith.constant 0 : index

    // ==========================================
    // Block 32 (VECTOR, small) — produces %91 and %92
    // After pass: all block_id 32 → 33
    // ==========================================
    %reinterpret_cast_12 = memref.reinterpret_cast %arg9 to offset: [%90], sizes: [128], strides: [1] {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    %alloc_13 = memref.alloc() {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : memref<128xf32>
    memref.copy %reinterpret_cast_12, %alloc_13 {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : memref<128xf32, strided<[1], offset: ?>> to memref<128xf32>
    %91 = bufferization.to_tensor %alloc_13 restrict writable {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : memref<128xf32> to tensor<128xf32>
    %reinterpret_cast_14 = memref.reinterpret_cast %arg9 to offset: [%90], sizes: [128], strides: [1] {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    %alloc_15 = memref.alloc() {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : memref<128xf32>
    memref.copy %reinterpret_cast_14, %alloc_15 {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : memref<128xf32, strided<[1], offset: ?>> to memref<128xf32>
    %92 = bufferization.to_tensor %alloc_15 restrict writable {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : memref<128xf32> to tensor<128xf32>

    %99 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%98, %transposed : tensor<128x64xbf16>, tensor<64x128xbf16>) outs(%18 : tensor<128x128xf32>) -> tensor<128x128xf32>

    // ==========================================
    // Block 33 (VECTOR) — consumes %91
    // ==========================================
    %broadcasted_19 = linalg.broadcast ins(%91 : tensor<128xf32>) outs(%4 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"}
    %100 = arith.mulf %99, %10 {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
    %101 = arith.subf %100, %broadcasted_19 {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
    %102 = math.exp %101 {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
    %103 = arith.truncf %102 {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32> to tensor<128x128xbf16>

    %106 = tensor.empty() {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xbf16>
    %transposed_22 = linalg.transpose ins(%103 : tensor<128x128xbf16>) outs(%106 : tensor<128x128xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"}
    %107 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%transposed_22, %105 : tensor<128x128xbf16>, tensor<128x64xbf16>) outs(%arg21 : tensor<128x64xf32>) -> tensor<128x64xf32>

    %108 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%105, %transposed_11 : tensor<128x64xbf16>, tensor<64x128xbf16>) outs(%18 : tensor<128x128xf32>) -> tensor<128x128xf32>

    // ==========================================
    // Block 35 (VECTOR) — consumes %92
    // ==========================================
    %104 = arith.extf %103 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xbf16> to tensor<128x128xf32>
    %broadcasted_23 = linalg.broadcast ins(%92 : tensor<128xf32>) outs(%4 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"}
    %109 = arith.subf %108, %broadcasted_23 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
    %110 = arith.mulf %104, %109 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
    %111 = arith.mulf %110, %10 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
    %112 = arith.truncf %111 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32> to tensor<128x128xbf16>

    return
  }

  // ==========================================
  // Test: small VECTOR block 29 (single arith.extf) merges into downstream
  // block 30 (the downstream user saves more UB than the upstream producer).
  // ==========================================
  // CHECK-LABEL: func.func @test_merge_small_block_extf
  func.func @test_merge_small_block_extf(
      %arg5: memref<?xbf16>,
      %2: tensor<64x128xf32>,
      %3: tensor<64x128xf32>,
      %9: tensor<64x128xf32>,
      %16: tensor<64x128xf32>,
      %93: tensor<64xf32>,
      %94: tensor<64xf32>,
      %95: tensor<64x128xi1>,
      %102: tensor<64x128xf32>,
    //   %112: tensor<64x128xf32>,
      %arg21: tensor<128x64xf32>,
      %transposed_11: tensor<64x128xbf16>) {
    %100 = arith.constant 0 : index

    // Block 28 (VECTOR) — produces %107
    %broadcasted_20 = linalg.broadcast ins(%93 : tensor<64xf32>) outs(%2 : tensor<64x128xf32>) dimensions = [1]  {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"}
    %103 = arith.mulf %102, %9 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xf32>
    %104 = arith.subf %103, %broadcasted_20 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xf32>
    %105 = math.exp %104 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xf32>
    // CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xi1>, tensor<64x128xf32>
    %106 = arith.select %95, %105, %3 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xi1>, tensor<64x128xf32>

    // Block 29 (VECTOR, small) — merges into block 28
    // CHECK-NEXT: arith.mulf %{{.*}} {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xf32>
    %108 = arith.mulf %106, %16 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xf32>

    %reinterpret_cast_21 = memref.reinterpret_cast %arg5 to offset: [%100], sizes: [64, 64], strides: [64, 1] {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
    %alloc_22 = memref.alloc() {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xbf16>
    memref.copy %reinterpret_cast_21, %alloc_22 {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
    %109 = bufferization.to_tensor %alloc_22 restrict writable {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xbf16> to tensor<64x64xbf16>
    %112 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%109, %transposed_11 : tensor<64x64xbf16>, tensor<64x128xbf16>) outs(%16 : tensor<64x128xf32>) -> tensor<64x128xf32>

    %broadcasted_24 = linalg.broadcast ins(%94 : tensor<64xf32>) outs(%2 : tensor<64x128xf32>) dimensions = [1]  {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"}
    %113 = arith.subf %112, %broadcasted_24 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xf32>
    %114 = arith.mulf %108, %113 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xf32>
    %115 = arith.mulf %114, %9 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xf32>
    %116 = arith.truncf %115 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xf32> to tensor<64x128xbf16>

    return
  }
}
