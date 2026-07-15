// RUN: triton-opt --async-load-hoisting %s | FileCheck %s

// CHECK-LABEL: func.func @test_gm_load_cube_marked
// Test 1: GM loads feeding Cube compute should be marked
// GM loads feeding linalg.matmul (Cube compute) need gm_load_bufferable mark
// Complex scene with subview: memref.copy from subview to subview
func.func @test_gm_load_cube_marked(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg1: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg2: tensor<128x128xf32>
) {
  %c0_i32 = arith.constant {ssbuffer.block_id = 5 : i32} 0 : i32
  %c128_i32 = arith.constant {ssbuffer.block_id = 5 : i32} 128 : i32
  %c0 = arith.constant {ssbuffer.block_id = 5 : i32} 0 : index
  %c1 = arith.constant {ssbuffer.block_id = 5 : i32} 1 : index

  // Load Q from GM - feeds linalg.matmul (Cube compute)
  // Complex scene: use subview for copy
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 5 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
  %alloc = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<128x128xf16>
  // subview copy: from reinterpret_cast subview to alloc subview
  %subview_src = memref.subview %reinterpret_cast[0, 0] [128, 128] [1, 1] {ssbuffer.block_id = 5 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16, strided<[128, 1], offset: ?>>
  %subview_dst = memref.subview %alloc[0, 0] [128, 128] [1, 1] {ssbuffer.block_id = 5 : i32} : memref<128x128xf16> to memref<128x128xf16, strided<[128, 1]>>
  memref.copy %subview_src, %subview_dst {ssbuffer.block_id = 5 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16, strided<[128, 1]>>
  // CHECK: bufferization.to_tensor
  // CHECK-SAME: gm_load_bufferable
  %to_tensor = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 5 : i32} : memref<128x128xf16> to tensor<128x128xf16>

  // Load K from GM - feeds linalg.matmul (Cube compute)
  // Complex scene: use subview for copy, similar to real code pattern
  %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 5 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
  %alloc_2 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<128x128xf16>
  // subview copy: from reinterpret_cast subview to alloc subview
  %subview_src_2 = memref.subview %reinterpret_cast_2[0, 0] [128, 128] [1, 1] {ssbuffer.block_id = 5 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16, strided<[128, 1], offset: ?>>
  %subview_dst_2 = memref.subview %alloc_2[0, 0] [128, 128] [1, 1] {ssbuffer.block_id = 5 : i32} : memref<128x128xf16> to memref<128x128xf16, strided<[128, 1]>>
  memref.copy %subview_src_2, %subview_dst_2 {ssbuffer.block_id = 5 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16, strided<[128, 1]>>
  // CHECK: bufferization.to_tensor
  // CHECK-SAME: gm_load_bufferable
  %to_tensor_2 = bufferization.to_tensor %alloc_2 restrict writable {ssbuffer.block_id = 5 : i32} : memref<128x128xf16> to tensor<128x128xf16>

  // Cube compute: matmul (128x128) x (128x128) = (128x128)
  %matmul = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 5 : i32} ins(%to_tensor, %to_tensor_2 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%arg2 : tensor<128x128xf32>) -> tensor<128x128xf32>

  return
}

// CHECK-LABEL: func.func @test_gm_load_mask_not_marked
// Test 2: GM loads feeding arith mask operations (maximumf/minimumf/select) should NOT be marked
// Element-wise mask ops do not feed Cube, should not be marked
func.func @test_gm_load_mask_not_marked(
  %arg0: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg1: tensor<128xf32>
) {
  %c0 = arith.constant {ssbuffer.block_id = 7 : i32} 0 : index

  // Load from GM feeding arith.maximumf (mask operation) - should NOT be marked
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128], strides: [1] {ssbuffer.block_id = 7 : i32} : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
  %alloc = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<128xf32>
  memref.copy %reinterpret_cast, %alloc {ssbuffer.block_id = 7 : i32} : memref<128xf32, strided<[1], offset: ?>> to memref<128xf32>
  // CHECK-NOT: gm_load_bufferable
  %to_tensor = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 7 : i32} : memref<128xf32> to tensor<128xf32>

  // arith.maximumf is a mask operation - result not fed to Cube
  %max = arith.maximumf %to_tensor, %arg1 {ssbuffer.block_id = 7 : i32} : tensor<128xf32>

  return
}

// CHECK-LABEL: func.func @test_gm_load_scalar_index_not_marked
// Test 3: memref.load with scalar (constant) indices should NOT be marked
// Scalar index load does not mark - memref.load with constant index is not marked
func.func @test_gm_load_scalar_index_not_marked(
  %arg0: memref<128x128xf16>,
  %arg1: tensor<128x128xf32>
) {
  %c0 = arith.constant {ssbuffer.block_id = 10 : i32} 0 : index

  // memref.load with scalar indices (constant %c0) - should NOT be marked
  // This is reading a scalar value from memory, not a GM data tile
  %loaded = memref.load %arg0[%c0, %c0] {ssbuffer.block_id = 10 : i32} : memref<128x128xf16>

  // The loaded scalar value used in element-wise computation
  %ext = arith.extf %loaded {ssbuffer.block_id = 10 : i32} : f16 to f32
  %add = arith.addf %ext, %ext {ssbuffer.block_id = 10 : i32} : f32

  // CHECK-NOT: gm_load_bufferable
  return
}

// CHECK-LABEL: func.func @test_gm_load_nonlinear_cast_not_marked
// Test 4: GM loads with non-linear reinterpret_cast offset should NOT be marked
// Address that cannot be linearly projected is not marked - reinterpret_cast offset depends on:
// 1. memref.load reading index as offset
// 2. scf.for/scf.while iter_arg drives address and yield value is non-linear
// Complex scene with subview, result feeds Cube (not mask), exclude scenario 2 interference
func.func @test_gm_load_nonlinear_cast_not_marked(
  %arg0: memref<1xi64>,
  %arg1: memref<128x128xf16>,
  %arg2: tensor<32x128xf16>
) {
  %c0 = arith.constant {ssbuffer.block_id = 8 : i32} 0 : index
  %c1 = arith.constant {ssbuffer.block_id = 8 : i32} 1 : index
  %c128_i64 = arith.constant {ssbuffer.block_id = 8 : i32} 128 : i64
  %c128 = arith.constant {ssbuffer.block_id = 8 : i32} 128 : index
  %c32 = arith.constant {ssbuffer.block_id = 8 : i32} 32 : index

  // ========== Scene 4a: memref.load causes non-linear offset ==========
  // Step 1: memref.load reads control flow indices from %arg0 (not GM data)
  %reinterpret_cast_idx = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [1], strides: [1] {ssbuffer.block_id = 8 : i32} : memref<1xi64> to memref<1xi64, strided<[1], offset: ?>>
  %loaded_idx = memref.load %reinterpret_cast_idx[%c0] {ssbuffer.block_id = 8 : i32} : memref<1xi64, strided<[1], offset: ?>>

  // Step 2: Compute offset using the loaded index - this makes address non-linear
  %multiplied = arith.muli %loaded_idx, %c128_i64 {ssbuffer.block_id = 8 : i32} : i64
  %base_offset = arith.index_cast %multiplied {ssbuffer.block_id = 8 : i32} : i64 to index

  // Step 3: reinterpret_cast offset depends on memref.load result - NON-LINEAR
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%base_offset], sizes: [32, 128], strides: [128, 1] {ssbuffer.block_id = 8 : i32} : memref<128x128xf16> to memref<32x128xf16, strided<[128, 1], offset: ?>>

  // Step 4: Allocate buffer
  %alloc = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<32x128xf16>

  // Step 5: Create subviews
  %subview_src = memref.subview %reinterpret_cast[0, 0] [%c32, 128] [1, 1] {ssbuffer.block_id = 8 : i32} : memref<32x128xf16, strided<[128, 1], offset: ?>> to memref<?x128xf16, strided<[128, 1], offset: ?>>
  %subview_dst = memref.subview %alloc[0, 0] [%c32, 128] [1, 1] {ssbuffer.block_id = 8 : i32} : memref<32x128xf16> to memref<?x128xf16, strided<[128, 1]>>

  // Step 6: memref.copy through subviews
  memref.copy %subview_src, %subview_dst {ssbuffer.block_id = 8 : i32} : memref<?x128xf16, strided<[128, 1], offset: ?>> to memref<?x128xf16, strided<[128, 1]>>

  // Step 7: bufferization.to_tensor - should NOT be marked because offset is non-linear (memref.load)
  // CHECK-NOT: gm_load_bufferable
  %to_tensor = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 8 : i32} : memref<32x128xf16> to tensor<32x128xf16>

  // ========== Scene 4b: scf.for iter_arg causes non-linear offset ==========
  %for_result:2 = scf.for %i = %c0 to %c128 step %c1 iter_args(%arg_iter = %arg1, %alloc_iter = %alloc) -> (memref<128x128xf16>, memref<32x128xf16>) {
    %for_offset = arith.index_cast %i {ssbuffer.block_id = 8 : i32} : index to i64
    %for_offset_index = arith.index_cast %for_offset {ssbuffer.block_id = 8 : i32} : i64 to index
    %reinterpret_for = memref.reinterpret_cast %arg_iter to offset: [%for_offset_index], sizes: [32, 128], strides: [128, 1] {ssbuffer.block_id = 8 : i32} : memref<128x128xf16> to memref<32x128xf16, strided<[128, 1], offset: ?>>
    %alloc_for = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<32x128xf16>
    memref.copy %reinterpret_for, %alloc_for {ssbuffer.block_id = 8 : i32} : memref<32x128xf16, strided<[128, 1], offset: ?>> to memref<32x128xf16>
    scf.yield %arg_iter, %alloc_for : memref<128x128xf16>, memref<32x128xf16>
  }
  // CHECK-NOT: gm_load_bufferable
  %to_tensor_for = bufferization.to_tensor %for_result#1 restrict writable {ssbuffer.block_id = 8 : i32} : memref<32x128xf16> to tensor<32x128xf16>

  // ========== Scene 4c: scf.while causes non-linear offset ==========
  %while_result:2 = scf.while (%arg_while = %arg1, %arg_while_alloc = %alloc) : (memref<128x128xf16>, memref<32x128xf16>) -> (memref<128x128xf16>, memref<32x128xf16>) {
    %cond = arith.constant {ssbuffer.block_id = 8 : i32} true
    scf.condition(%cond) %arg_while, %arg_while_alloc : memref<128x128xf16>, memref<32x128xf16>
  } do {
  ^bb0(%arg_while_prev: memref<128x128xf16>, %arg_while_alloc_prev: memref<32x128xf16>):
    %while_offset_base = arith.constant {ssbuffer.block_id = 8 : i32} 0 : index
    %reinterpret_while = memref.reinterpret_cast %arg_while_prev to offset: [%while_offset_base], sizes: [32, 128], strides: [128, 1] {ssbuffer.block_id = 8 : i32} : memref<128x128xf16> to memref<32x128xf16, strided<[128, 1], offset: ?>>
    %alloc_while = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<32x128xf16>
    memref.copy %reinterpret_while, %alloc_while {ssbuffer.block_id = 8 : i32} : memref<32x128xf16, strided<[128, 1], offset: ?>> to memref<32x128xf16>
    scf.yield %arg_while_prev, %alloc_while : memref<128x128xf16>, memref<32x128xf16>
  }
  // CHECK-NOT: gm_load_bufferable
  %to_tensor_while = bufferization.to_tensor %while_result#1 restrict writable {ssbuffer.block_id = 8 : i32} : memref<32x128xf16> to tensor<32x128xf16>

  // ========== Feeds Cube compute ==========
  %add = arith.addf %to_tensor, %to_tensor {ssbuffer.block_id = 8 : i32} : tensor<32x128xf16>

  return
}

// CHECK-LABEL: func.func @test_gm_load_non_gm_source_not_marked
// Test 5: bufferization.to_tensor with non-GM source should NOT be marked
// Data from non-GM is not marked - readback from internal compute (non-GM load) is not marked
func.func @test_gm_load_non_gm_source_not_marked(
  %arg0: tensor<32x256xf32>
) {
  // Step 1: Allocate a new buffer (NOT loaded from GM)
  %alloc = memref.alloc() {ssbuffer.block_id = 26 : i32} : memref<32x256xf32>

  // Step 2: bufferization.to_tensor from internal alloc (NOT from GM) - should NOT be marked
  // CHECK-NOT: gm_load_bufferable
  %to_tensor = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 26 : i32} : memref<32x256xf32> to tensor<32x256xf32>

  // The loaded data is used in element-wise operations (not Cube compute)
  %broadcasted = linalg.broadcast ins(%to_tensor : tensor<32x256xf32>) outs(%arg0 : tensor<32x256xf32>) dimensions = [] {ssbuffer.block_id = 26 : i32}

  return
}

// CHECK-LABEL: func.func @test_gm_load_cross_block_marked
// Test 6: Cross-block chain with block argument source should be marked
// Cross-block address generation chain but source is block argument (%arg), should be marked
// Scene: bufferization.to_tensor -> memref.copy -> memref.subview -> reinterpret_cast -> %arg
// alloc's block_id differs from copy's block_id, but since source is %arg, hasBlockArg=true
func.func @test_gm_load_cross_block_marked(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg1: tensor<128x128xf32>
) {
  %c0_i32 = arith.constant {ssbuffer.block_id = 10 : i32} 0 : i32
  %c128_i32 = arith.constant {ssbuffer.block_id = 10 : i32} 128 : i32
  %c0 = arith.constant {ssbuffer.block_id = 10 : i32} 0 : index
  %c1 = arith.constant {ssbuffer.block_id = 10 : i32} 1 : index

  // GM load chain: bufferization.to_tensor -> copy -> subview -> reinterpret_cast -> %arg0
  // Note: alloc block_id=10, copy block_id=11, subview block_id=11
  // But since source is %arg0 (block argument), hasBlockArg=true, should be marked
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 11 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
  %alloc = memref.alloc() {ssbuffer.block_id = 10 : i32} : memref<128x128xf16>
  %subview_src = memref.subview %reinterpret_cast[0, 0] [128, 128] [1, 1] {ssbuffer.block_id = 11 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16, strided<[128, 1], offset: ?>>
  %subview_dst = memref.subview %alloc[0, 0] [128, 128] [1, 1] {ssbuffer.block_id = 11 : i32} : memref<128x128xf16> to memref<128x128xf16, strided<[128, 1]>>
  memref.copy %subview_src, %subview_dst {ssbuffer.block_id = 11 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16, strided<[128, 1]>>
  // CHECK: bufferization.to_tensor
  // CHECK-SAME: gm_load_bufferable
  %0 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 11 : i32} : memref<128x128xf16> to tensor<128x128xf16>

  return
}
