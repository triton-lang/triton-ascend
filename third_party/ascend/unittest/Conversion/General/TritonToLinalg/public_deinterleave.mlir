// RUN: triton-opt %s --triton-to-unstructure --bubble-up-operation --triton-to-linalg --canonicalize --cse --verify-each | FileCheck %s --implicit-check-not='!tt.ptr' --implicit-check-not=unrealized_conversion_cast

// Both load paths must use the memref materialized from public pointer facts.
// A stride-2 source becomes a contiguous wide load followed by lane extraction.
// CHECK-LABEL: func.func @unmasked_pair
// CHECK: memref.reinterpret_cast {{.*}} to offset: [64], sizes: [32], strides: [1]
// CHECK: tensor.extract_slice {{.*}}[0] [16] [2]
// CHECK: tensor.extract_slice {{.*}}[1] [16] [2]
// CHECK: return
module {
  tt.func public @unmasked_pair(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %two = arith.constant dense<2> : tensor<16xi32>
    %bias = arith.constant dense<64> : tensor<16xi32>
    %one = arith.constant dense<1> : tensor<16xi32>
    %half = arith.constant dense<16> : tensor<16xi32>
    %step = arith.muli %r, %two : tensor<16xi32>
    %offset = arith.addi %step, %bias : tensor<16xi32>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %even = tt.addptr %s, %offset : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %odd = tt.addptr %even, %one : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %x = tt.load %even : tensor<16x!tt.ptr<f32>>
    %y = tt.load %odd : tensor<16x!tt.ptr<f32>>
    %out0 = tt.addptr %d, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %out1 = tt.addptr %out0, %half : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %out0, %x : tensor<16x!tt.ptr<f32>>
    tt.store %out1, %y : tensor<16x!tt.ptr<f32>>
    tt.return
  }

  // The mask limits rows while covering the entire interleaved last axis.
  // CHECK-LABEL: func.func @masked_odd_rows
  // CHECK: memref.reinterpret_cast {{.*}} to offset: [0], sizes: [4, 16], strides: [16, 1]
  // CHECK: tensor.extract_slice {{.*}}[0, 1] [4, 8] [1, 2]
  // CHECK: return
  tt.func public @masked_odd_rows(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %c = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %width = arith.constant dense<16> : tensor<4xi32>
    %two = arith.constant dense<2> : tensor<8xi32>
    %one = arith.constant dense<1> : tensor<4x8xi32>
    %zero = arith.constant dense<0.0> : tensor<4x8xf32>
    %row = arith.muli %r, %width : tensor<4xi32>
    %col = arith.muli %c, %two : tensor<8xi32>
    %re = tt.expand_dims %row {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %ce = tt.expand_dims %col {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %rb = tt.broadcast %re : tensor<4x1xi32> -> tensor<4x8xi32>
    %cb = tt.broadcast %ce : tensor<1x8xi32> -> tensor<4x8xi32>
    %even = arith.addi %rb, %cb : tensor<4x8xi32>
    %offset = arith.addi %even, %one : tensor<4x8xi32>
    %limit = tt.splat %n : i32 -> tensor<4xi32>
    %valid = arith.cmpi slt, %r, %limit : tensor<4xi32>
    %me = tt.expand_dims %valid {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
    %mask = tt.broadcast %me : tensor<4x1xi1> -> tensor<4x8xi1>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
    %sp = tt.addptr %s, %offset : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %dp = tt.addptr %d, %offset : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %value = tt.load %sp, %mask, %zero : tensor<4x8x!tt.ptr<f32>>
    tt.store %dp, %value, %mask : tensor<4x8x!tt.ptr<f32>>
    tt.return
  }

  // Gathered rows still need buffer writeback. Complementary loads should
  // share one gathered wide tile and extract their lanes after the row loop.
  // CHECK-LABEL: func.func @discrete_pair
  // CHECK: scf.for
  // CHECK: ExtractedLoadOrStore
  // CHECK-NOT: ExtractedLoadOrStore
  // CHECK: tensor.extract_slice {{.*}}[0, 0] [4, 8] [1, 2]
  // CHECK-NOT: ExtractedLoadOrStore
  // CHECK: tensor.extract_slice {{.*}}[0, 1] [4, 8] [1, 2]
  // CHECK-NOT: ExtractedLoadOrStore
  // CHECK: return
  tt.func public @discrete_pair(%indices: !tt.ptr<i32>, %src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %c = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %row_width = arith.constant dense<24> : tensor<4xi32>
    %out_width = arith.constant dense<16> : tensor<4xi32>
    %two = arith.constant dense<2> : tensor<8xi32>
    %one = arith.constant dense<1> : tensor<4x8xi32>
    %half = arith.constant dense<8> : tensor<4x8xi32>
    %ib = tt.splat %indices : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %ip = tt.addptr %ib, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %rows = tt.load %ip : tensor<4x!tt.ptr<i32>>
    %row = arith.muli %rows, %row_width : tensor<4xi32>
    %col = arith.muli %c, %two : tensor<8xi32>
    %re = tt.expand_dims %row {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %ce = tt.expand_dims %col {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %rb = tt.broadcast %re : tensor<4x1xi32> -> tensor<4x8xi32>
    %cb = tt.broadcast %ce : tensor<1x8xi32> -> tensor<4x8xi32>
    %offset = arith.addi %rb, %cb : tensor<4x8xi32>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
    %even = tt.addptr %s, %offset : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %odd = tt.addptr %even, %one : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %x = tt.load %even : tensor<4x8x!tt.ptr<f32>>
    %y = tt.load %odd : tensor<4x8x!tt.ptr<f32>>
    %out_row = arith.muli %r, %out_width : tensor<4xi32>
    %ore = tt.expand_dims %out_row {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %oce = tt.expand_dims %c {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %orb = tt.broadcast %ore : tensor<4x1xi32> -> tensor<4x8xi32>
    %ocb = tt.broadcast %oce : tensor<1x8xi32> -> tensor<4x8xi32>
    %out_offset = arith.addi %orb, %ocb : tensor<4x8xi32>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
    %out0 = tt.addptr %d, %out_offset : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %out1 = tt.addptr %out0, %half : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    tt.store %out0, %x : tensor<4x8x!tt.ptr<f32>>
    tt.store %out1, %y : tensor<4x8x!tt.ptr<f32>>
    tt.return
  }
}
