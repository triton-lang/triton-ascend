// RUN: triton-opt --triton-to-linalg %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @copy_row
  tt.func public @copy_row(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %row_offset: i32) {
    %row_base = tt.addptr %src, %row_offset : !tt.ptr<f32>, i32
    %cols = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %row = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %src_base = tt.splat %row_base : !tt.ptr<f32> -> tensor<1x32x!tt.ptr<f32>>
    %src_row = tt.addptr %src_base, %row : tensor<1x32x!tt.ptr<f32>>, tensor<1x32xi32>
    // CHECK: memref.reinterpret_cast {{.*}} to offset: [%{{.*}}], sizes: [1, 32], strides: [32, 1]
    %values = tt.load %src_row : tensor<1x32x!tt.ptr<f32>>

    %four = arith.constant dense<4> : tensor<1x32xi32>
    %spaced_cols = arith.muli %row, %four : tensor<1x32xi32>
    %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<1x32x!tt.ptr<f32>>
    %dst_row = tt.addptr %dst_base, %spaced_cols : tensor<1x32x!tt.ptr<f32>>, tensor<1x32xi32>
    // CHECK: memref.reinterpret_cast {{.*}} to offset: [0], sizes: [1, 32], strides: [1, 4]
    tt.store %dst_row, %values : tensor<1x32x!tt.ptr<f32>>
    tt.return
  }

  // CHECK-LABEL: func.func @copy_plane
  tt.func public @copy_plane(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %rows = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %cols = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %eight = arith.constant dense<8> : tensor<4x1xi32>
    %row_offsets = arith.muli %row, %eight : tensor<4x1xi32>
    %r = tt.broadcast %row_offsets : tensor<4x1xi32> -> tensor<4x8xi32>
    %c = tt.broadcast %col : tensor<1x8xi32> -> tensor<4x8xi32>
    %offsets = arith.addi %r, %c : tensor<4x8xi32>
    %plane = tt.expand_dims %offsets {axis = 0 : i32} : tensor<4x8xi32> -> tensor<1x4x8xi32>
    %base = tt.splat %src : !tt.ptr<f32> -> tensor<1x4x8x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %plane : tensor<1x4x8x!tt.ptr<f32>>, tensor<1x4x8xi32>
    // CHECK: memref.reinterpret_cast {{.*}} to offset: [0], sizes: [1, 4, 8], strides: [32, 8, 1]
    %values = tt.load %ptrs : tensor<1x4x8x!tt.ptr<f32>>
    %out = tt.splat %dst : !tt.ptr<f32> -> tensor<1x4x8x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out, %plane : tensor<1x4x8x!tt.ptr<f32>>, tensor<1x4x8xi32>
    tt.store %out_ptrs, %values : tensor<1x4x8x!tt.ptr<f32>>
    tt.return
  }
}
