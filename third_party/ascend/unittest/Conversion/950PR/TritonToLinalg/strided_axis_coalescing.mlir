// RUN: triton-opt %s --triton-control-flow-opt \
// RUN:                --triton-to-unstructure='compile-on-910-95=true force-simt-template=true' \
// RUN:                --triton-to-linalg='compile-on-910-95=true' \
// RUN:                --split-input-file \
// RUN: | FileCheck %s

// BlockPtrPattern: fold `pid % S` from the scalar block-pointer base into an
// inner [S] block dimension.
// CHECK-LABEL: module attributes {hacc.coalesce_axis = 0 : i32, hacc.coalesce_factor = 4 : i32
// CHECK-LABEL: func.func @block_ptr_h_axis
// CHECK-NOT:   arith.divsi %{{.*}}, %{{.*}} : i32 loc
// CHECK:       sizes: [16, 4]
// CHECK:       tensor<16x4xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @block_ptr_h_axis(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                   %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c128 = arith.constant 128 : i32
    %size = arith.constant 128 : i64
    %stride = arith.constant 4 : i64
    %h = arith.remsi %pid, %c4 : i32
    %batch = arith.divsi %pid, %c4 : i32
    %batch_offset = arith.muli %batch, %c128 : i32
    %src_batch = tt.addptr %src, %batch_offset : !tt.ptr<f32>, i32
    %dst_batch = tt.addptr %dst, %batch_offset : !tt.ptr<f32>, i32
    %src_base = tt.addptr %src_batch, %h : !tt.ptr<f32>, i32
    %dst_base = tt.addptr %dst_batch, %h : !tt.ptr<f32>, i32
    %src_ptr = tt.make_tensor_ptr %src_base, [%size], [%stride], [%c0]
              {order = array<i32: 0>} : <tensor<16xf32>>
    %val = tt.load %src_ptr {boundaryCheck = array<i32: 0>, padding = 1 : i32}
           : !tt.ptr<tensor<16xf32>>
    %sum = arith.addf %val, %val : tensor<16xf32>
    %dst_ptr = tt.make_tensor_ptr %dst_base, [%size], [%stride], [%c0]
              {order = array<i32: 0>} : <tensor<16xf32>>
    tt.store %dst_ptr, %sum : !tt.ptr<tensor<16xf32>>
    tt.return
  }
}

// -----
// AddPtrPattern: fold the H lane out of output_row in a jagged load plus dense
// scalar side-input kernel.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @dense_vec_jagged_h_fuse(%jagged: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                          %dense: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                          %offsets: !tt.ptr<i64> {tt.divisibility = 16 : i32},
                                          %d: i32 {tt.divisibility = 16 : i32},
                                          %out: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c31_i32 = arith.constant 31 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %pid = tt.get_program_id x : i32
    %grid_num = arith.addi %d, %c31_i32 : i32
    %grid_dim_col = arith.divsi %grid_num, %c32_i32 : i32
    %output_row = arith.divsi %pid, %grid_dim_col : i32
    %batch = arith.divsi %output_row, %c8_i32 : i32
    %h = arith.remsi %output_row, %c8_i32 : i32
    %group = arith.remsi %pid, %grid_dim_col : i32
    %col_base = arith.muli %group, %c32_i32 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %col_base_splat = tt.splat %col_base : i32 -> tensor<32xi32>
    %cols = arith.addi %col_base_splat, %range : tensor<32xi32>
    %begin_ptr = tt.addptr %offsets, %batch : !tt.ptr<i64>, i32
    %begin = tt.load %begin_ptr : !tt.ptr<i64>
    %batch_next = arith.addi %batch, %c1_i32 : i32
    %end_ptr = tt.addptr %offsets, %batch_next : !tt.ptr<i64>, i32
    %end = tt.load %end_ptr : !tt.ptr<i64>
    %dense_ptr = tt.addptr %dense, %output_row : !tt.ptr<f32>, i32
    %h_d = arith.muli %h, %d : i32
    %h_d_i64 = arith.extsi %h_d : i32 to i64
    %jagged_scalar_off = arith.addi %begin, %h_d_i64 : i64
    %jagged_ptr = tt.addptr %jagged, %jagged_scalar_off : !tt.ptr<f32>, i64
    %out_scalar_off = arith.muli %d, %output_row : i32
    %out_ptr = tt.addptr %out, %out_scalar_off : !tt.ptr<f32>, i32
    %len = arith.subi %end, %begin : i64
    %len_index = arith.index_cast %len : i64 to index
    %trip = arith.minsi %len_index, %c1 : index
    %d_splat = tt.splat %d : i32 -> tensor<32xi32>
    %col_mask = arith.cmpi slt, %cols, %d_splat : tensor<32xi32>
    %res:2 = scf.for %iv = %c0 to %trip step %c1 iter_args(%acc = %zero, %ptr = %jagged_ptr) -> (tensor<32xf32>, !tt.ptr<f32>) {
      %iv_i64 = arith.index_cast %iv : index to i64
      %scalar_ptr = tt.addptr %dense_ptr, %iv_i64 : !tt.ptr<f32>, i64
      %scalar = tt.load %scalar_ptr : !tt.ptr<f32>
      %jagged_splat = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
      %vec_ptr = tt.addptr %jagged_splat, %cols : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      %vec = tt.load %vec_ptr, %col_mask, %zero : tensor<32x!tt.ptr<f32>>
      %scalar_splat = tt.splat %scalar : f32 -> tensor<32xf32>
      %prod = arith.mulf %scalar_splat, %vec : tensor<32xf32>
      %sum = arith.addf %acc, %prod : tensor<32xf32>
      %next = tt.addptr %ptr, %c1_i32 : !tt.ptr<f32>, i32
      scf.yield %sum, %next : tensor<32xf32>, !tt.ptr<f32>
    }
    %out_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %store_ptr = tt.addptr %out_splat, %cols : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %store_ptr, %res#0, %col_mask : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: module attributes {hacc.coalesce_axis = 0 : i32, hacc.coalesce_factor = 8 : i32
// CHECK-LABEL: func.func @dense_vec_jagged_h_fuse
// CHECK-NOT:   scf.for
// CHECK:       %[[GRID0:.*]] = arith.divsi %{{.*}}, %{{.*}} : i32
// CHECK:       %[[BATCH0:.*]] = arith.divsi %{{.*}}, %[[GRID0]] : i32
// CHECK-NOT:   arith.divsi %[[BATCH0]], %{{.*}} : i32
// CHECK:       sizes: [8]
// CHECK:       sizes: [8, 32]
// CHECK:       strides: [{{.*}}, 1]
// CHECK:       tensor<8x32xf32>
// CHECK:       linalg.generic
// CHECK:       tensor<8x32xf32>

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @dense_vec_jagged_h_fuse_permuted_args(%dense: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                        %out: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                        %d: i32 {tt.divisibility = 16 : i32},
                                                        %offsets: !tt.ptr<i64> {tt.divisibility = 16 : i32},
                                                        %jagged: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c15_i32 = arith.constant 15 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %pid = tt.get_program_id x : i32
    %grid_num = arith.addi %d, %c15_i32 : i32
    %grid_dim_col = arith.divsi %grid_num, %c16_i32 : i32
    %output_row = arith.divsi %pid, %grid_dim_col : i32
    %batch = arith.divsi %output_row, %c4_i32 : i32
    %h = arith.remsi %output_row, %c4_i32 : i32
    %group = arith.remsi %pid, %grid_dim_col : i32
    %col_base = arith.muli %group, %c16_i32 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %col_base_splat = tt.splat %col_base : i32 -> tensor<16xi32>
    %cols = arith.addi %range, %col_base_splat : tensor<16xi32>
    %begin_ptr = tt.addptr %offsets, %batch : !tt.ptr<i64>, i32
    %begin = tt.load %begin_ptr : !tt.ptr<i64>
    %batch_next = arith.addi %c1_i32, %batch : i32
    %end_ptr = tt.addptr %offsets, %batch_next : !tt.ptr<i64>, i32
    %end = tt.load %end_ptr : !tt.ptr<i64>
    %dense_ptr = tt.addptr %dense, %output_row : !tt.ptr<f32>, i32
    %h_d = arith.muli %d, %h : i32
    %h_d_i64 = arith.extsi %h_d : i32 to i64
    %jagged_scalar_off = arith.addi %h_d_i64, %begin : i64
    %jagged_ptr = tt.addptr %jagged, %jagged_scalar_off : !tt.ptr<f32>, i64
    %out_scalar_off = arith.muli %output_row, %d : i32
    %out_ptr = tt.addptr %out, %out_scalar_off : !tt.ptr<f32>, i32
    %len = arith.subi %end, %begin : i64
    %len_index = arith.index_cast %len : i64 to index
    %trip = arith.minsi %len_index, %c1 : index
    %d_splat = tt.splat %d : i32 -> tensor<16xi32>
    %col_mask = arith.cmpi slt, %cols, %d_splat : tensor<16xi32>
    %res:2 = scf.for %iv = %c0 to %trip step %c1 iter_args(%acc = %zero, %ptr = %jagged_ptr) -> (tensor<16xf32>, !tt.ptr<f32>) {
      %iv_i64 = arith.index_cast %iv : index to i64
      %scalar_ptr = tt.addptr %dense_ptr, %iv_i64 : !tt.ptr<f32>, i64
      %scalar = tt.load %scalar_ptr : !tt.ptr<f32>
      %jagged_splat = tt.splat %ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %vec_ptr = tt.addptr %jagged_splat, %cols : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %vec = tt.load %vec_ptr, %col_mask, %zero : tensor<16x!tt.ptr<f32>>
      %scalar_splat = tt.splat %scalar : f32 -> tensor<16xf32>
      %prod = arith.mulf %vec, %scalar_splat : tensor<16xf32>
      %sum = arith.addf %prod, %acc : tensor<16xf32>
      %next = tt.addptr %ptr, %c1_i32 : !tt.ptr<f32>, i32
      scf.yield %sum, %next : tensor<16xf32>, !tt.ptr<f32>
    }
    %out_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %store_ptr = tt.addptr %out_splat, %cols : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %store_ptr, %res#0, %col_mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: module attributes {hacc.coalesce_axis = 0 : i32, hacc.coalesce_factor = 4 : i32
// CHECK-LABEL: func.func @dense_vec_jagged_h_fuse_permuted_args
// CHECK-NOT:   scf.for
// CHECK:       %[[GRID1:.*]] = arith.divsi %{{.*}}, %{{.*}} : i32
// CHECK:       %[[BATCH1:.*]] = arith.divsi %{{.*}}, %[[GRID1]] : i32
// CHECK-NOT:   arith.divsi %[[BATCH1]], %{{.*}} : i32
// CHECK:       sizes: [4]
// CHECK:       sizes: [4, 16]
// CHECK:       strides: [{{.*}}, 1]
// CHECK:       tensor<4x16xf32>
// CHECK:       linalg.generic
// CHECK:       tensor<4x16xf32>

// -----

// AddPtrPattern also handles dense row bases:
//   splat(base + output_row * D) + (range(0, BT) + group * BT)
// This uses the same launch decomposition as the jagged case but has no
// offsets-table begin/end loads.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @dense_vec_dense_h_fuse(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                         %scale: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                         %out: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                         %d: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c7_i32 = arith.constant 7 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %pid = tt.get_program_id x : i32
    %grid_num = arith.addi %d, %c7_i32 : i32
    %grid_dim_col = arith.divsi %grid_num, %c8_i32 : i32
    %output_row = arith.divsi %pid, %grid_dim_col : i32
    %batch = arith.divsi %output_row, %c2_i32 : i32
    %h = arith.remsi %output_row, %c2_i32 : i32
    %group = arith.remsi %pid, %grid_dim_col : i32
    %col_base = arith.muli %group, %c8_i32 : i32
    %range = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %col_base_splat = tt.splat %col_base : i32 -> tensor<8xi32>
    %cols = arith.addi %col_base_splat, %range : tensor<8xi32>
    %src_scalar_off = arith.muli %output_row, %d : i32
    %src_ptr = tt.addptr %src, %src_scalar_off : !tt.ptr<f32>, i32
    %scale_ptr = tt.addptr %scale, %batch : !tt.ptr<f32>, i32
    %scale_val = tt.load %scale_ptr : !tt.ptr<f32>
    %out_scalar_off = arith.muli %d, %output_row : i32
    %out_ptr = tt.addptr %out, %out_scalar_off : !tt.ptr<f32>, i32
    %d_splat = tt.splat %d : i32 -> tensor<8xi32>
    %col_mask = arith.cmpi slt, %cols, %d_splat : tensor<8xi32>
    %src_splat = tt.splat %src_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %src_vec_ptr = tt.addptr %src_splat, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %vec = tt.load %src_vec_ptr, %col_mask, %zero : tensor<8x!tt.ptr<f32>>
    %scale_splat = tt.splat %scale_val : f32 -> tensor<8xf32>
    %sum = arith.mulf %vec, %scale_splat : tensor<8xf32>
    %out_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %store_ptr = tt.addptr %out_splat, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %store_ptr, %sum, %col_mask : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: module attributes {hacc.coalesce_axis = 0 : i32, hacc.coalesce_factor = 2 : i32
// CHECK-LABEL: func.func @dense_vec_dense_h_fuse
// CHECK:       %[[GRID2:.*]] = arith.divsi %{{.*}}, %{{.*}} : i32
// CHECK:       %[[BATCH2:.*]] = arith.divsi %{{.*}}, %[[GRID2]] : i32
// CHECK-NOT:   arith.divsi %[[BATCH2]], %{{.*}} : i32
// CHECK:       sizes: [2, 8]
// CHECK:       strides: [{{.*}}, 1]
// CHECK:       tensor<2x8xf32>
