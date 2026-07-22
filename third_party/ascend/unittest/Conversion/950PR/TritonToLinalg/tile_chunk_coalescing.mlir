// RUN: triton-opt %s --triton-to-unstructure='compile-on-910-95=true force-simt-template=true' \
// RUN:                --triton-to-linalg='compile-on-910-95=true' --split-input-file \
// RUN: | FileCheck %s

// -----
// Adjacent 16-f32 tiles form a small contiguous DMA. The pass should merge
// 16 tiles per program, drop the all-true tile mask, and record the launch-grid
// shrink metadata on the tile program-id axis.
// CHECK-LABEL: module attributes {hacc.coalesce_axis = 0 : i32, hacc.coalesce_factor = 16 : i32
// CHECK-LABEL: func.func @tile_chunk_coalesce_simple
// CHECK: memref.reinterpret_cast
// CHECK-SAME: sizes: [16, 16]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_coalesce_simple(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                             %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %c256 = arith.constant dense<256> : tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %mask = arith.cmpi slt, %offs, %c256 : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<16x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Unmasked kernels do not carry a static tile count in the IR. The pass cannot
// prove runtime grid[axis] is >= H and divisible by H, so it must leave the
// kernel uncoalesced.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: func.func @tile_chunk_skip_unmasked_unknown_grid
// CHECK-NOT: sizes: [16, 16]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_skip_unmasked_unknown_grid(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                        %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// A one-tile mask has BOUND == tileLen. It is provably all true for the only
// tile, but the static tile count is one, so there is nothing to coalesce.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: func.func @tile_chunk_single_full_tile
// CHECK-NOT: sizes: [16, 16]
// CHECK: sizes: [16]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_single_full_tile(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                              %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %c16_tensor = arith.constant dense<16> : tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %mask = arith.cmpi slt, %offs, %c16_tensor : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<16x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Reading num_programs on the coalesced axis is unsafe because the host launcher
// divides that grid dimension by H. The pass must leave the kernel uncoalesced.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: func.func @tile_chunk_reads_num_programs
// CHECK-NOT: sizes: [16, 16]
// CHECK: sizes: [16]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_reads_num_programs(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %num = tt.get_num_programs x : i32
    %c16 = arith.constant 16 : i32
    %c512 = arith.constant dense<512> : tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %num_splat = tt.splat %num : i32 -> tensor<16xi32>
    %guard_offs = arith.addi %offs, %num_splat : tensor<16xi32>
    %mask = arith.cmpi slt, %guard_offs, %c512 : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<16x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Partial tail masks are not separable after prepending the H lane, so the pass
// must keep the original one-tile program shape.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: func.func @tile_chunk_partial_tail
// CHECK-NOT: sizes: [16, 16]
// CHECK: sizes: [16]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_partial_tail(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                          %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %c250 = arith.constant dense<250> : tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %mask = arith.cmpi slt, %offs, %c250 : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<16x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// A 2-D masked block may contain the tile-index signature on the outer grid
// axis, but a dynamic boundary mask on that axis means grid/H is not provably
// exact and the lifted mask is not a single structured slice. Keep the original
// program shape instead.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: func.func @tile_chunk_skip_2d_dynamic_boundary
// CHECK-NOT: tensor<2x256x256
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_skip_2d_dynamic_boundary(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                      %m: i32, %n: i32) {
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %c256 = arith.constant 256 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<256x256xf32>

    %row_blk = arith.muli %pid_m, %c256 : i32
    %row_range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %row_splat = tt.splat %row_blk : i32 -> tensor<256xi32>
    %row = arith.addi %row_splat, %row_range : tensor<256xi32>
    %row_2d = tt.expand_dims %row {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>

    %col_blk = arith.muli %pid_n, %c256 : i32
    %col_range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %col_splat = tt.splat %col_blk : i32 -> tensor<256xi32>
    %col = arith.addi %col_splat, %col_range : tensor<256xi32>
    %col_2d = tt.expand_dims %col {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>

    %n_row = tt.splat %n : i32 -> tensor<256x1xi32>
    %row_offset = arith.muli %row_2d, %n_row : tensor<256x1xi32>
    %row_offset_bc = tt.broadcast %row_offset : tensor<256x1xi32> -> tensor<256x256xi32>
    %col_bc = tt.broadcast %col_2d : tensor<1x256xi32> -> tensor<256x256xi32>
    %offsets = arith.addi %row_offset_bc, %col_bc : tensor<256x256xi32>

    %m_bound = tt.splat %m : i32 -> tensor<256x1xi32>
    %row_mask = arith.cmpi slt, %row_2d, %m_bound : tensor<256x1xi32>
    %row_mask_bc = tt.broadcast %row_mask : tensor<256x1xi1> -> tensor<256x256xi1>
    %n_bound = tt.splat %n : i32 -> tensor<1x256xi32>
    %col_mask = arith.cmpi slt, %col_2d, %n_bound : tensor<1x256xi32>
    %col_mask_bc = tt.broadcast %col_mask : tensor<1x256xi1> -> tensor<256x256xi1>
    %mask = arith.andi %row_mask_bc, %col_mask_bc : tensor<256x256xi1>

    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x256x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_base, %offsets : tensor<256x256x!tt.ptr<f32>>, tensor<256x256xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<256x256x!tt.ptr<f32>>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x256x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %offsets : tensor<256x256x!tt.ptr<f32>>, tensor<256x256xi32>
    tt.store %dst_ptr, %val, %mask : tensor<256x256x!tt.ptr<f32>>
    tt.return
  }
}
