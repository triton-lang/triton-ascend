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
// Unmasked kernels do not carry a static tile count in the IR. The pass should
// still coalesce with the power-of-two fallback factor.
// CHECK-LABEL: module attributes {hacc.coalesce_axis = 0 : i32, hacc.coalesce_factor = 16 : i32
// CHECK-LABEL: func.func @tile_chunk_coalesce_unmasked
// CHECK: memref.reinterpret_cast
// CHECK-SAME: sizes: [16, 16]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_chunk_coalesce_unmasked(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
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
