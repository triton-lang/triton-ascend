// RUN: triton-opt --triton-to-structured --gather-optimization %s | FileCheck %s

// CHECK:   scf.if {{%[0-9]+}}
// CHECK:   tt.load {{%[0-9]+}} {gather.optimised.load = "source"} : tensor<2x16x64x!tt.ptr<f32>>
// CHECK:   tt.gather {{%[0-9]+}}[{{%[0-9]+}}] {axis = 2 : i32} : (tensor<2x16x64xf32>, tensor<2x16x128xi32>) -> tensor<2x16x128xf32>
// CHECK:   else
// CHECK:   tt.load {{%[0-9]+, %[0-9]+, %cst_[0-9]+}} {gather.optimised.load = "fallback"} : tensor<2x16x128x!tt.ptr<f32>>

module attributes {hacc.target = #hacc.target<"Ascend910B3">} {
  tt.func public @scalar_loop_load_nd(%src_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %idx_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32})  attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x16x128xf32>
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<2x16x128xi32>
    %cst_1 = arith.constant dense<128> : tensor<2x16x1xi32>
    %cst_2 = arith.constant dense<64> : tensor<2x16x1xi32>
    %cst_3 = arith.constant dense<16> : tensor<2x1xi32>
    %c65536_i32 = arith.constant 65536 : i32
    %c1639_i32 = arith.constant 1639 : i32
    %row_begin = tt.get_program_id x : i32
    %row_begin_4 = arith.muli %row_begin, %c1639_i32 : i32
    %row_end = arith.addi %row_begin_4, %c1639_i32 : i32
    %row_end_5 = arith.minsi %row_end, %c65536_i32 : i32
    %in_offsets_base = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %in_offsets_base_6 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %in_offsets_base_7 = tt.expand_dims %in_offsets_base_6 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %in_offsets_base_8 = tt.broadcast %in_offsets_base_7 : tensor<1x16xi32> -> tensor<2x16xi32>
    %mask = tt.splat %row_end_5 : i32 -> tensor<2xi32>
    %idx_offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %idx_offsets_9 = tt.expand_dims %idx_offsets {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %idx_offsets_10 = tt.expand_dims %idx_offsets_9 {axis = 1 : i32} : tensor<1x128xi32> -> tensor<1x1x128xi32>
    %idx_offsets_11 = tt.broadcast %idx_offsets_10 : tensor<1x1x128xi32> -> tensor<2x16x128xi32>
    %idx = tt.splat %idx_ptr : !tt.ptr<i32> -> tensor<2x16x128x!tt.ptr<i32>>
    %out = tt.splat %src_ptr : !tt.ptr<f32> -> tensor<2x16x128x!tt.ptr<f32>>
    %0 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<2x16x128x!tt.ptr<f32>>
    scf.for %rb = %c0_i32 to %c1639_i32 step %c2_i32  : i32 {
      %in_offsets_base_12 = arith.addi %row_begin_4, %rb : i32
      %in_offsets_base_13 = tt.splat %in_offsets_base_12 : i32 -> tensor<2xi32>
      %in_offsets_base_14 = arith.addi %in_offsets_base_13, %in_offsets_base : tensor<2xi32>
      %in_offsets_base_15 = tt.expand_dims %in_offsets_base_14 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
      %in_offsets_base_16 = arith.muli %in_offsets_base_15, %cst_3 : tensor<2x1xi32>
      %in_offsets_base_17 = tt.broadcast %in_offsets_base_16 : tensor<2x1xi32> -> tensor<2x16xi32>
      %in_offsets_base_18 = arith.addi %in_offsets_base_17, %in_offsets_base_8 : tensor<2x16xi32>
      %in_offsets_base_19 = tt.expand_dims %in_offsets_base_18 {axis = 2 : i32} : tensor<2x16xi32> -> tensor<2x16x1xi32>
      %in_offsets_base_20 = arith.muli %in_offsets_base_19, %cst_2 : tensor<2x16x1xi32>
      %mask_21 = arith.cmpi slt, %in_offsets_base_14, %mask : tensor<2xi32>
      %mask_22 = tt.expand_dims %mask_21 {axis = 1 : i32} : tensor<2xi1> -> tensor<2x1xi1>
      %idx_offsets_23 = arith.muli %in_offsets_base_19, %cst_1 : tensor<2x16x1xi32>
      %idx_offsets_24 = tt.broadcast %idx_offsets_23 : tensor<2x16x1xi32> -> tensor<2x16x128xi32>
      %idx_offsets_25 = arith.addi %idx_offsets_24, %idx_offsets_11 : tensor<2x16x128xi32>
      %mask_26 = tt.expand_dims %mask_22 {axis = 2 : i32} : tensor<2x1xi1> -> tensor<2x1x1xi1>
      %idx_27 = tt.addptr %idx, %idx_offsets_25 : tensor<2x16x128x!tt.ptr<i32>>, tensor<2x16x128xi32>
      %idx_28 = tt.broadcast %mask_26 : tensor<2x1x1xi1> -> tensor<2x16x128xi1>
      %idx_29 = tt.load %idx_27, %idx_28, %cst_0 : tensor<2x16x128x!tt.ptr<i32>>
      %in_offsets = tt.broadcast %in_offsets_base_20 : tensor<2x16x1xi32> -> tensor<2x16x128xi32>
      %in_offsets_30 = arith.addi %in_offsets, %idx_29 : tensor<2x16x128xi32>
      %out_31 = tt.addptr %out, %in_offsets_30 : tensor<2x16x128x!tt.ptr<f32>>, tensor<2x16x128xi32>
      %out_32 = tt.load %out_31, %idx_28, %cst : tensor<2x16x128x!tt.ptr<f32>>
      %1 = tt.addptr %0, %idx_offsets_25 : tensor<2x16x128x!tt.ptr<f32>>, tensor<2x16x128xi32>
      tt.store %1, %out_32, %idx_28 : tensor<2x16x128x!tt.ptr<f32>>
    }
    tt.return
  }
}
