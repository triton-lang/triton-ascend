// RUN: triton-opt '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=false' --split-input-file --verify-each %s | FileCheck %s
// RUN: triton-opt '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true' --split-input-file --verify-each %s | FileCheck %s

// The bool load consumes ptr<i1> through a ptr<i8> bitcast. The shared axis
// analysis follows the loop-carried pointer to its broadcast source and slices
// the original ptr<i1> before applying the final ptr<i8> bitcast.
module {
  tt.func public @broadcast_bool_bitcast_load_small_mask(%input: !tt.ptr<i1>, %output: !tt.ptr<i1>, %limit: i32) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %step = arith.constant dense<16> : tensor<16x16xi32>
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_offsets = tt.expand_dims %rows {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %input_splat = tt.splat %input : !tt.ptr<i1> -> tensor<1x16x!tt.ptr<i1>>
    %input_ptrs = tt.addptr %input_splat, %row_offsets : tensor<1x16x!tt.ptr<i1>>, tensor<1x16xi32>
    %input_broadcast = tt.broadcast %input_ptrs : tensor<1x16x!tt.ptr<i1>> -> tensor<16x16x!tt.ptr<i1>>
    %limit_vec = tt.splat %limit : i32 -> tensor<16xi32>
    %mask_row = arith.cmpi slt, %rows, %limit_vec : tensor<16xi32>
    %mask_small = tt.expand_dims %mask_row {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
    %mask = tt.broadcast %mask_small : tensor<1x16xi1> -> tensor<16x16xi1>
    %zero = arith.constant dense<0> : tensor<16x16xi8>
    %loaded:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%ptr = %input_broadcast, %acc = %zero) -> (tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi8>) {
      %as_i8 = tt.bitcast %ptr : tensor<16x16x!tt.ptr<i1>> -> tensor<16x16x!tt.ptr<i8>>
      %value = tt.load %as_i8, %mask, %zero {was_bool_to_int8 = true} : tensor<16x16x!tt.ptr<i8>>
      %next_ptr = tt.addptr %ptr, %step : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi32>
      scf.yield %next_ptr, %value : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi8>
    }
    %as_bool = arith.trunci %loaded#1 : tensor<16x16xi8> to tensor<16x16xi1>
    %output_splat = tt.splat %output : !tt.ptr<i1> -> tensor<16x16x!tt.ptr<i1>>
    %output_offsets = tt.broadcast %row_offsets : tensor<1x16xi32> -> tensor<16x16xi32>
    %output_ptrs = tt.addptr %output_splat, %output_offsets : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi32>
    tt.store %output_ptrs, %as_bool : tensor<16x16x!tt.ptr<i1>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @broadcast_bool_bitcast_load_small_mask
// CHECK: %[[STEP:.*]] = arith.constant dense<16> : tensor<16x16xi32>
// CHECK: %[[LOOP:[^:]+]]:2 = scf.for {{.*}} iter_args(%[[PTR:[^ ,]+]] = {{.*}}) -> (tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi8>) {
// CHECK: %[[SMALL_PTR_I1:.*]] = tensor.extract_slice %[[PTR]]{{.*}} : tensor<16x16x!tt.ptr<i1>> to tensor<1x16x!tt.ptr<i1>>
// CHECK: %[[SMALL_PTR:.*]] = tt.bitcast %[[SMALL_PTR_I1]] : tensor<1x16x!tt.ptr<i1>> -> tensor<1x16x!tt.ptr<i8>>
// CHECK: %[[LOAD:.*]] = tt.load %[[SMALL_PTR]], %[[SMALL_MASK:[^ ,]+]], {{.*}} {was_bool_to_int8 = true} : tensor<1x16x!tt.ptr<i8>>
// CHECK: %[[VALUE:.*]] = tt.broadcast %[[LOAD]] : tensor<1x16xi8> -> tensor<16x16xi8>

// -----

// Any non-volatile load can use the same normalization. A broadcast pointer
// with no mask is loaded in its source shape directly.
module {
  tt.func public @generic_invariant_axis_no_mask(%input: !tt.ptr<f16>) -> tensor<16x16xf16> {
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %small_offsets = tt.expand_dims %rows {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %small_splat = tt.splat %input : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>>
    %small_ptrs = tt.addptr %small_splat, %small_offsets : tensor<1x16x!tt.ptr<f16>>, tensor<1x16xi32>
    %ptrs = tt.broadcast %small_ptrs : tensor<1x16x!tt.ptr<f16>> -> tensor<16x16x!tt.ptr<f16>>
    %value = tt.load %ptrs : tensor<16x16x!tt.ptr<f16>>
    tt.return %value : tensor<16x16xf16>
  }
}

// CHECK-LABEL: tt.func public @generic_invariant_axis_no_mask
// CHECK: %[[SMALL_PTR:.*]] = tt.addptr {{.*}} : tensor<1x16x!tt.ptr<f16>>, tensor<1x16xi32>
// CHECK: %[[LOAD:.*]] = tt.load %[[SMALL_PTR]] : tensor<1x16x!tt.ptr<f16>>
// CHECK: %[[VALUE:.*]] = tt.broadcast %[[LOAD]] : tensor<1x16xf16> -> tensor<16x16xf16>
// CHECK: tt.return %[[VALUE]] : tensor<16x16xf16>

// -----

// The pointer axis is invariant, but the mask varies along that axis. Load the
// small pointer without a mask and apply the complete-shape mask afterwards.
module {
  tt.func public @generic_invariant_axis_varying_mask(%input: !tt.ptr<f32>, %limit: i32) -> tensor<16x16xf32> {
    %cols = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %small_offsets = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %small_splat = tt.splat %input : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %small_ptrs = tt.addptr %small_splat, %small_offsets : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
    %ptrs = tt.broadcast %small_ptrs : tensor<1x16x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %limit_vec = tt.splat %limit : i32 -> tensor<16xi32>
    %mask_row = arith.cmpi slt, %rows, %limit_vec : tensor<16xi32>
    %mask_col = tt.expand_dims %mask_row {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %mask = tt.broadcast %mask_col : tensor<16x1xi1> -> tensor<16x16xi1>
    %other = arith.constant dense<1.000000e+00> : tensor<16x16xf32>
    %value = tt.load %ptrs, %mask, %other : tensor<16x16x!tt.ptr<f32>>
    tt.return %value : tensor<16x16xf32>
  }
}

// CHECK-LABEL: tt.func public @generic_invariant_axis_varying_mask
// CHECK: %[[SMALL_PTR:.*]] = tt.addptr {{.*}} : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
// CHECK: %[[MASK:.*]] = tt.broadcast {{.*}} : tensor<16x1xi1> -> tensor<16x16xi1>
// CHECK: %[[LOAD:.*]] = tt.load %[[SMALL_PTR]] : tensor<1x16x!tt.ptr<f32>>
// CHECK: %[[VALUE:.*]] = tt.broadcast %[[LOAD]] : tensor<1x16xf32> -> tensor<16x16xf32>
// CHECK: %[[SELECT:.*]] = arith.select %[[MASK]], %[[VALUE]], {{.*}} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK: tt.return %[[SELECT]] : tensor<16x16xf32>

// -----

// A mask that is invariant on the compressed axis stays attached to the small
// masked load. A varying fallback is applied with a full-shape select.
module {
  tt.func public @generic_invariant_axis_varying_other(%input: !tt.ptr<f32>, %limit: i32) -> tensor<16x16xf32> {
    %cols = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %small_offsets = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %small_splat = tt.splat %input : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %small_ptrs = tt.addptr %small_splat, %small_offsets : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
    %ptrs = tt.broadcast %small_ptrs : tensor<1x16x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %limit_vec = tt.splat %limit : i32 -> tensor<16xi32>
    %mask_row = arith.cmpi slt, %cols, %limit_vec : tensor<16xi32>
    %mask_small = tt.expand_dims %mask_row {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
    %mask = tt.broadcast %mask_small : tensor<1x16xi1> -> tensor<16x16xi1>
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %rows_f32 = arith.sitofp %rows : tensor<16xi32> to tensor<16xf32>
    %other_col = tt.expand_dims %rows_f32 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    %other = tt.broadcast %other_col : tensor<16x1xf32> -> tensor<16x16xf32>
    %value = tt.load %ptrs, %mask, %other : tensor<16x16x!tt.ptr<f32>>
    tt.return %value : tensor<16x16xf32>
  }
}

// CHECK-LABEL: tt.func public @generic_invariant_axis_varying_other
// CHECK: %[[SMALL_PTR:.*]] = tt.addptr {{.*}} : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
// CHECK: %[[MASK:.*]] = tt.broadcast {{.*}} : tensor<1x16xi1> -> tensor<16x16xi1>
// CHECK: %[[OTHER:.*]] = tt.broadcast {{.*}} : tensor<16x1xf32> -> tensor<16x16xf32>
// CHECK: %[[LOAD:.*]] = tt.load %[[SMALL_PTR]], %[[SMALL_MASK:[^ ,]+]] : tensor<1x16x!tt.ptr<f32>>
// CHECK: %[[VALUE:.*]] = tt.broadcast %[[LOAD]] : tensor<1x16xf32> -> tensor<16x16xf32>
// CHECK: %[[SELECT:.*]] = arith.select %[[MASK]], %[[VALUE]], %[[OTHER]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK: tt.return %[[SELECT]] : tensor<16x16xf32>

// -----

// A genuinely two-dimensional pointer has no invariant non-unit axis.
module {
  tt.func public @generic_non_invariant_load_unchanged(%input: !tt.ptr<f32>) -> tensor<16x16xf32> {
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %cols = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_col = tt.expand_dims %rows {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %col_row = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %row_bc = tt.broadcast %row_col : tensor<16x1xi32> -> tensor<16x16xi32>
    %col_bc = tt.broadcast %col_row : tensor<1x16xi32> -> tensor<16x16xi32>
    %offsets = arith.addi %row_bc, %col_bc : tensor<16x16xi32>
    %splat = tt.splat %input : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %ptrs = tt.addptr %splat, %offsets : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %value = tt.load %ptrs : tensor<16x16x!tt.ptr<f32>>
    tt.return %value : tensor<16x16xf32>
  }
}

// CHECK-LABEL: tt.func public @generic_non_invariant_load_unchanged
// CHECK: %[[PTRS:.*]] = tt.addptr {{.*}} : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
// CHECK-NOT: tensor.extract_slice
// CHECK: %[[VALUE:.*]] = tt.load %[[PTRS]] : tensor<16x16x!tt.ptr<f32>>
// CHECK-NOT: tensor.extract_slice
// CHECK: tt.return %[[VALUE]] : tensor<16x16xf32>

// -----

// The loop backedge can contain multiple pointer updates and control flow.
// Broadcast-axis analysis must follow the complete data flow instead of
// matching only the direct addptr form emitted by the original reproducer.
module {
  tt.func public @broadcast_bool_bitcast_compound_loop_update(%input: !tt.ptr<i1>, %output: !tt.ptr<i1>, %limit: i32) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %step = tt.splat %limit : i32 -> tensor<16x16xi32>
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_offsets = tt.expand_dims %rows {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %input_splat = tt.splat %input : !tt.ptr<i1> -> tensor<1x16x!tt.ptr<i1>>
    %input_ptrs = tt.addptr %input_splat, %row_offsets : tensor<1x16x!tt.ptr<i1>>, tensor<1x16xi32>
    %input_broadcast = tt.broadcast %input_ptrs : tensor<1x16x!tt.ptr<i1>> -> tensor<16x16x!tt.ptr<i1>>
    %limit_vec = tt.splat %limit : i32 -> tensor<16xi32>
    %mask_row = arith.cmpi slt, %rows, %limit_vec : tensor<16xi32>
    %mask_small = tt.expand_dims %mask_row {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
    %mask = tt.broadcast %mask_small : tensor<1x16xi1> -> tensor<16x16xi1>
    %zero = arith.constant dense<0> : tensor<16x16xi8>
    %loaded:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%ptr = %input_broadcast, %acc = %zero) -> (tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi8>) {
      %as_i8 = tt.bitcast %ptr : tensor<16x16x!tt.ptr<i1>> -> tensor<16x16x!tt.ptr<i8>>
      %value = tt.load %as_i8, %mask, %zero {was_bool_to_int8 = true} : tensor<16x16x!tt.ptr<i8>>
      %take_first = arith.cmpi slt, %iv, %c2 : index
      %next_ptr = scf.if %take_first -> tensor<16x16x!tt.ptr<i1>> {
        %first = tt.addptr %ptr, %step : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi32>
        %second = tt.addptr %first, %step : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi32>
        scf.yield %second : tensor<16x16x!tt.ptr<i1>>
      } else {
        scf.yield %ptr : tensor<16x16x!tt.ptr<i1>>
      }
      scf.yield %next_ptr, %value : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi8>
    }
    %as_bool = arith.trunci %loaded#1 : tensor<16x16xi8> to tensor<16x16xi1>
    %output_splat = tt.splat %output : !tt.ptr<i1> -> tensor<16x16x!tt.ptr<i1>>
    %output_offsets = tt.broadcast %row_offsets : tensor<1x16xi32> -> tensor<16x16xi32>
    %output_ptrs = tt.addptr %output_splat, %output_offsets : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi32>
    tt.store %output_ptrs, %as_bool : tensor<16x16x!tt.ptr<i1>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @broadcast_bool_bitcast_compound_loop_update
// CHECK: %[[LOOP:[^:]+]]:2 = scf.for {{.*}} iter_args(%[[PTR:[^ ,]+]] = {{.*}}) -> (tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi8>) {
// CHECK: %[[SMALL_PTR_I1:.*]] = tensor.extract_slice %[[PTR]]{{.*}} : tensor<16x16x!tt.ptr<i1>> to tensor<1x16x!tt.ptr<i1>>
// CHECK: %[[SMALL_PTR:.*]] = tt.bitcast %[[SMALL_PTR_I1]] : tensor<1x16x!tt.ptr<i1>> -> tensor<1x16x!tt.ptr<i8>>
// CHECK: %[[LOAD:.*]] = tt.load %[[SMALL_PTR]], %[[SMALL_MASK:[^ ,]+]], {{.*}} {was_bool_to_int8 = true} : tensor<1x16x!tt.ptr<i8>>
// CHECK: %[[VALUE:.*]] = tt.broadcast %[[LOAD]] : tensor<1x16xi8> -> tensor<16x16xi8>
// CHECK: scf.if
// CHECK: tt.addptr %[[PTR]], %{{.*}} : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi32>
// CHECK: scf.yield {{.*}}, %[[VALUE]] : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi8>

// -----

// Pointer analysis can rebuild the bool broadcast offset inside the loop as
// addptr(splat(base), broadcast(offset) + splat(loop_offset)). The bitcast must
// still expose the broadcast source shape so the load does not lower with a
// zero stride on the broadcast axis.
module {
  tt.func public @broadcast_bool_bitcast_addptr_load_small_mask(%input: !tt.ptr<i1>, %output: !tt.ptr<i1>, %limit: i32) {
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_offsets = tt.expand_dims %rows {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %broadcast_offsets = tt.broadcast %row_offsets : tensor<1x16xi32> -> tensor<16x16xi32>
    %loop_offset = tt.splat %limit : i32 -> tensor<16x16xi32>
    %offsets = arith.addi %broadcast_offsets, %loop_offset : tensor<16x16xi32>
    %input_splat = tt.splat %input : !tt.ptr<i1> -> tensor<16x16x!tt.ptr<i1>>
    %input_ptrs = tt.addptr %input_splat, %offsets {PointerDescriptorRebuild, PointerDescriptorStructuredAxes = array<i32: 1, 1>} : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi32>
    %limit_vec = tt.splat %limit : i32 -> tensor<16xi32>
    %mask_small = arith.cmpi slt, %rows, %limit_vec : tensor<16xi32>
    %mask_row = tt.expand_dims %mask_small {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
    %mask = tt.broadcast %mask_row : tensor<1x16xi1> -> tensor<16x16xi1>
    %zero = arith.constant dense<0> : tensor<16x16xi8>
    %as_i8 = tt.bitcast %input_ptrs : tensor<16x16x!tt.ptr<i1>> -> tensor<16x16x!tt.ptr<i8>>
    %value = tt.load %as_i8, %mask, %zero {was_bool_to_int8 = true} : tensor<16x16x!tt.ptr<i8>>
    %as_bool = arith.trunci %value : tensor<16x16xi8> to tensor<16x16xi1>
    %output_splat = tt.splat %output : !tt.ptr<i1> -> tensor<16x16x!tt.ptr<i1>>
    %output_offsets = tt.broadcast %row_offsets : tensor<1x16xi32> -> tensor<16x16xi32>
    %output_ptrs = tt.addptr %output_splat, %output_offsets : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi32>
    tt.store %output_ptrs, %as_bool : tensor<16x16x!tt.ptr<i1>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @broadcast_bool_bitcast_addptr_load_small_mask
// CHECK: %[[PTRS:.*]] = tt.addptr {{.*}} {PointerDescriptorRebuild, PointerDescriptorStructuredAxes = array<i32: 1, 1>}
// CHECK: %[[SMALL_PTR_I1:.*]] = tensor.extract_slice %[[PTRS]]{{.*}} : tensor<16x16x!tt.ptr<i1>> to tensor<1x16x!tt.ptr<i1>>
// CHECK: %[[SMALL_PTR:.*]] = tt.bitcast %[[SMALL_PTR_I1]] : tensor<1x16x!tt.ptr<i1>> -> tensor<1x16x!tt.ptr<i8>>
// CHECK: %[[LOAD:.*]] = tt.load %[[SMALL_PTR]], %[[SMALL_MASK:[^ ,]+]], {{.*}} {was_bool_to_int8 = true} : tensor<1x16x!tt.ptr<i8>>
// CHECK: %[[VALUE:.*]] = tt.broadcast %[[LOAD]] : tensor<1x16xi8> -> tensor<16x16xi8>
