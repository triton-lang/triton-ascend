// RUN: triton-opt --ssbuf-standardize-op-pack-memref %s | FileCheck %s

module {
  // ============================================================================
  // 1. pack_custom_op
  // A custom op unit is entered through its local alloc and packed into a scope
  // carrying the analyzed core type.
  // ============================================================================
  // CHECK-LABEL: func.func @pack_custom_op(
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<2048xf32> {
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<2048xf32>
  // CHECK:   %[[T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<2048xf32>
  // CHECK:   %[[CUSTOM:.*]] = hivm.hir.custom {{.*}} outs(%[[T]] : tensor<2048xf32>) -> tensor<2048xf32>
  // CHECK:   scope.return %[[CUSTOM]] : tensor<2048xf32>
  // CHECK: } {ssbuffer.core_type = "VECTOR"}
  // CHECK: return %[[SCOPE]]
  func.func @pack_custom_op(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %alloc = memref.alloc() : memref<2048xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<2048xf32> to tensor<2048xf32>
    %1 = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "tanh_fp32"} "tanh_fp32" ins(%arg0 : tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
    return %1 : tensor<2048xf32>
  }

  // ============================================================================
  // 2. custom_op_core_or_vector_leaves_core_type_open
  // `CUBE_OR_VECTOR` does not pin a core, so the packed scope must not stamp
  // `ssbuffer.core_type`: OpClassifierPass then chooses a single core instead of
  // splitting the op into a CUBE and a VECTOR copy.
  // ============================================================================
  // CHECK-LABEL: func.func @custom_op_core_or_vector(
  // CHECK: %[[OR_SCOPE:.*]] = scope.scope : () -> tensor<2048xf32> {
  // CHECK:   %[[OR_ALLOC:.*]] = memref.alloc() : memref<2048xf32>
  // CHECK:   %[[OR_T:.*]] = bufferization.to_tensor %[[OR_ALLOC]] restrict writable : memref<2048xf32>
  // CHECK:   hivm.hir.custom {{.*}} symbol = "or_vector_custom"
  // CHECK:   scope.return
  // CHECK-NOT: ssbuffer.core_type
  // CHECK: return %[[OR_SCOPE]]
  func.func @custom_op_core_or_vector(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %alloc = memref.alloc() : memref<2048xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<2048xf32> to tensor<2048xf32>
    %1 = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "or_vector_custom"} "or_vector_custom" ins(%arg0 : tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
    return %1 : tensor<2048xf32>
  }
  // ============================================================================
  // 3. custom_op_cube_and_vector_keeps_both_cores
  // `CUBE_AND_VECTOR` pins both cores: the scope must carry it so that
  // OpClassifierPass splits the op into a CUBE copy and a VECTOR copy.
  // ============================================================================
  // CHECK-LABEL: func.func @custom_op_cube_and_vector(
  // CHECK: %[[BOTH_SCOPE:.*]] = scope.scope : () -> tensor<2048xf32> {
  // CHECK:   %[[BOTH_ALLOC:.*]] = memref.alloc() : memref<2048xf32>
  // CHECK:   %[[BOTH_T:.*]] = bufferization.to_tensor %[[BOTH_ALLOC]] restrict writable : memref<2048xf32>
  // CHECK:   hivm.hir.custom {{.*}} symbol = "both_custom"
  // CHECK:   scope.return
  // CHECK: } {ssbuffer.core_type = "CUBE_AND_VECTOR"}
  // CHECK: return %[[BOTH_SCOPE]]
  func.func @custom_op_cube_and_vector(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %alloc = memref.alloc() : memref<2048xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<2048xf32> to tensor<2048xf32>
    %1 = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "both_custom"} "both_custom" ins(%arg0 : tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
    return %1 : tensor<2048xf32>
  }

  // ============================================================================
  // 4. pack_custom_op_filling_local_buffer
  // The custom op writes the local buffer itself, so the alloc has two users:
  // the custom op and the to_tensor. The unit is still packed and the
  // to_tensor follows the custom op inside the scope.
  // ============================================================================
  // CHECK-LABEL: func.func @pack_custom_op_filling_local_buffer(
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<2048xf32> {
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<2048xf32>
  // CHECK:   %[[CUSTOM:.*]] = hivm.hir.custom {{.*}} outs(%[[ALLOC]] : memref<2048xf32>) -> memref<2048xf32>
  // CHECK:   %[[T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<2048xf32>
  // CHECK:   scope.return %[[T]] : tensor<2048xf32>
  // CHECK: } {ssbuffer.core_type = "VECTOR"}
  // CHECK: return %[[SCOPE]]
  func.func @pack_custom_op_filling_local_buffer(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %alloc = memref.alloc() : memref<2048xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<2048xf32> to tensor<2048xf32>
    %1 = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "tanh_fp32"} "tanh_fp32" ins(%arg0 : tensor<2048xf32>) outs(%alloc : memref<2048xf32>) -> memref<2048xf32>
    return %0 : tensor<2048xf32>
  }

  // ============================================================================
  // 5. pack_custom_op_with_convert_layout
  // A to_tensor whose single user is a layout conversion drags that conversion
  // into the unit, so the packed scope returns the converted tensor.
  // ============================================================================
  // CHECK-LABEL: func.func @pack_custom_op_with_convert_layout(
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<2048xf32> {
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<2048xf32>
  // CHECK:   %[[CUSTOM:.*]] = hivm.hir.custom {{.*}} outs(%[[ALLOC]] : memref<2048xf32>) -> memref<2048xf32>
  // CHECK:   %[[T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<2048xf32>
  // CHECK:   %[[CL:.*]] = hivm.hir.convert_layout %[[T]] output_shape [2048] {{.*}} : (tensor<2048xf32>) -> tensor<2048xf32>
  // CHECK:   scope.return %[[CL]] : tensor<2048xf32>
  // CHECK: } {ssbuffer.core_type = "VECTOR"}
  // CHECK: return %[[SCOPE]]
  func.func @pack_custom_op_with_convert_layout(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %alloc = memref.alloc() : memref<2048xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<2048xf32> to tensor<2048xf32>
    %1 = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "tanh_fp32"} "tanh_fp32" ins(%arg0 : tensor<2048xf32>) outs(%alloc : memref<2048xf32>) -> memref<2048xf32>
    %2 = hivm.hir.convert_layout %0 output_shape [2048] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>} : (tensor<2048xf32>) -> tensor<2048xf32>
    return %2 : tensor<2048xf32>
  }

  // ============================================================================
  // 6. no_pack_with_second_to_tensor
  // A buffer read through two to_tensor ops cannot be packed: the scope would
  // tear a live value apart, so the unit is left untouched.
  // ============================================================================
  // CHECK-LABEL: func.func @no_pack_with_second_to_tensor(
  // CHECK-NOT: scope.scope
  // CHECK: memref.alloc
  // CHECK: bufferization.to_tensor
  // CHECK: hivm.hir.custom
  // CHECK: bufferization.to_tensor
  // CHECK: return
  func.func @no_pack_with_second_to_tensor(%arg0: tensor<2048xf32>) -> (tensor<2048xf32>, tensor<2048xf32>) {
    %alloc = memref.alloc() : memref<2048xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<2048xf32> to tensor<2048xf32>
    %1 = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "tanh_fp32"} "tanh_fp32" ins(%arg0 : tensor<2048xf32>) outs(%alloc : memref<2048xf32>) -> memref<2048xf32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<2048xf32> to tensor<2048xf32>
    return %0, %2 : tensor<2048xf32>, tensor<2048xf32>
  }

  // ============================================================================
  // 7. no_pack_inside_loop_iter_arg
  // The buffer resolves to a block argument of a non-function op, which has no
  // home of its own; the analysis fails and nothing is packed.
  // ============================================================================
  // CHECK-LABEL: func.func @no_pack_inside_loop_iter_arg(
  // CHECK-NOT: scope.scope
  // CHECK: scf.for
  // CHECK: return
  func.func @no_pack_inside_loop_iter_arg(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<2048xf32>
    %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%buf = %alloc) -> (memref<2048xf32>) {
      %1 = bufferization.to_tensor %buf restrict writable : memref<2048xf32> to tensor<2048xf32>
      %2 = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "tanh_fp32"} "tanh_fp32" ins(%arg0 : tensor<2048xf32>) outs(%1 : tensor<2048xf32>) -> tensor<2048xf32>
      scf.yield %buf : memref<2048xf32>
    }
    %3 = bufferization.to_tensor %0 restrict writable : memref<2048xf32> to tensor<2048xf32>
    return %3 : tensor<2048xf32>
  }

  // ============================================================================
  // 8. pack_read_only_custom_op_with_convert_layout
  // The custom op only reads the local buffer, so the unit's output is the custom
  // op result: a layout conversion consuming it joins the unit as well.
  // ============================================================================
  // CHECK-LABEL: func.func @pack_read_only_custom_op_with_convert_layout(
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<2048xf32> {
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<2048xf32>
  // CHECK:   %[[T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<2048xf32>
  // CHECK:   %[[CUSTOM:.*]] = hivm.hir.custom {{.*}} outs(%[[T]] : tensor<2048xf32>) -> tensor<2048xf32>
  // CHECK:   %[[CL:.*]] = hivm.hir.convert_layout %[[CUSTOM]] output_shape [2048] {{.*}} : (tensor<2048xf32>) -> tensor<2048xf32>
  // CHECK:   scope.return %[[CL]] : tensor<2048xf32>
  // CHECK: } {ssbuffer.core_type = "VECTOR"}
  // CHECK: return %[[SCOPE]]
  func.func @pack_read_only_custom_op_with_convert_layout(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %alloc = memref.alloc() : memref<2048xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<2048xf32> to tensor<2048xf32>
    %1 = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "tanh_fp32"} "tanh_fp32" ins(%arg0 : tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
    %2 = hivm.hir.convert_layout %1 output_shape [2048] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>} : (tensor<2048xf32>) -> tensor<2048xf32>
    return %2 : tensor<2048xf32>
  }

  // ============================================================================
  // 9. no_pack_of_convert_layout_for_multi_result
  // `hivm.hir.custom` has no single-result trait: with two results there is no
  // single output value the unit produces, so the layout conversion stays out.
  // ============================================================================
  // CHECK-LABEL: func.func @no_pack_of_convert_layout_for_multi_result(
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<2048xf32> {
  // CHECK:   %[[ALLOC0:.*]] = memref.alloc() : memref<2048xf32>
  // CHECK:   %[[T0:.*]] = bufferization.to_tensor %[[ALLOC0]] restrict writable : memref<2048xf32>
  // CHECK:   %[[ALLOC1:.*]] = memref.alloc() : memref<2048xf32>
  // CHECK:   %[[T1:.*]] = bufferization.to_tensor %[[ALLOC1]] restrict writable : memref<2048xf32>
  // CHECK:   %[[CUSTOM:.*]]:2 = hivm.hir.custom {{.*}} -> (tensor<2048xf32>, tensor<2048xf32>)
  // CHECK:   scope.return %[[CUSTOM]]#0 : tensor<2048xf32>
  // CHECK: } {ssbuffer.core_type = "VECTOR"}
  // CHECK: %[[CL:.*]] = hivm.hir.convert_layout %[[SCOPE]] output_shape [2048]
  // CHECK: return %[[CL]]
  func.func @no_pack_of_convert_layout_for_multi_result(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %alloc = memref.alloc() : memref<2048xf32>
    %a0 = bufferization.to_tensor %alloc restrict writable : memref<2048xf32> to tensor<2048xf32>
    %alloc2 = memref.alloc() : memref<2048xf32>
    %a1 = bufferization.to_tensor %alloc2 restrict writable : memref<2048xf32> to tensor<2048xf32>
    %1:2 = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "tanh_fp32"} "tanh_fp32" ins(%arg0 : tensor<2048xf32>) outs(%a0, %a1 : tensor<2048xf32>, tensor<2048xf32>) -> (tensor<2048xf32>, tensor<2048xf32>)
    %2 = hivm.hir.convert_layout %1#0 output_shape [2048] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>} : (tensor<2048xf32>) -> tensor<2048xf32>
    return %2 : tensor<2048xf32>
  }
}
