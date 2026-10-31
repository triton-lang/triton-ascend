// RUN: triton-opt --ssbuf-standardize-op-pack-memref %s | FileCheck %s

module {
  // ============================================================================
  // 1. pack_extracted_load_unit
  // An ExtractedLoadOrStore load unit (alloc -> copy loop -> to_tensor) is
  // packed into a single `ssbuffer.load` scope stamped VECTOR.
  // ============================================================================
  // CHECK-LABEL: func.func @pack_extracted_load_unit(
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<32x64xf16> {
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<32x64xf16>
  // CHECK:   scf.for
  // CHECK:     %[[SUBVIEW:.*]] = memref.subview %[[ALLOC]]
  // CHECK:     memref.copy {{.*}}, %[[SUBVIEW]]
  // CHECK:   } {ExtractedLoadOrStore}
  // CHECK:   %[[T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<32x64xf16>
  // CHECK:   scope.return %[[T]] : tensor<32x64xf16>
  // CHECK: } {ssbuffer.core_type = "VECTOR", ssbuffer.load}
  // CHECK: return %[[SCOPE]]
  func.func @pack_extracted_load_unit(%arg0: memref<?xf16>) -> tensor<32x64xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %alloc = memref.alloc() : memref<32x64xf16>
    scf.for %i = %c0 to %c32 step %c1 {
      %subview = memref.subview %alloc[%i, 0] [1, 64] [1, 1] : memref<32x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
      %src = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<?xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
      memref.copy %src, %subview : memref<1x64xf16, strided<[64, 1], offset: ?>> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    } {ExtractedLoadOrStore}
    %lhs = bufferization.to_tensor %alloc restrict writable : memref<32x64xf16> to tensor<32x64xf16>
    return %lhs : tensor<32x64xf16>
  }

  // ============================================================================
  // 2. pack_custom_op
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
  // 3. skip_plain_alloc
  // An allocation that is not part of a custom op or a load unit is left as-is.
  // ============================================================================
  // CHECK-LABEL: func.func @skip_plain_alloc(
  // CHECK-NOT: scope.scope
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64xf32>
  // CHECK: memref.copy %arg0, %[[ALLOC]]
  // CHECK: return
  func.func @skip_plain_alloc(%arg0: memref<64xf32>) {
    %alloc = memref.alloc() : memref<64xf32>
    memref.copy %arg0, %alloc : memref<64xf32> to memref<64xf32>
    return
  }

  // ============================================================================
  // 4. skip_compute_unit
  // The `alloc -> ops -> to_tensor` shape alone is not a load unit: the unit has
  // to read from GM, otherwise it is a compute pattern and stays unpacked.
  // ============================================================================
  // CHECK-LABEL: func.func @skip_compute_unit(
  // CHECK-NOT: scope.scope
  // CHECK: %[[ALLOC_C:.*]] = memref.alloc() : memref<64xf32>
  // CHECK: linalg.fill ins(%{{.*}} : f32) outs(%[[ALLOC_C]] : memref<64xf32>)
  // CHECK: %{{.*}} = bufferization.to_tensor %[[ALLOC_C]] restrict writable : memref<64xf32>
  // CHECK: return
  func.func @skip_compute_unit() -> tensor<64xf32> {
    %alloc = memref.alloc() : memref<64xf32>
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst : f32) outs(%alloc : memref<64xf32>)
    %t = bufferization.to_tensor %alloc restrict writable : memref<64xf32> to tensor<64xf32>
    return %t : tensor<64xf32>
  }

  // ============================================================================
  // 5. skip_nested_to_tensor
  // The closing to_tensor lives in a nested region, so it cannot move into a
  // scope together with the allocation: the unit is left unpacked instead of
  // being ordered across block boundaries.
  // ============================================================================
  // CHECK-LABEL: func.func @skip_nested_to_tensor(
  // CHECK-NOT: scope.scope
  // CHECK: %[[ALLOC_N:.*]] = memref.alloc() : memref<32x64xf16>
  // CHECK: %{{.*}} = scf.if
  // CHECK:   memref.copy
  // CHECK:   %{{.*}} = bufferization.to_tensor %[[ALLOC_N]] restrict writable : memref<32x64xf16>
  // CHECK:   scf.yield
  // CHECK: return
  func.func @skip_nested_to_tensor(%arg0: memref<32x64xf16>, %cond: i1)
      -> tensor<32x64xf16> {
    %alloc = memref.alloc() : memref<32x64xf16>
    %empty = tensor.empty() : tensor<32x64xf16>
    %res = scf.if %cond -> (tensor<32x64xf16>) {
      memref.copy %arg0, %alloc : memref<32x64xf16> to memref<32x64xf16>
      %t = bufferization.to_tensor %alloc restrict writable : memref<32x64xf16> to tensor<32x64xf16>
      scf.yield %t : tensor<32x64xf16>
    } else {
      scf.yield %empty : tensor<32x64xf16>
    }
    return %res : tensor<32x64xf16>
  }
  // ============================================================================
  // 6. pack_guarded_load_unit
  // A load unit whose alloc is pre-filled under a rare-branch guard and then
  // copied into from GM through subviews. The guard stays inside the scope.
  // ============================================================================
  // CHECK-LABEL: func.func @pack_guarded_load_unit(
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<32x64xf16> {
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<32x64xf16>
  // CHECK:   scf.if
  // CHECK:     linalg.fill
  // CHECK:   memref.copy {{.*}}, {{.*}} : memref<32x64xf16, strided<[64, 1]>> to memref<32x64xf16, strided<[64, 1]>>
  // CHECK:   %[[T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<32x64xf16>
  // CHECK:   scope.return %[[T]] : tensor<32x64xf16>
  // CHECK: } {ssbuffer.load}
  // CHECK: return %[[SCOPE]]
  func.func @pack_guarded_load_unit(%gm: memref<32x64xf16>, %cond: i1) -> tensor<32x64xf16> {
    %alloc = memref.alloc() : memref<32x64xf16>
    %cst = arith.constant 0.0 : f16
    %zero = memref.subview %alloc[0, 0] [32, 64] [1, 1] : memref<32x64xf16> to memref<32x64xf16, strided<[64, 1]>>
    scf.if %cond {
      linalg.fill ins(%cst : f16) outs(%zero : memref<32x64xf16, strided<[64, 1]>>)
    }
    %src = memref.subview %gm[0, 0] [32, 64] [1, 1] : memref<32x64xf16> to memref<32x64xf16, strided<[64, 1]>>
    %dst = memref.subview %alloc[0, 0] [32, 64] [1, 1] : memref<32x64xf16> to memref<32x64xf16, strided<[64, 1]>>
    memref.copy %src, %dst : memref<32x64xf16, strided<[64, 1]>> to memref<32x64xf16, strided<[64, 1]>>
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x64xf16> to tensor<32x64xf16>
    return %t : tensor<32x64xf16>
  }

  // ============================================================================
  // 7. custom_op_core_or_vector_leaves_core_type_open
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
  // 8. custom_op_cube_and_vector_keeps_both_cores
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
  // 9. pack_custom_op_filling_local_buffer
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
  // 10. pack_custom_op_with_convert_layout
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
  // 11. no_pack_with_second_to_tensor
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
  // 12. no_pack_inside_loop_iter_arg
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

  // ============================================================================
  // 13. pack_load_unit_closure_feeding_outside_consumer
  // A closure scalar whose only closure consumer is itself copied (it has a
  // consumer before the unit) must be copied as well: the copy rule cascades,
  // otherwise the copied op's original would read a value that moved into a
  // scope created after it.
  // ============================================================================
  // CHECK-LABEL: func.func @pack_load_unit_closure_feeding_outside_consumer(
  // CHECK: %[[A:.*]] = arith.muli
  // CHECK: %[[B:.*]] = arith.addi %[[A]]
  // CHECK: memref.store %[[B]], {{.*}} : memref<1xi32>
  // The whole closure prefix of `%[[B]]` is copied into the scope, so the store
  // above keeps reading the outer chain.
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<32x64xf16> {
  // CHECK:   %[[A2:.*]] = arith.muli
  // CHECK:   %[[B2:.*]] = arith.addi %[[A2]]
  // CHECK:   %[[IDX:.*]] = arith.index_cast %[[B2]] : i32 to index
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<32x64xf16>
  // CHECK:   %[[SRC:.*]] = memref.reinterpret_cast {{.*}} to offset: [%[[IDX]]], sizes: [32, 64], strides: [64, 1]
  // CHECK:   %[[DST:.*]] = memref.subview %[[ALLOC]]
  // CHECK:   memref.copy %[[SRC]], %[[DST]]
  // CHECK:   %[[T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<32x64xf16>
  // CHECK:   scope.return %[[T]] : tensor<32x64xf16>
  // CHECK: } {ssbuffer.load}
  // CHECK: return %[[SCOPE]]
  func.func @pack_load_unit_closure_feeding_outside_consumer(
      %gm: memref<?xf16>, %flag: memref<1xi32>) -> tensor<32x64xf16> {
    %c0 = arith.constant 0 : index
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %a = arith.muli %c32_i32, %c32_i32 : i32
    %b = arith.addi %a, %c1_i32 : i32
    memref.store %b, %flag[%c0] : memref<1xi32>
    %idx = arith.index_cast %b : i32 to index
    %alloc = memref.alloc() : memref<32x64xf16>
    %src = memref.reinterpret_cast %gm to offset: [%idx], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
    %dst = memref.subview %alloc[0, 0] [32, 64] [1, 1] : memref<32x64xf16> to memref<32x64xf16, strided<[64, 1]>>
    memref.copy %src, %dst : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16, strided<[64, 1]>>
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x64xf16> to tensor<32x64xf16>
    return %t : tensor<32x64xf16>
  }
}
