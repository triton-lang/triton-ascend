// RUN: triton-opt --ssbuf-standardize-op-pack-memref %s | FileCheck %s

module {
  // Keep the custom write, cast, tensor view and layout conversion in one unit.
  // CHECK-LABEL: func.func @pack_l1_output_cast
  // CHECK: %[[S:.*]] = scope.scope : () -> tensor<16x16xbf16> {
  // CHECK-NEXT: %[[A:.*]] = memref.alloc()
  // CHECK-NEXT: hivm.hir.custom {{.*}} outs(%[[A]] :
  // CHECK-NEXT: %[[C:.*]] = memref.memory_space_cast %[[A]]
  // CHECK-NEXT: %[[T:.*]] = bufferization.to_tensor %[[C]]
  // CHECK-NEXT: %[[L:.*]] = hivm.hir.convert_layout %[[T]]
  // CHECK-NEXT: scope.return %[[L]] : tensor<16x16xbf16>
  // CHECK-NEXT: } {ssbuffer.core_type = "CUBE"}
  // CHECK-NEXT: return %[[S]]
  func.func @pack_l1_output_cast(%gm: memref<1xbf16>) -> tensor<16x16xbf16> {
    %alloc = memref.alloc() : memref<1x1x16x16xbf16, #hivm.address_space<cbuf>>
    hivm.hir.custom {arg_attrs = [{}, {address_space = #hivm.address_space<cbuf>, data_layout = #hivm.data_layout<Fractal>}], hivm.pipe = #hivm.pipe<PIPE_MTE2>, hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "load_l1"} "load_l1" ins(%gm : memref<1xbf16>) outs(%alloc : memref<1x1x16x16xbf16, #hivm.address_space<cbuf>>)
    %cast = memref.memory_space_cast %alloc : memref<1x1x16x16xbf16, #hivm.address_space<cbuf>> to memref<1x1x16x16xbf16>
    %tensor = bufferization.to_tensor %cast restrict : memref<1x1x16x16xbf16> to tensor<1x1x16x16xbf16>
    %layout = hivm.hir.convert_layout %tensor output_shape [16, 16] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<1x1x16x16xbf16>) -> tensor<16x16xbf16>
    return %layout : tensor<16x16xbf16>
  }

  // The tensor-DPS input pattern may also contain an address-space cast.
  // CHECK-LABEL: func.func @pack_tensor_dps_cast
  // CHECK: %[[S:.*]] = scope.scope : () -> tensor<2048xf32> {
  // CHECK-NEXT: %[[A:.*]] = memref.alloc()
  // CHECK-NEXT: %[[C:.*]] = memref.memory_space_cast %[[A]]
  // CHECK-NEXT: %[[T:.*]] = bufferization.to_tensor %[[C]]
  // CHECK-NEXT: %[[R:.*]] = hivm.hir.custom {{.*}} outs(%[[T]] :
  // CHECK-NEXT: scope.return %[[R]] : tensor<2048xf32>
  // CHECK-NEXT: } {ssbuffer.core_type = "VECTOR"}
  // CHECK-NEXT: return %[[S]]
  func.func @pack_tensor_dps_cast(%input: tensor<2048xf32>) -> tensor<2048xf32> {
    %alloc = memref.alloc() : memref<2048xf32, #hivm.address_space<ub>>
    %cast = memref.memory_space_cast %alloc : memref<2048xf32, #hivm.address_space<ub>> to memref<2048xf32>
    %tensor = bufferization.to_tensor %cast restrict writable : memref<2048xf32> to tensor<2048xf32>
    %result = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "vector_custom"} "vector_custom" ins(%input : tensor<2048xf32>) outs(%tensor : tensor<2048xf32>) -> tensor<2048xf32>
    return %result : tensor<2048xf32>
  }

  // A cast with another consumer must stay outside a packed custom unit.
  // CHECK-LABEL: func.func @leave_shared_cast_unpacked
  // CHECK-NOT: scope.scope
  // CHECK: hivm.hir.custom
  // CHECK-NOT: scope.scope
  // CHECK: memref.memory_space_cast
  // CHECK-NOT: scope.scope
  // CHECK: return
  func.func @leave_shared_cast_unpacked(%gm: memref<1xbf16>) -> (tensor<1x1x16x16xbf16>, index) {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1x1x16x16xbf16, #hivm.address_space<cbuf>>
    hivm.hir.custom {arg_attrs = [{}, {address_space = #hivm.address_space<cbuf>, data_layout = #hivm.data_layout<Fractal>}], hivm.pipe = #hivm.pipe<PIPE_MTE2>, hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "load_l1"} "load_l1" ins(%gm : memref<1xbf16>) outs(%alloc : memref<1x1x16x16xbf16, #hivm.address_space<cbuf>>)
    %cast = memref.memory_space_cast %alloc : memref<1x1x16x16xbf16, #hivm.address_space<cbuf>> to memref<1x1x16x16xbf16>
    %tensor = bufferization.to_tensor %cast restrict : memref<1x1x16x16xbf16> to tensor<1x1x16x16xbf16>
    %dim = memref.dim %cast, %c0 : memref<1x1x16x16xbf16>
    return %tensor, %dim : tensor<1x1x16x16xbf16>, index
  }
}
