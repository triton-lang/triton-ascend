// RUN: triton-opt --split-input-file --materialize-cube-page-loaders --reorder-ops-by-block-id --verify-each %s | FileCheck %s --check-prefix=MATERIAL
// RUN: triton-opt --split-input-file --materialize-cube-page-loaders --reorder-ops-by-block-id --clone-ops --verify-each %s | FileCheck %s --check-prefix=CLONE

// Different page metadata must not be treated as shared preparation.
// MATERIAL-LABEL: func.func @different_metadata
// MATERIAL-NOT: ssbuffer.shared_page_write
// MATERIAL: return
// CLONE-LABEL: func.func @different_metadata
// CLONE-NOT: ssbuffer.shared_page_write
// CLONE: return
func.func @different_metadata(%metadata0: memref<?xi32>, %metadata1: memref<?xi32>, %k: memref<?xf16>, %v: memref<?xf16>, %q: tensor<16x16xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %empty = tensor.empty() : tensor<16x16xf16>
  %acc = arith.constant dense<0.0> : tensor<16x16xf32>
  scope.scope : () -> () {
    scf.for %iv = %c0 to %c3 step %c1 {
      %meta0 = memref.load %metadata0[%iv] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32>
      %offset0 = arith.index_cast %meta0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %src0 = memref.reinterpret_cast %k to offset: [%offset0], sizes: [16, 16], strides: [16, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
      %page0 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16>
      memref.copy %src0, %page0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<16x16xf16>
      %tensor0 = bufferization.to_tensor %page0 restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16> to tensor<16x16xf16>
      %insert0 = tensor.insert_slice %tensor0 into %empty[0, 0] [16, 16] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x16xf16> into tensor<16x16xf16>
      %mm0 = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %insert0 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      %meta1 = memref.load %metadata1[%iv] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32>
      %offset1 = arith.index_cast %meta1 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %src1 = memref.reinterpret_cast %v to offset: [%offset1], sizes: [16, 16], strides: [16, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
      %page1 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16>
      memref.copy %src1, %page1 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<16x16xf16>
      %tensor1 = bufferization.to_tensor %page1 restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16> to tensor<16x16xf16>
      %insert1 = tensor.insert_slice %tensor1 into %empty[0, 0] [16, 16] [1, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x16xf16> into tensor<16x16xf16>
      %mm1 = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %insert1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    } {ssbuffer.block_id = 0 : i32, ssbuffer.main_loop = 0 : i32}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}
