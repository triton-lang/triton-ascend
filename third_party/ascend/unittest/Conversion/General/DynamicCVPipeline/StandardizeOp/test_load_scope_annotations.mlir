// RUN: triton-opt --ssbuf-standardize-op-pack-memref %s | FileCheck %s --check-prefix=PACK
// RUN: triton-opt --ssbuf-standardize-op-pack-memref --op-classifier %s | FileCheck %s --check-prefix=CORE

module {
  // A late annotation must move into the load without delaying the scope past
  // an existing tensor user. Preserve all annotations in their source order.
  // PACK-LABEL: func.func @late_tensor_annotations(
  // PACK: %[[SCOPE:.*]] = scope.scope : () -> tensor<4x4xf16> {
  // PACK: %[[T:.*]] = bufferization.to_tensor
  // PACK-NEXT: annotation.mark %[[T]] {MayImplicitTransposeWithLastAxis}
  // PACK-NEXT: annotation.mark %[[T]] {test.load_metadata}
  // PACK-NEXT: scope.return %[[T]] : tensor<4x4xf16>
  // PACK-NEXT: } {ssbuffer.core_type = "VECTOR", ssbuffer.load}
  // PACK-NEXT: %[[EXT:.*]] = arith.extf %[[SCOPE]]
  // PACK-NOT: annotation.mark
  // PACK: return %[[EXT]]
  // CORE-LABEL: func.func @late_tensor_annotations(
  // CORE: } {ssbuffer.core_type = "VECTOR", ssbuffer.load}
  func.func @late_tensor_annotations(%gm: memref<4x4xf16>) -> tensor<4x4xf32> {
    %alloc = memref.alloc() : memref<4x4xf16>
    memref.copy %gm, %alloc : memref<4x4xf16> to memref<4x4xf16>
    %t = bufferization.to_tensor %alloc restrict writable : memref<4x4xf16> to tensor<4x4xf16>
    %ext = arith.extf %t : tensor<4x4xf16> to tensor<4x4xf32>
    annotation.mark %t {MayImplicitTransposeWithLastAxis} : tensor<4x4xf16>
    annotation.mark %t {test.load_metadata} : tensor<4x4xf16>
    return %ext : tensor<4x4xf32>
  }

  // The same restriction on the allocation must also survive packing.
  // PACK-LABEL: func.func @implicit_transpose_on_buffer(
  // PACK: scope.scope
  // PACK: annotation.mark {{.*}} {MayImplicitTransposeWithLastAxis}
  // PACK: scope.return
  // PACK-NEXT: } {ssbuffer.core_type = "VECTOR", ssbuffer.load}
  // CORE-LABEL: func.func @implicit_transpose_on_buffer(
  // CORE: } {ssbuffer.core_type = "VECTOR", ssbuffer.load}
  func.func @implicit_transpose_on_buffer(%gm: memref<4x4xf16>) -> tensor<4x4xf16> {
    %alloc = memref.alloc() : memref<4x4xf16>
    memref.copy %gm, %alloc : memref<4x4xf16> to memref<4x4xf16>
    annotation.mark %alloc {MayImplicitTransposeWithLastAxis} : memref<4x4xf16>
    %t = bufferization.to_tensor %alloc restrict writable : memref<4x4xf16> to tensor<4x4xf16>
    return %t : tensor<4x4xf16>
  }

  // Hiding the tensor annotation must not let the matmul classifier recolour
  // an implicit-transpose load as CUBE.
  // PACK-LABEL: func.func @annotated_matmul_input(
  // PACK: annotation.mark {{.*}} {MayImplicitTransposeWithLastAxis}
  // PACK: } {ssbuffer.core_type = "VECTOR", ssbuffer.load}
  // CORE-LABEL: func.func @annotated_matmul_input(
  // CORE: %[[LOAD:.*]] = scope.scope
  // CORE: } {ssbuffer.core_type = "VECTOR", ssbuffer.load}
  // CORE: linalg.matmul {{.*}}ssbuffer.core_type = "CUBE"{{.*}} ins(%[[LOAD]],
  func.func @annotated_matmul_input(%gm: memref<4x4xf16>, %rhs: tensor<4x4xf16>, %init: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %alloc = memref.alloc() : memref<4x4xf16>
    memref.copy %gm, %alloc : memref<4x4xf16> to memref<4x4xf16>
    %t = bufferization.to_tensor %alloc restrict writable : memref<4x4xf16> to tensor<4x4xf16>
    annotation.mark %t {MayImplicitTransposeWithLastAxis} : tensor<4x4xf16>
    %mm = linalg.matmul ins(%t, %rhs : tensor<4x4xf16>, tensor<4x4xf16>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %mm : tensor<4x4xf32>
  }

  // Other metadata travels with the load but does not pin its core type.
  // PACK-LABEL: func.func @ordinary_annotation_matmul_input(
  // PACK: annotation.mark {{.*}} {test.load_metadata}
  // PACK: scope.return
  // PACK-NEXT: } {ssbuffer.load}
  // CORE-LABEL: func.func @ordinary_annotation_matmul_input(
  // CORE: %[[LOAD:.*]] = scope.scope
  // CORE: } {ssbuffer.core_type = "CUBE", ssbuffer.load}
  // CORE: linalg.matmul {{.*}}ssbuffer.core_type = "CUBE"{{.*}} ins(%[[LOAD]],
  func.func @ordinary_annotation_matmul_input(%gm: memref<4x4xf16>, %rhs: tensor<4x4xf16>, %init: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %alloc = memref.alloc() : memref<4x4xf16>
    memref.copy %gm, %alloc : memref<4x4xf16> to memref<4x4xf16>
    %t = bufferization.to_tensor %alloc restrict writable : memref<4x4xf16> to tensor<4x4xf16>
    annotation.mark %t {test.load_metadata} : tensor<4x4xf16>
    %mm = linalg.matmul ins(%t, %rhs : tensor<4x4xf16>, tensor<4x4xf16>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %mm : tensor<4x4xf32>
  }
}
