// RUN: triton-opt --plan-compute-block %s | FileCheck %s

// Only an op whose core type is CUBE_AND_VECTOR is duplicated into a CUBE copy
// and a VECTOR copy. A pre-classified single-core marker (CUBE / VECTOR) is
// authoritative and is never duplicated; cross-core consumption of it is
// carried by the dependency chain.
//
// Every scope body is a real load unit: GM -> alloc `memref.copy`, or the same
// copy under a rare-branch `scf.if` that pre-fills the alloc, reading the GM
// through subviews of both sides.
module {

  // A pre-classified CUBE scope stays CUBE: a single-core marker is
  // authoritative and is not duplicated. The VECTOR addf consumes it through
  // the dependency chain.
  // CHECK-LABEL: func.func @shared_load_scope_pre_classified_cube
  // CHECK: %[[CUBE:.*]] = scope.scope : () -> tensor<4x4xf16> {
  // CHECK: } {ssbuffer.block_id = {{[0-9]+}} : i32, ssbuffer.core_type = "CUBE", ssbuffer.load}
  // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.block_id = {{[0-9]+}} : i32, ssbuffer.core_type = "CUBE"} ins(%[[CUBE]],
  // CHECK: %[[ADD:.*]] = arith.addf %[[CUBE]], %[[CUBE]] {ssbuffer.block_id = {{[0-9]+}} : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: return %[[MM]], %[[ADD]]
  // CHECK-NOT: scope.scope
  func.func @shared_load_scope_pre_classified_cube(%gm: memref<4x4xf16>, %arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf16>) {
    %0 = scope.scope : () -> tensor<4x4xf16> {
      %alloc = memref.alloc() : memref<4x4xf16>
      memref.copy %gm, %alloc : memref<4x4xf16> to memref<4x4xf16>
      %t = bufferization.to_tensor %alloc restrict writable : memref<4x4xf16> to tensor<4x4xf16>
      scope.return %t : tensor<4x4xf16>
    } {ssbuffer.core_type = "CUBE", ssbuffer.load}
    %1 = linalg.matmul ins(%0, %arg0 : tensor<4x4xf16>, tensor<4x4xf16>) outs(%arg1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = arith.addf %0, %0 : tensor<4x4xf16>
    return %1, %2 : tensor<4x4xf32>, tensor<4x4xf16>
  }

  // Same for a pre-classified VECTOR scope: it stays VECTOR; the CUBE matmul
  // and the VECTOR addf both read the single VECTOR scope.
  // CHECK-LABEL: func.func @shared_load_scope_pre_classified_vector
  // CHECK: %[[VEC:.*]] = scope.scope : () -> tensor<4x4xf16> {
  // CHECK: } {ssbuffer.block_id = {{[0-9]+}} : i32, ssbuffer.core_type = "VECTOR", ssbuffer.load}
  // CHECK: %[[ADD:.*]] = arith.addf %[[VEC]], %[[VEC]] {ssbuffer.block_id = {{[0-9]+}} : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.block_id = {{[0-9]+}} : i32, ssbuffer.core_type = "CUBE"} ins(%[[VEC]],
  // CHECK: return %[[MM]], %[[ADD]]
  // CHECK-NOT: scope.scope
  func.func @shared_load_scope_pre_classified_vector(%gm: memref<4x4xf16>, %arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf32>, %rare: i1) -> (tensor<4x4xf32>, tensor<4x4xf16>) {
    %0 = scope.scope : () -> tensor<4x4xf16> {
      %alloc = memref.alloc() : memref<4x4xf16>
      %cst = arith.constant 0.0 : f16
      scf.if %rare {
        linalg.fill ins(%cst : f16) outs(%alloc : memref<4x4xf16>)
      }
      %src = memref.subview %gm[0, 0] [4, 4] [1, 1] : memref<4x4xf16> to memref<4x4xf16, strided<[4, 1]>>
      %dst = memref.subview %alloc[0, 0] [4, 4] [1, 1] : memref<4x4xf16> to memref<4x4xf16, strided<[4, 1]>>
      memref.copy %src, %dst : memref<4x4xf16, strided<[4, 1]>> to memref<4x4xf16, strided<[4, 1]>>
      %t = bufferization.to_tensor %alloc restrict writable : memref<4x4xf16> to tensor<4x4xf16>
      scope.return %t : tensor<4x4xf16>
    } {ssbuffer.core_type = "VECTOR", ssbuffer.load}
    %1 = linalg.matmul ins(%0, %arg0 : tensor<4x4xf16>, tensor<4x4xf16>) outs(%arg1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = arith.addf %0, %0 : tensor<4x4xf16>
    return %1, %2 : tensor<4x4xf32>, tensor<4x4xf16>
  }

  // An op already stamped CUBE_AND_VECTOR must still run on both cores, so
  // OpClassifier splits it into a CUBE copy and a VECTOR copy.
  // CHECK-LABEL: func.func @pre_classified_cube_and_vector_is_split
  // CHECK: %[[VEC:.*]] = scope.scope : () -> tensor<4x4xf16> {
  // CHECK: } {ssbuffer.block_id = {{[0-9]+}} : i32, ssbuffer.core_type = "VECTOR", ssbuffer.load}
  // CHECK: %[[ADD:.*]] = arith.addf %[[VEC]], %[[VEC]] {ssbuffer.block_id = {{[0-9]+}} : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: %[[CUBE:.*]] = scope.scope : () -> tensor<4x4xf16> {
  // CHECK: } {ssbuffer.block_id = {{[0-9]+}} : i32, ssbuffer.core_type = "CUBE", ssbuffer.load}
  // CHECK: linalg.matmul {ssbuffer.block_id = {{[0-9]+}} : i32, ssbuffer.core_type = "CUBE"} ins(%[[CUBE]],
  func.func @pre_classified_cube_and_vector_is_split(%gm: memref<4x4xf16>, %arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf16>) {
    %0 = scope.scope : () -> tensor<4x4xf16> {
      %alloc = memref.alloc() : memref<4x4xf16>
      memref.copy %gm, %alloc : memref<4x4xf16> to memref<4x4xf16>
      %t = bufferization.to_tensor %alloc restrict writable : memref<4x4xf16> to tensor<4x4xf16>
      scope.return %t : tensor<4x4xf16>
    } {ssbuffer.core_type = "CUBE_AND_VECTOR", ssbuffer.load}
    %1 = linalg.matmul ins(%0, %arg0 : tensor<4x4xf16>, tensor<4x4xf16>) outs(%arg1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = arith.addf %0, %0 : tensor<4x4xf16>
    return %1, %2 : tensor<4x4xf32>, tensor<4x4xf16>
  }

  // A CUBE scope is not copied just because a CUBE_AND_VECTOR op consumes it:
  // only the CUBE_AND_VECTOR op itself is split. Its VECTOR copy still reads
  // the single CUBE scope; upstream duplication is intentionally out of scope.
  // CHECK-LABEL: func.func @cube_scope_with_cav_downstream
  // CHECK: %[[CUBE:.*]] = scope.scope : () -> tensor<4x4xf16> {
  // CHECK: } {ssbuffer.block_id = {{[0-9]+}} : i32, ssbuffer.core_type = "CUBE", ssbuffer.load}
  // CHECK: %[[ADDC:.*]] = arith.addf %[[CUBE]], %[[CUBE]] {ssbuffer.block_id = {{[0-9]+}} : i32, ssbuffer.core_type = "CUBE"}
  // CHECK: %[[ADDV:.*]] = arith.addf %[[CUBE]], %[[CUBE]] {ssbuffer.block_id = {{[0-9]+}} : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: return %{{.*}}, %[[ADDV]]
  // CHECK-NOT: scope.scope
  func.func @cube_scope_with_cav_downstream(%gm: memref<4x4xf16>, %arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf32>, %rare: i1) -> (tensor<4x4xf32>, tensor<4x4xf16>) {
    %0 = scope.scope : () -> tensor<4x4xf16> {
      %alloc = memref.alloc() : memref<4x4xf16>
      %cst = arith.constant 0.0 : f16
      scf.if %rare {
        linalg.fill ins(%cst : f16) outs(%alloc : memref<4x4xf16>)
      }
      %src = memref.subview %gm[0, 0] [4, 4] [1, 1] : memref<4x4xf16> to memref<4x4xf16, strided<[4, 1]>>
      %dst = memref.subview %alloc[0, 0] [4, 4] [1, 1] : memref<4x4xf16> to memref<4x4xf16, strided<[4, 1]>>
      memref.copy %src, %dst : memref<4x4xf16, strided<[4, 1]>> to memref<4x4xf16, strided<[4, 1]>>
      %t = bufferization.to_tensor %alloc restrict writable : memref<4x4xf16> to tensor<4x4xf16>
      scope.return %t : tensor<4x4xf16>
    } {ssbuffer.core_type = "CUBE", ssbuffer.load}
    %1 = arith.addf %0, %0 {ssbuffer.core_type = "CUBE_AND_VECTOR"} : tensor<4x4xf16>
    return %arg1, %1 : tensor<4x4xf32>, tensor<4x4xf16>
  }
}
