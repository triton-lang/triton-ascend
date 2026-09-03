// RUN: triton-opt --plan-compute-block %s | FileCheck %s

// A masked load lowers to linalg.fill (padding fill) + partial memref.copy
// into a freshly allocated buffer. When the loaded value feeds both a CUBE
// consumer (matmul) and a VECTOR consumer (here: arith.addf), the load chain
// is split into per-core copies. The fill has no SSA result and is not an
// operand of the load chain, so it used to stay CUBE-only: the VECTOR buffer
// copy kept uninitialized padding lanes. Both buffer copies must be filled.
//
// Cube vs Vector clone order is not stable across hosts, so CHECKs are DAG.
module {
  // CHECK-LABEL: func.func @masked_load_split_fill_cloned(
  // CHECK-DAG: %[[ALLOC_C:.*]] = memref.alloc() {{.*}}ssbuffer.core_type = "CUBE"} : memref<64x64xf32>
  // CHECK-DAG: linalg.fill {{.*}}ssbuffer.core_type = "CUBE"} ins({{.*}} : f32) outs(%[[ALLOC_C]] : memref<64x64xf32>)
  // CHECK-DAG: memref.copy {{.*}}ssbuffer.core_type = "CUBE"} {{.*}}to memref<32x64xf32, strided<[64, 1]>>
  // CHECK-DAG: %[[ALLOC_V:.*]] = memref.alloc() {{.*}}ssbuffer.core_type = "VECTOR"} : memref<64x64xf32>
  // CHECK-DAG: linalg.fill {{.*}}ssbuffer.core_type = "VECTOR"} ins({{.*}} : f32) outs(%[[ALLOC_V]] : memref<64x64xf32>)
  // CHECK-DAG: memref.copy {{.*}}ssbuffer.core_type = "VECTOR"} {{.*}}to memref<32x64xf32, strided<[64, 1]>>
  func.func @masked_load_split_fill_cloned(%gm: memref<?xf32>, %rhs: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %src = memref.reinterpret_cast %gm to offset: [%c0], sizes: [64, 64], strides: [32, 1]
      : memref<?xf32> to memref<64x64xf32, strided<[32, 1], offset: ?>>
    %alloc = memref.alloc() : memref<64x64xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
    %sv_src = memref.subview %src[0, 0] [32, 64] [1, 1]
      : memref<64x64xf32, strided<[32, 1], offset: ?>> to memref<32x64xf32, strided<[32, 1], offset: ?>>
    %sv_dst = memref.subview %alloc[0, 0] [32, 64] [1, 1]
      : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1]>>
    memref.copy %sv_src, %sv_dst
      : memref<32x64xf32, strided<[32, 1], offset: ?>> to memref<32x64xf32, strided<[64, 1]>>
    %t = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
    // VECTOR consumer keeps the load chain alive on the vector side
    %add = arith.addf %t, %t : tensor<64x64xf32>
    %out = tensor.empty() : tensor<64x64xf32>
    %init = linalg.fill ins(%cst : f32) outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm = linalg.matmul {input_precision = "ieee"}
      ins(%t, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
      outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    %res = arith.addf %mm, %add : tensor<64x64xf32>
    return %res : tensor<64x64xf32>
  }
}
