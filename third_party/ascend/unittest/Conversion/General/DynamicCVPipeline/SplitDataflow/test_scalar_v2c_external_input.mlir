// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync %s | FileCheck %s

// Scalar V->C dependency via external-input.
//
// A VECTOR block produces a 1D tensor (math.floor is a VECTOR-only op), then
// `tensor.extract`s a scalar from it. A CUBE block consumes that scalar
// (arith.addf). analyzeExternalInputs sees the scalar as a CUBE-block external
// input defined by a VECTOR op and records a V->C dependency.
//
// inter-core-transfer-and-sync then routes the scalar through the SSBuffer
// scalar channel:
//   VECTOR side: llvm.store %extracted + sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>]
//   CUBE side:   sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] + llvm.load
// The scalar PIPE_S sync stays isolated from the tensor flag space.
// CHECK-LABEL: func.func @scalar_v2c
// CHECK: %{{.*}} = tensor.extract
// CHECK: llvm.store volatile {{.*}} {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.crossCoreDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32} : f32, !llvm.ptr<11>
// CHECK: hivm.hir.sync_block_set {{.*}}[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 1
// CHECK: hivm.hir.sync_block_wait {{.*}}[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 1
// CHECK: llvm.load volatile {{.*}} {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.crossCoreDeps = [0 : i32, 0 : i32], ssbuffer.transfer_id = 0 : i32} : !llvm.ptr<11> -> f32
// The CUBE op must consume the loaded scalar, not the raw extract.
// CHECK: arith.addf
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @scalar_v2c(%arg0: memref<?xf32> {tt.tensor_kind = 0 : i32}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1.000000e+00 : f32
    %c0 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %t = tensor.empty() {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %filled = linalg.fill {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%t : tensor<64xf32>) -> tensor<64xf32>
    %floor = math.floor %filled {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %scalar = tensor.extract %floor[%c0] {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %cst_c = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 2.000000e+00 : f32
    %used = arith.addf %scalar, %cst_c {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : f32
    %dst = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
    %subview = memref.subview %dst[0] [1] [1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<1xf32, strided<[1]>> to memref<1xf32, strided<[1]>>
    %tt = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<1xf32>
    %filled_c = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%used : f32) outs(%tt : tensor<1xf32>) -> tensor<1xf32>
    %extracted_slice = tensor.extract_slice %filled_c[0] [1] [1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<1xf32> to tensor<1xf32>
    bufferization.materialize_in_destination %extracted_slice in writable %subview {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : (tensor<1xf32>, memref<1xf32, strided<[1]>>) -> ()
    return
  }
}
