// RUN: triton-opt --clone-ops --verify-each %s | FileCheck %s

// A sync op wrapped by multi-buffering in an scf.if is cloned from block 1
// into block 3 and survives cleanup, because the consumer reads the same
// buffer and the memory dependences keep the clone alive. The kClone marker is
// on the scf.if, not on the nested sync op, so validateClonedSyncOpsErased has
// to look inside it and fall back instead of letting the duplicated handshake
// through.

// CHECK-LABEL: module attributes
// CHECK-SAME: triton_ascend.dynamic_cv_pipeline.rc = 1

func.func @cloned_wrapped_sync(%out: memref<16x16xf16>, %upper: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f16
  scope.scope : () -> () {
    %buffer = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    scf.for %iv = %c0 to %upper step %c1 {
      linalg.fill {ssbuffer.block_id = 1 : i32} ins(%zero : f16) outs(%buffer : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      %layout = hivm.hir.convert_layout %buffer output_shape [16, 16] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32} : (memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) -> memref<16x16xf16, #hivm.address_space<cbuf>>
      %cast = memref.memory_space_cast %layout {ssbuffer.block_id = 1 : i32} : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      %condition = arith.cmpi eq, %iv, %c0 {ssbuffer.block_id = 1 : i32} : index
      scf.if %condition {
        hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
      } else {
        hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
      } {ssbuffer.block_id = 1 : i32}
      %tensor = bufferization.to_tensor %cast restrict writable {ssbuffer.block_id = 3 : i32} : memref<16x16xf16> to tensor<16x16xf16>
      hivm.hir.copy ins(%tensor : tensor<16x16xf16>) outs(%out : memref<16x16xf16>) {ssbuffer.block_id = 3 : i32}
    } {ssbuffer.main_loop = 0 : i32}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}
