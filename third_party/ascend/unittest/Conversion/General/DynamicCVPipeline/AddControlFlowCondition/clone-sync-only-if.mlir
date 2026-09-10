// RUN: triton-opt --clone-ops --verify-each %s | FileCheck %s

// The memory effects of the shared L1 reader must not keep a cloned readiness
// notification alive in the consumer block. Otherwise one produced tile sends
// two notifications, allowing the vector consumer to read the next tile early.
// Keep the views before the notification so the memory graph tracks their
// aliases when it analyzes the cloned notification and the live to_tensor.
// CHECK-LABEL: func.func @shared_l1_reader
// CHECK: scf.for
// CHECK: scf.if
// CHECK: hivm.hir.sync_block_set
// CHECK: hivm.hir.sync_block_set
// CHECK: } {ssbuffer.block_id = 1 : i32
// CHECK-NEXT: %[[LAYOUT:.*]] = hivm.hir.convert_layout {{.*}}ssbuffer.block_id = 3 : i32, ssbuffer.clone = 1 : i32
// CHECK-NEXT: %[[CAST:.*]] = memref.memory_space_cast %[[LAYOUT]]
// CHECK-NEXT: %[[TENSOR:.*]] = bufferization.to_tensor %[[CAST]]
// CHECK-NEXT: hivm.hir.copy ins(%[[TENSOR]]
// CHECK-NOT: hivm.hir.sync_block_set
// CHECK: return

func.func @shared_l1_reader(%out: memref<16x16xf16>, %upper: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f16
  scope.scope : () -> () {
    %buffer = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    scf.for %iv = %c0 to %upper step %c1 {
      linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.shared_page_write} ins(%zero : f16) outs(%buffer : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      %layout = hivm.hir.convert_layout %buffer output_shape [16, 16] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32} : (memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) -> memref<16x16xf16, #hivm.address_space<cbuf>>
      %cast = memref.memory_space_cast %layout {ssbuffer.block_id = 1 : i32} : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      %condition = arith.cmpi eq, %iv, %c0 {ssbuffer.block_id = 1 : i32} : index
      scf.if %condition {
        hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
      } else {
        hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
      } {ssbuffer.block_id = 1 : i32, ssbuffer.cross_buffer = 1 : i32}
      %tensor = bufferization.to_tensor %cast restrict writable {ssbuffer.block_id = 3 : i32} : memref<16x16xf16> to tensor<16x16xf16>
      hivm.hir.copy ins(%tensor : tensor<16x16xf16>) outs(%out : memref<16x16xf16>) {ssbuffer.block_id = 3 : i32}
    } {ssbuffer.main_loop = 0 : i32}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}
