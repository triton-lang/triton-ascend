// RUN: triton-opt --add_multi_buffer_outer_scope %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @tc_os02_vtoc() {
  %c0_i32 = arith.constant 0 : i32
  %c100_i32 = arith.constant 100 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<128xf16>
  scope.scope : () -> () {
    %buf_cbuf = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args() -> () : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 4
      hivm.hir.copy ins(%0 : tensor<128xf16>) outs(%buf_cbuf : memref<128xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 2 : i32}
      hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 4
  hivm.hir.sync_block_wait {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
  scope.scope : () -> () {
    %buf_cbuf2 = memref.alloc() {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args() -> () : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
      %cvt = hivm.hir.convert_layout %buf_cbuf2 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
      %m = memref.memory_space_cast %cvt {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf16, #hivm.address_space<cbuf>> to memref<128x128xf16>
      %t = bufferization.to_tensor %m restrict writable {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf16> to tensor<128x128xf16>
      hivm.hir.sync_block_set {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 4
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}
}

// CHECK: crossDeps = [2 : i32, 1 : i32]
// CHECK: crossDeps = [2 : i32, 0 : i32]
