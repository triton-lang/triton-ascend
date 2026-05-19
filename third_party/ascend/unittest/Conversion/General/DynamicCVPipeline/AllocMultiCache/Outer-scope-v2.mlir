// RUN: triton-opt --add_multi_buffer_outer_scope %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @tc_v2_01_ctov() {
  %c0 = arith.constant 0 : i32
  %c100 = arith.constant 128 : i32
  %c1 = arith.constant 1 : i32
  scope.scope : () -> () {
    %buf = memref.alloc() {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.for %i = %c0 to %c100 step %c1 iter_args() -> () : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
      %m = memref.memory_space_cast %buf {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>> to memref<128xf16>
      %t = bufferization.to_tensor %m restrict writable : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}
func.func @tc_v2_02_vtoc() {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %c1 = arith.constant 1 : i32
  %t0 = tensor.empty() {ssbuffer.block_id = 50 : i32} : tensor<64x64xf16>
  scope.scope : () -> () {
    %buf = memref.alloc() {ssbuffer.block_id = 50 : i32, ssbuffer.transfer_id = 2 : i32} : memref<64x64xf16, #hivm.address_space<cbuf>>
    scf.for %i = %c0 to %c128 step %c1 iter_args() -> () : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 50 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 5
      hivm.hir.copy ins(%t0 : tensor<64x64xf16>) outs(%buf : memref<64x64xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 50 : i32, ssbuffer.transfer_id = 2 : i32}
      hivm.hir.sync_block_set {ssbuffer.block_id = 50 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 5
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}
func.func @tc_v2_03_cube_receiver_convert() {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %c1 = arith.constant 1 : i32
  %t0 = tensor.empty() {ssbuffer.block_id = 60 : i32} : tensor<64x64xf16>
  scope.scope : () -> () {
    %buf = memref.alloc() {ssbuffer.block_id = 60 : i32, ssbuffer.transfer_id = 3 : i32} : memref<64x64xf16, #hivm.address_space<cbuf>>
    scf.for %i = %c0 to %c128 step %c1 iter_args() -> () : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 60 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 7
      hivm.hir.copy ins(%t0 : tensor<64x64xf16>) outs(%buf : memref<64x64xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 60 : i32, ssbuffer.transfer_id = 3 : i32}
      hivm.hir.sync_block_set {ssbuffer.block_id = 60 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 7
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  hivm.hir.sync_block_set {ssbuffer.block_id = 60 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 7
  hivm.hir.sync_block_wait {ssbuffer.block_id = 70 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 7
  scope.scope : () -> () {
    %buf2 = memref.alloc() {ssbuffer.block_id = 70 : i32, ssbuffer.transfer_id = 3 : i32} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    scf.for %i = %c0 to %c128 step %c1 iter_args() -> () : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 70 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 7
      %cvt = hivm.hir.convert_layout %buf2 output_shape [64, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 70 : i32, ssbuffer.transfer_id = 3 : i32} : (memref<4x4x16x16xf16, #hivm.address_space<cbuf>>) -> memref<64x64xf16, #hivm.address_space<cbuf>>
      %m = memref.memory_space_cast %cvt {ssbuffer.block_id = 70 : i32, ssbuffer.transfer_id = 3 : i32} : memref<64x64xf16, #hivm.address_space<cbuf>> to memref<64x64xf16>
      %t = bufferization.to_tensor %m restrict writable {ssbuffer.block_id = 70 : i32, ssbuffer.transfer_id = 3 : i32} : memref<64x64xf16> to tensor<64x64xf16>
      hivm.hir.sync_block_set {ssbuffer.block_id = 70 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 7
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}
}

// crossDeps on receiver-side alloc [tid,1] and transferOp [tid,0]
// CHECK: crossDeps = [1 : i32, 0 : i32]
// CHECK: crossDeps = [3 : i32, 1 : i32]
// CHECK: crossDeps = [3 : i32, 0 : i32]
