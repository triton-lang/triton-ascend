// RUN: triton-opt --add_multi_buffer_outer_scope %s | FileCheck %s --dump-input=fail

// UT: ssbuffer.crossDeps + ssbuffer.cross_buffer tag verification
// NOTE: double-buffer ifOp creation requires outer-loop extra_sync structure
// (ssbuffer.main_loop=0). This UT has single-layer loop, so only crossDeps
// on allocs and transfer ops are verified.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {

// TC-01: C→V
// CHECK-LABEL: func.func @tc_tag_01_ctov
// CHECK: ssbuffer.crossDeps = [1 : i32, 1 : i32]
// CHECK: ssbuffer.crossDeps = [1 : i32, 0 : i32]
// CHECK-NOT: transfer_id = -1
func.func @tc_tag_01_ctov() {
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i32 = arith.constant 1 : i32

  scope.scope : () -> () {
    %buf_ub = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.for %i = %c0_i32 to %c128_i32 step %c1_i32 iter_args() -> () : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
      %buf = memref.memory_space_cast %buf_ub {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>> to memref<128xf16>
      %t = bufferization.to_tensor %buf restrict writable {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

  scope.scope : () -> () {
    %src = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cc>>
    %dst = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    scf.for %i = %c0_i32 to %c128_i32 step %c1_i32 iter_args() -> () : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
      hivm.hir.fixpipe {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} ins(%src : memref<128xf16, #hivm.address_space<cc>>) outs(%dst : memref<128xf16, #hivm.address_space<cbuf>>)
      hivm.hir.sync_block_set {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}

  return
}

// TC-02: V→C
// CHECK-LABEL: func.func @tc_tag_02_vtoc
// CHECK: ssbuffer.crossDeps = [2 : i32, 1 : i32]
// CHECK: ssbuffer.crossDeps = [2 : i32, 0 : i32]
// CHECK-NOT: transfer_id = -1
func.func @tc_tag_02_vtoc() {
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i32 = arith.constant 1 : i32
  %src_tensor = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<128xf16>

  scope.scope : () -> () {
    %dst = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    scf.for %i = %c0_i32 to %c128_i32 step %c1_i32 iter_args() -> () : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 4
      hivm.hir.copy ins(%src_tensor : tensor<128xf16>) outs(%dst : memref<128xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 2 : i32}
      hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

  scope.scope : () -> () {
    %src_cvt = memref.alloc() {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    scf.for %i = %c0_i32 to %c128_i32 step %c1_i32 iter_args() -> () : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
      %cvt = hivm.hir.convert_layout %src_cvt output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
      %msc = memref.memory_space_cast %cvt {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf16, #hivm.address_space<cbuf>> to memref<128x128xf16>
      %t = bufferization.to_tensor %msc restrict writable {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf16> to tensor<128x128xf16>
      hivm.hir.sync_block_set {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 4
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}

  return
}

}
