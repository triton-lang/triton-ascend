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

// TC-03: llvm.store volatile (producer) and llvm.load volatile (consumer)
// Producer: crossDeps should be on the defining op of store's ptr operand (llvm.inttoptr)
// Consumer: crossDeps should be on the load op itself
// CHECK-LABEL: func.func @tc_tag_03_store_load_volatile
// Producer llvm.inttoptr should have crossDeps = [3, 1]
// CHECK: llvm.inttoptr{{.*}}ssbuffer.crossDeps = [3 : i32, 1 : i32]
// llvm.store volatile should NOT have crossDeps
// CHECK-NOT: llvm.store volatile{{.*}}ssbuffer.crossDeps
// llvm.load volatile should have crossDeps = [3, 0]
// CHECK: llvm.load volatile{{.*}}ssbuffer.crossDeps = [3 : i32, 0 : i32]
// Consumer llvm.inttoptr should NOT have crossDeps
// CHECK-NOT: llvm.inttoptr{{.*}}ssbuffer.crossDeps
func.func @tc_tag_03_store_load_volatile() {
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %cst_2 = arith.constant 9.99999997E-7 : f32
  %c64_i64 = llvm.mlir.constant(64 : i64) : i64

  // Producer block (VECTOR)
  scope.scope : () -> () {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %extracted = arith.constant {ssbuffer.block_id = 11 : i32} 1.000000e+00 : f32
    %ptr = llvm.inttoptr %0 {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 3 : i32} : i64 to !llvm.ptr<11>
    hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 2
    llvm.store volatile %extracted, %ptr {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 3 : i32} : f32, !llvm.ptr<11>
    hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 2
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

  // Consumer block (CUBE)
  scope.scope : () -> () {
    %1 = llvm.mlir.constant(0 : i64) : i64
    %ptr_consumer = llvm.inttoptr %1 {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32} : i64 to !llvm.ptr<11>
    hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 2
    %loaded = llvm.load volatile %ptr_consumer {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32} : !llvm.ptr<11> -> f32
    %cmp = arith.cmpf ogt, %loaded, %cst_2 {ssbuffer.block_id = 4 : i32} : f32
    %result = scf.if %cmp -> (f32) {
      %div = arith.divf %cst_1, %loaded {ssbuffer.block_id = 4 : i32} : f32
      scf.yield %div : f32
    } else {
      scf.yield %cst_1 : f32
    }
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
