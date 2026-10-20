// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_assume_memdep(%arg2: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg14: i32) {
    %c0_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %10 = arith.cmpi sge, %arg14, %c0_i32 {Undefined, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : i32
    llvm.intr.assume %10 : i1 {Undefined, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}

    %c4096_i32 = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} 4096 : i32
    %13 = arith.muli %arg14, %c4096_i32 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : i32
    %16 = arith.index_cast %13 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [32, 128], strides: [128, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<32x128xbf16, strided<[128, 1], offset: ?>>
    %alloc = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
    memref.copy %reinterpret_cast, %alloc {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16, strided<[128, 1], offset: ?>> to memref<32x128xbf16>

    return
  }
}

// CHECK-LABEL: func.func @test_assume_memdep
// CHECK-NOT: hivm.hir.sync_block_set
// CHECK-NOT: hivm.hir.sync_block_wait
// CHECK: return
