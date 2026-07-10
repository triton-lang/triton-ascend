// RUN: triton-opt --allow-unregistered-dialect --async-load-hoisting %s | FileCheck %s

// Unit Test: SubViewOp with non-CopyOp users
// Regression test for: segfault when a SubViewOp result has users that are
// NOT memref::CopyOp (e.g., bufferization.to_tensor).
// Previously, llvm::dyn_cast<memref::CopyOp> would return nullptr, and
// visited.insert(nullptr).second would succeed, passing nullptr down the
// recursion chain, causing a crash.

module {
  func.func @subview_noncopy_users(%arg0: memref<?xf16>, %arg1: i64) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index

    // block_id=10: cache group with annotation.mark
    %alloc = memref.alloc() {ssbuffer.block_id = 10 : i32} : memref<128x32xf16>

    %offset = arith.index_cast %arg1 {ssbuffer.block_id = 10 : i32} : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%offset], sizes: [1, 32], strides: [32, 1] {ssbuffer.block_id = 10 : i32} : memref<?xf16> to memref<1x32xf16, strided<[32, 1], offset: ?>>
    %subview = memref.subview %alloc[%c0, %c0] [1, 32] [1, 1] {ssbuffer.block_id = 10 : i32} : memref<128x32xf16> to memref<1x32xf16, strided<[32, 1], offset: ?>>
    memref.copy %reinterpret_cast, %subview {ssbuffer.block_id = 10 : i32} : memref<1x32xf16, strided<[32, 1], offset: ?>> to memref<1x32xf16, strided<[32, 1], offset: ?>>

    // Key: bufferization.to_tensor uses %subview but is NOT a CopyOp
    // CHECK: bufferization.to_tensor
    %tensor = bufferization.to_tensor %subview {ssbuffer.block_id = 10 : i32} : memref<1x32xf16, strided<[32, 1], offset: ?>>

    // annotation.mark uses %alloc — triggers chainContainsBlockArg path
    annotation.mark %alloc {MayImplicitTransposeWithLastAxis, ssbuffer.block_id = 10 : i32} : memref<128x32xf16>

    // Load from subview using data loaded by CopyOp — should get gm_load_bufferable
    // CHECK: bufferization.to_tensor
    // CHECK-SAME: gm_load_bufferable
    %subview_2 = memref.subview %alloc[%c0, %c0] [1, 32] [1, 1] {ssbuffer.block_id = 10 : i32} : memref<128x32xf16> to memref<1x32xf16, strided<[32, 1], offset: ?>>
    %loaded = bufferization.to_tensor %subview_2 {ssbuffer.block_id = 10 : i32} : memref<1x32xf16, strided<[32, 1], offset: ?>>

    return
  }
}
