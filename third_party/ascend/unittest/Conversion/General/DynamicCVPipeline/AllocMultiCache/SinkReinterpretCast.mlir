// RUN: triton-opt --ssbuf-sink-reinterpret-cast %s | FileCheck %s

// Test: sink memref.reinterpret_cast before every user.
// - cast defined in parent block, one user per scf.if (memref.subview)
// - clone inserted immediately before each user; block may split into block_ids later
// - original erased once all uses redirected, so cast appears once per scf.if

// CHECK-LABEL: func.func @test_sink_reinterpret_cast
// CHECK: scf.if
// CHECK-NEXT: %{{.+}} = memref.reinterpret_cast %arg0 to offset: [%{{.+}}], sizes: [64], strides: [8]
// CHECK-NEXT: %{{.+}} = memref.subview
// CHECK: scf.if
// CHECK-NEXT: %{{.+}} = memref.reinterpret_cast %arg0 to offset: [%{{.+}}], sizes: [64], strides: [8]
// CHECK-NEXT: %{{.+}} = memref.subview
// CHECK: return

module {
  func.func @test_sink_reinterpret_cast(%arg0: memref<?xf32>, %arg1: i32) -> index {
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %idx = arith.index_cast %arg1 : i32 to index
    %offset = arith.muli %idx, %c8 : index

    // reinterpret_cast defined in the parent block — should be removed by the pass
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%offset], sizes: [64], strides: [8] : memref<?xf32> to memref<64xf32, strided<[8], offset: ?>>

    %cond = arith.cmpi ne, %arg1, %c0_i32 : i32
    %result1 = scf.if %cond -> (index) {
      %subview = memref.subview %reinterpret_cast[0] [%c64] [1] : memref<64xf32, strided<[8], offset: ?>> to memref<?xf32, strided<[8], offset: ?>>
      %dim = memref.dim %subview, %c0 : memref<?xf32, strided<[8], offset: ?>>
      scf.yield %dim : index
    } else {
      scf.yield %c0 : index
    }

    %cond2 = arith.cmpi eq, %arg1, %c0_i32 : i32
    %result2 = scf.if %cond2 -> (index) {
      %subview2 = memref.subview %reinterpret_cast[0] [%c64] [1] : memref<64xf32, strided<[8], offset: ?>> to memref<?xf32, strided<[8], offset: ?>>
      %dim2 = memref.dim %subview2, %c0 : memref<?xf32, strided<[8], offset: ?>>
      scf.yield %dim2 : index
    } else {
      scf.yield %c0 : index
    }

    %sum = arith.addi %result1, %result2 : index
    return %sum : index
  }
}
