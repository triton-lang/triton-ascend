// RUN: triton-opt "--triton-to-linalg=global-kernel=false named-ops=True" --split-input-file %s | FileCheck %s

// The precise PointerCastOp rewrite must keep the source memory space. A TLA
// pointer entry carries a GM address space on its 1-D pointer_cast; dropping it
// during the rewrite makes the downstream reinterpret_cast compare a plain
// memref<?xf32> against a GM result type and fail verification.

// CHECK-LABEL: func.func @test_gm_pointer_entry
// CHECK: %[[POINTER:.*]] = hivm.hir.pointer_cast({{.*}}) [{{.*}}] : memref<?xf32, #hivm.address_space<gm>>
// CHECK: memref.reinterpret_cast %[[POINTER]] to offset: [0], sizes: [128], strides: [1]
// CHECK-SAME: : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @test_gm_pointer_entry(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %addr = tt.ptr_to_int %arg0 : !tt.ptr<f32> -> i64
    %cast = hivm.hir.pointer_cast(%addr) : memref<1xf32, #hivm.address_space<gm>>
    %view = memref.reinterpret_cast %cast to offset: [%c0], sizes: [128], strides: [1] : memref<1xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%view : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<128xf32, #hivm.address_space<ub>>)
    tt.return
  }
}

