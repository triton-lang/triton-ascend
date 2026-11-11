// RUN: triton-opt '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true' --triton-to-linalg --split-input-file %s | FileCheck %s

// Tests for PtrAnalysis::visitOperandBitcast.
//
// For bool tensors the frontend casts ptr<i1> to ptr<i8> before load/store, so
// the pointer chain is broadcast -> addptr -> bitcast -> load. Before the fix
// PtrAnalysis stopped at the bitcast, kept the full [2,4] pointer and the
// broadcast axis ended up with stride 0 in the memref descriptor, which
// BiShengIR rejects with "strides must not be zero".
//
// After the fix the analysis looks through the cast, the broadcast axis is
// dropped from the pointer tile and restored with linalg.broadcast.

// Case 1: 2D bool load with a broadcast row axis.
// The [2,4] pointer is compressed to [2]; the missing axis is re-materialized
// by linalg.broadcast, so the memref descriptor only has stride 1.

module {
  tt.func public @bool_load_bitcast_broadcast(%base: !tt.ptr<i1>) -> tensor<2x4xi8> {
    %r = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
    %o = tt.expand_dims %r {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %b = tt.broadcast %o : tensor<2x1xi32> -> tensor<2x4xi32>
    %sp = tt.splat %base : !tt.ptr<i1> -> tensor<2x4x!tt.ptr<i1>>
    %p = tt.addptr %sp, %b : tensor<2x4x!tt.ptr<i1>>, tensor<2x4xi32>
    %bc = tt.bitcast %p : tensor<2x4x!tt.ptr<i1>> -> tensor<2x4x!tt.ptr<i8>>
    %v = tt.load %bc : tensor<2x4x!tt.ptr<i8>>
    tt.return %v : tensor<2x4xi8>
  }
}

// CHECK-LABEL: func.func @bool_load_bitcast_broadcast
// The descriptor must not carry the zero stride broadcast axis.
// CHECK: memref.reinterpret_cast {{.*}} sizes: [2], strides: [1]
// CHECK-NOT: strides: [{{.*}}, 0]
// The broadcast axis is restored after the load.
// CHECK: linalg.broadcast
// CHECK-SAME: dimensions = [1]

// -----

// Case 2: 3D bool load, the same structure the fused kernel produces:
// out_ptrs is [M, 1, N] where the middle axis comes from arange(0, 1) * 0.
// The zero stride axis must be dropped and restored by linalg.broadcast.

module {
  tt.func public @bool_load_bitcast_middle_axis(%base: !tt.ptr<i1>) -> tensor<2x1x4xi8> {
    %r = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
    %e1 = tt.expand_dims %r {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %e2 = tt.expand_dims %e1 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
    %o = tt.broadcast %e2 : tensor<2x1x1xi32> -> tensor<2x1x4xi32>
    %sp = tt.splat %base : !tt.ptr<i1> -> tensor<2x1x4x!tt.ptr<i1>>
    %p = tt.addptr %sp, %o : tensor<2x1x4x!tt.ptr<i1>>, tensor<2x1x4xi32>
    %bc = tt.bitcast %p : tensor<2x1x4x!tt.ptr<i1>> -> tensor<2x1x4x!tt.ptr<i8>>
    %v = tt.load %bc : tensor<2x1x4x!tt.ptr<i8>>
    tt.return %v : tensor<2x1x4xi8>
  }
}

// CHECK-LABEL: func.func @bool_load_bitcast_middle_axis
// CHECK: memref.reinterpret_cast {{.*}} sizes: [2, 1], strides: [1, 1]
// CHECK-NOT: strides: [{{.*}}, 0]
// CHECK: linalg.broadcast
// CHECK-SAME: dimensions = [2]

// -----

// Case 3: non bool control. The same pointer structure without the frontend
// cast must behave identically, which shows the compression is driven by the
// broadcast axis and not by the cast itself.

module {
  tt.func public @i8_load_no_bitcast_broadcast(%base: !tt.ptr<i8>) -> tensor<2x4xi8> {
    %r = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
    %o = tt.expand_dims %r {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %b = tt.broadcast %o : tensor<2x1xi32> -> tensor<2x4xi32>
    %sp = tt.splat %base : !tt.ptr<i8> -> tensor<2x4x!tt.ptr<i8>>
    %p = tt.addptr %sp, %b : tensor<2x4x!tt.ptr<i8>>, tensor<2x4xi32>
    %v = tt.load %p : tensor<2x4x!tt.ptr<i8>>
    tt.return %v : tensor<2x4xi8>
  }
}

// CHECK-LABEL: func.func @i8_load_no_bitcast_broadcast
// CHECK: memref.reinterpret_cast {{.*}} sizes: [2, 1], strides: [1, 1]
// CHECK-NOT: strides: [{{.*}}, 0]
// CHECK: linalg.broadcast
// CHECK-SAME: dimensions = [1]
