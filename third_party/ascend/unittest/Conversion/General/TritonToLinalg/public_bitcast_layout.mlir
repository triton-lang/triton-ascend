// RUN: triton-opt %s --triton-to-linalg --verify-each --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @bitcast_reverse
// CHECK: %[[CAP_REVERSE:.*]] = arith.constant 4 : index
// CHECK: hivm.hir.pointer_cast(%{{.*}}) [%[[CAP_REVERSE]]] : memref<?xi32>
// CHECK: memref.reinterpret_cast {{.*}}strides: [-1]
// CHECK: return
module {
  tt.func public @bitcast_reverse(%base: !tt.ptr<f32>, %out: !tt.ptr<i32>) {
    %cast = tt.bitcast %base : !tt.ptr<f32> -> !tt.ptr<i32>
    %b = tt.splat %cast : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %o = tt.splat %out : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %three = arith.constant dense<3> : tensor<4xi32>
    %offset = arith.subi %three, %r : tensor<4xi32>
    %p = tt.addptr %b, %offset : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %q = tt.addptr %o, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %v = tt.load %p : tensor<4x!tt.ptr<i32>>
    tt.store %q, %v : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @bitcast_zero
// CHECK: %[[ADDRESS:.*]] = arith.index_cast %{{.*}} : index to i64
// CHECK: %[[CAP_ZERO:.*]] = arith.constant 1 : index
// CHECK: hivm.hir.pointer_cast(%[[ADDRESS]]) [%[[CAP_ZERO]]] : memref<?xi32>
// CHECK: memref.reinterpret_cast {{.*}}strides: [0]
// CHECK: return
module {
  tt.func public @bitcast_zero(%base: !tt.ptr<f32>, %out: !tt.ptr<i32>) {
    %cast = tt.bitcast %base : !tt.ptr<f32> -> !tt.ptr<i32>
    %b = tt.splat %cast : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %o = tt.splat %out : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %offset = arith.constant dense<0> : tensor<4xi32>
    %p = tt.addptr %b, %offset : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %q = tt.addptr %o, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %v = tt.load %p : tensor<4x!tt.ptr<i32>>
    tt.store %q, %v : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @bitcast_dynamic
// CHECK: arith.maxsi
// CHECK: hivm.hir.pointer_cast
// CHECK: return
module {
  tt.func public @bitcast_dynamic(%base: !tt.ptr<f32>, %out: !tt.ptr<i32>, %stride: i32) {
    %cast = tt.bitcast %base : !tt.ptr<f32> -> !tt.ptr<i32>
    %b = tt.splat %cast : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %o = tt.splat %out : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %s = tt.splat %stride : i32 -> tensor<4xi32>
    %offset = arith.muli %r, %s : tensor<4xi32>
    %p = tt.addptr %b, %offset : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %q = tt.addptr %o, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %v = tt.load %p : tensor<4x!tt.ptr<i32>>
    tt.store %q, %v : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @bitcast_matrix
// CHECK: %[[CAP_MATRIX:.*]] = arith.constant 16 : index
// CHECK: hivm.hir.pointer_cast(%{{.*}}) [%[[CAP_MATRIX]]] : memref<?xi32>
// CHECK: memref.reinterpret_cast {{.*}}strides: [8, 1]
// CHECK: return
module {
  tt.func public @bitcast_matrix(%base: !tt.ptr<f32>, %out: !tt.ptr<i32>) {
    %cast = tt.bitcast %base : !tt.ptr<f32> -> !tt.ptr<i32>
    %b = tt.splat %cast : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %o = tt.splat %out : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %row = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
    %col = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %eight = arith.constant dense<8> : tensor<2xi32>
    %r = arith.muli %row, %eight : tensor<2xi32>
    %rx = tt.expand_dims %r {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %cx = tt.expand_dims %col {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %rb = tt.broadcast %rx : tensor<2x1xi32> -> tensor<2x8xi32>
    %cb = tt.broadcast %cx : tensor<1x8xi32> -> tensor<2x8xi32>
    %offset = arith.addi %rb, %cb : tensor<2x8xi32>
    %p = tt.addptr %b, %offset : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    %q = tt.addptr %o, %offset : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    %v = tt.load %p : tensor<2x8x!tt.ptr<i32>>
    tt.store %q, %v : tensor<2x8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// A local reverse view must start at the lowest address, with the first
// logical element at offset three. Rebasing to base+3 with offset zero would
// put three accessed elements before the described allocation.
// CHECK-LABEL: func.func @local_reverse_carrier
// CHECK: %[[SPAN:.*]] = arith.constant 4 : index
// CHECK: %[[CAST:.*]] = hivm.hir.pointer_cast({{.*}}) [%[[SPAN]]] : memref<?xi32>
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[CAST]] to offset: [3], sizes: [4], strides: [-1]
// CHECK: memref.load %[[VIEW]]
// CHECK: return
module {
  func.func @local_reverse_carrier(%address: i64, %index: index) -> i32 {
    %placeholder = arith.constant 1 : index
    %base = hivm.hir.pointer_cast(%address) [%placeholder] {ScalarPointerCarrier} : memref<?xi32>
    %view = memref.reinterpret_cast %base to offset: [3], sizes: [4], strides: [-1] : memref<?xi32> to memref<4xi32, strided<[-1], offset: 3>>
    %value = memref.load %view[%index] : memref<4xi32, strided<[-1], offset: 3>>
    return %value : i32
  }
}
