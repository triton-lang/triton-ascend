// RUN: triton-opt %s --triton-to-linalg --verify-each | FileCheck %s

// One pointer producer has two independent effectful users. Expand each at
// its own use site, and normalize the scalar store and atomic before taking
// the conversion snapshot.
// CHECK-LABEL: func.func @shared_indirect_pointer
// CHECK: scf.for
// CHECK: bufferization.materialize_in_destination
// CHECK: scf.for
// CHECK: hivm
// CHECK: return
module {
  tt.func public @shared_indirect_pointer(%indices: !tt.ptr<i32>, %base: !tt.ptr<i32>) {
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %indexbase = tt.splat %indices : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %indexptr = tt.addptr %indexbase, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %offsets = tt.load %indexptr : tensor<4x!tt.ptr<i32>>
    %bases = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %ptrs = tt.addptr %bases, %offsets : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %ones = arith.constant dense<1> : tensor<4xi32>
    tt.store %ptrs, %ones : tensor<4x!tt.ptr<i32>>
    %previous = tt.atomic_rmw add, acq_rel, gpu, %ptrs, %ones : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
    tt.return
  }
}
