// RUN: triton-opt %s --triton-to-linalg --verify-each | FileCheck %s
// RUN: triton-opt %s --triton-to-unstructure --triton-to-linalg --verify-each | FileCheck %s

// One complete offset has an earlier unmarked user and two marked users in
// sibling regions. CFO's proof applies to this SSA, so sampled fields must
// dominate every use, including the earlier load.
// CHECK-LABEL: func.func @shared_descriptor_offset
// CHECK: memref.copy
// CHECK: scf.if
// CHECK: memref.copy
// CHECK: } else {
// CHECK: memref.copy
// CHECK: return
module {
  tt.func public @shared_descriptor_offset(%indices: !tt.ptr<i32>, %base: !tt.ptr<f32>, %out: !tt.ptr<f32>, %pred: i1) {
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %ib = tt.splat %indices : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %ip = tt.addptr %ib, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %offset = tt.load %ip : tensor<4x!tt.ptr<i32>>
    %b = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %o = tt.splat %out : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %op = tt.addptr %o, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %early = tt.addptr %b, %offset : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %v = tt.load %early : tensor<4x!tt.ptr<f32>>
    tt.store %op, %v : tensor<4x!tt.ptr<f32>>
    scf.if %pred {
      %p = tt.addptr %b, %offset {PointerDescriptorRebuild, PointerDescriptorStructuredAxes = array<i32: 1>} : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %x = tt.load %p : tensor<4x!tt.ptr<f32>>
      tt.store %op, %x : tensor<4x!tt.ptr<f32>>
    } else {
      %p = tt.addptr %b, %offset {PointerDescriptorRebuild, PointerDescriptorStructuredAxes = array<i32: 1>} : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %x = tt.load %p : tensor<4x!tt.ptr<f32>>
      tt.store %op, %x : tensor<4x!tt.ptr<f32>>
    }
    tt.return
  }
}
