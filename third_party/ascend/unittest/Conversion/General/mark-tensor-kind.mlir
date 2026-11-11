// RUN: triton-opt %s --mark-tensor-kind | FileCheck %s
// RUN: triton-opt %s --mark-tensor-kind --mark-tensor-kind | FileCheck %s

// Unused pointers at the beginning, middle and end must retain their positions.
// Scalars do not consume a profiling tensor slot.
// CHECK-LABEL: tt.func public @unused_pointers(
// CHECK-SAME: %arg0: !tt.ptr<f32> {tt.tensor_kind = -1 : i32}
// CHECK-SAME: %arg1: !tt.ptr<f32> {tt.tensor_kind = 0 : i32}
// CHECK-SAME: %arg2: i32,
// CHECK-SAME: %arg3: !tt.ptr<f32> {tt.tensor_kind = -1 : i32}
// CHECK-SAME: %arg4: !tt.ptr<f32> {tt.tensor_kind = 1 : i32}
// CHECK-SAME: %arg5: !tt.ptr<f32> {tt.tensor_kind = 2 : i32}
// CHECK-SAME: %arg6: !tt.ptr<f32> {tt.tensor_kind = -1 : i32}
tt.func public @unused_pointers(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                               %arg2: i32, %arg3: !tt.ptr<f32>,
                               %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>,
                               %arg6: !tt.ptr<f32>) {
  %value = tt.load %arg1 : !tt.ptr<f32>
  tt.store %arg4, %value : !tt.ptr<f32>
  %old = tt.load %arg5 : !tt.ptr<f32>
  %sum = arith.addf %value, %old : f32
  tt.store %arg5, %sum : !tt.ptr<f32>
  tt.return
}

// Existing direction metadata must not be replaced by NONE.
// CHECK-LABEL: tt.func public @existing_kinds(
// CHECK-SAME: %arg0: !tt.ptr<f32> {tt.tensor_kind = 1 : i32}
// CHECK-SAME: %arg1: !tt.ptr<f32> {tt.tensor_kind = 0 : i32}
// CHECK-SAME: %arg2: !tt.ptr<f32> {tt.tensor_kind = 2 : i32}
// CHECK-SAME: %arg3: !tt.ptr<f32> {tt.tensor_kind = -1 : i32}
tt.func public @existing_kinds(%arg0: !tt.ptr<f32> {tt.tensor_kind = 1 : i32},
                              %arg1: !tt.ptr<f32> {tt.tensor_kind = 0 : i32},
                              %arg2: !tt.ptr<f32> {tt.tensor_kind = 2 : i32},
                              %arg3: !tt.ptr<f32> {tt.tensor_kind = -1 : i32}) {
  tt.return
}

// Internal helpers do not contribute slots to launcher profiling metadata.
// CHECK-LABEL: tt.func private @helper(
// CHECK-SAME: %arg0: !tt.ptr<f32>, %arg1: i32)
tt.func private @helper(%arg0: !tt.ptr<f32>, %arg1: i32) {
  tt.return
}
