// RUN: triton-opt --mark-tensor-kind %s | FileCheck %s

// CHECK-LABEL: tt.func public @unused_middle(
// CHECK-SAME: %arg0: !tt.ptr<i32> {tt.tensor_kind = 0 : i32}
// CHECK-SAME: %arg1: !tt.ptr<i32> {tt.tensor_kind = -1 : i32}
// CHECK-SAME: %arg2: !tt.ptr<i32> {tt.tensor_kind = 1 : i32}
// CHECK-SAME: %arg3: i32
tt.func public @unused_middle(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>,
                              %arg2: !tt.ptr<i32>, %arg3: i32) {
  %value = tt.load %arg0 : !tt.ptr<i32>
  tt.store %arg2, %value : !tt.ptr<i32>
  tt.return
}

// CHECK-LABEL: tt.func public @pointer_from_if(
// CHECK-SAME: %arg0: !tt.ptr<i32> {tt.tensor_kind = 0 : i32}
// CHECK-SAME: %arg1: !tt.ptr<i32> {tt.tensor_kind = 0 : i32}
// CHECK-SAME: %arg2: !tt.ptr<i32> {tt.tensor_kind = 1 : i32}
tt.func public @pointer_from_if(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>,
                                %arg2: !tt.ptr<i32>, %condition: i1) {
  %selected = scf.if %condition -> (!tt.ptr<i32>) {
    scf.yield %arg0 : !tt.ptr<i32>
  } else {
    scf.yield %arg1 : !tt.ptr<i32>
  }
  %value = tt.load %selected : !tt.ptr<i32>
  tt.store %arg2, %value : !tt.ptr<i32>
  tt.return
}

// CHECK-LABEL: tt.func private @helper(
// CHECK-SAME: %arg0: !tt.ptr<i32>)
tt.func private @helper(%arg0: !tt.ptr<i32>) {
  tt.return
}
