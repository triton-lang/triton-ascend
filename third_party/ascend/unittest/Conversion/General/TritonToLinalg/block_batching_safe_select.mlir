// RUN: triton-opt --triton-to-linalg='global-kernel=false enable-select-analysis=true' %s -verify-each | FileCheck %s
// A safety select must remain before the protected operation, without a
// slice/insert copy. Ordinary structured selects retain existing lowering.
// CHECK-LABEL: func.func @safe_input
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: tensor.insert_slice
// CHECK: %[[SAFE:.*]] = arith.select {{.*}}hacc.block_batching_safe_input
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: tensor.insert_slice
// CHECK: arith.divsi {{.*}}, %[[SAFE]]
// CHECK: return
module {
  tt.func public @safe_input(%X: !tt.ptr<i32>, %Y: !tt.ptr<i32>, %n: i32) {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %xp = tt.splat %X : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %xa = tt.addptr %xp, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %x = tt.load %xa : tensor<8x!tt.ptr<i32>>
    %yp = tt.splat %Y : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ya = tt.addptr %yp, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %limit = tt.splat %n : i32 -> tensor<8xi32>
    %mask = arith.cmpi slt, %range, %limit : tensor<8xi32>
    %one = arith.constant dense<1> : tensor<8xi32>
    %safe = arith.select %mask, %x, %one {hacc.block_batching_safe_input} : tensor<8xi1>, tensor<8xi32>
    %y = arith.divsi %one, %safe : tensor<8xi32>
    tt.store %ya, %y : tensor<8x!tt.ptr<i32>>
    tt.return
  }
  // CHECK-LABEL: func.func @ordinary_select
  // CHECK: tensor.extract_slice
  // CHECK: tensor.insert_slice
  // CHECK: return
  tt.func public @ordinary_select(%X: !tt.ptr<i32>, %Y: !tt.ptr<i32>, %n: i32) {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %xp = tt.splat %X : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %xa = tt.addptr %xp, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %x = tt.load %xa : tensor<8x!tt.ptr<i32>>
    %yp = tt.splat %Y : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ya = tt.addptr %yp, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %limit = tt.splat %n : i32 -> tensor<8xi32>
    %mask = arith.cmpi slt, %range, %limit : tensor<8xi32>
    %one = arith.constant dense<1> : tensor<8xi32>
    %selected = arith.select %mask, %x, %one : tensor<8xi1>, tensor<8xi32>
    tt.store %ya, %selected : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}
