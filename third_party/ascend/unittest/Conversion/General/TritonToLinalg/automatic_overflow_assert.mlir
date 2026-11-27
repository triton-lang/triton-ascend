// RUN: triton-opt %s --triton-to-linalg --verify-each | FileCheck %s

// Keep the widened arithmetic and bounds check emitted by the integer
// overflow sanitizer, and pass its result to the device assertion.
// CHECK: func.func private @[[$ASSERT:triton_assert[^ (]*]](i1) attributes {msg = "int32 overflow detected for operation add"}
// CHECK-LABEL: func.func @checked_add(
// CHECK-DAG: %[[MIN:.*]] = arith.constant -2147483648 : i64
// CHECK-DAG: %[[MAX:.*]] = arith.constant 2147483647 : i64
// CHECK: %[[X:.*]] = arith.extsi {{.*}} : i32 to i64
// CHECK: %[[Y:.*]] = arith.extsi {{.*}} : i32 to i64
// CHECK: %[[SUM:.*]] = arith.addi %[[X]], %[[Y]] : i64
// CHECK: %[[UPPER:.*]] = arith.cmpi sle, %[[SUM]], %[[MAX]] : i64
// CHECK: %[[LOWER:.*]] = arith.cmpi sge, %[[SUM]], %[[MIN]] : i64
// CHECK: %[[OK:.*]] = arith.andi %[[UPPER]], %[[LOWER]] : i1
// CHECK: call @[[$ASSERT]](%[[OK]]) : (i1) -> ()
// CHECK: %[[RESULT:.*]] = arith.addi {{.*}} : i32
// CHECK: %[[TENSOR:.*]] = tensor.insert %[[RESULT]] into {{.*}}
// CHECK: bufferization.materialize_in_destination %[[TENSOR]] in writable
// CHECK: return
tt.func @checked_add(%x_ptr: !tt.ptr<i32>, %y_ptr: !tt.ptr<i32>, %z_ptr: !tt.ptr<i32>) {
  %min = arith.constant -2147483648 : i64
  %max = arith.constant 2147483647 : i64
  %x = tt.load %x_ptr : !tt.ptr<i32>
  %y = tt.load %y_ptr : !tt.ptr<i32>
  %x_wide = arith.extsi %x : i32 to i64
  %y_wide = arith.extsi %y : i32 to i64
  %sum = arith.addi %x_wide, %y_wide : i64
  %upper = arith.cmpi sle, %sum, %max : i64
  %lower = arith.cmpi sge, %sum, %min : i64
  %ok = arith.andi %upper, %lower : i1
  tt.assert %ok, "int32 overflow detected for operation add" {tt.auto_overflow_assert} : i1
  %result = arith.addi %x, %y : i32
  tt.store %z_ptr, %result : !tt.ptr<i32>
  tt.return
}
