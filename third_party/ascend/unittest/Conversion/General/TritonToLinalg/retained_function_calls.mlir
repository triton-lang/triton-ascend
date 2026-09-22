// RUN: triton-opt %s --triton-to-linalg --verify-each | FileCheck %s

// CHECK-LABEL: func.func @caller(
// CHECK-SAME: %[[X:arg[0-9]+]]: i32, %[[NX:arg[0-9]+]]: i32, %[[NY:arg[0-9]+]]: i32, %[[NZ:arg[0-9]+]]: i32, %[[PX:arg[0-9]+]]: i32, %[[PY:arg[0-9]+]]: i32, %[[PZ:arg[0-9]+]]: i32)
// CHECK: %[[RESULT:.*]] = call @helper(%[[X]], %[[NX]], %[[NY]], %[[NZ]], %[[PX]], %[[PY]], %[[PZ]]) : (i32, i32, i32, i32, i32, i32, i32) -> i32
// CHECK: tensor.insert %[[RESULT]]
// CHECK: return
// CHECK-LABEL: func.func private @helper(
// CHECK-SAME: %[[HX:arg[0-9]+]]: i32, %{{arg[0-9]+}}: i32, %[[HNY:arg[0-9]+]]: i32, %{{arg[0-9]+}}: i32, %{{arg[0-9]+}}: i32, %{{arg[0-9]+}}: i32, %[[HPZ:arg[0-9]+]]: i32) -> i32
// CHECK-NOT: global_kernel
// CHECK-SAME: no_inline
// CHECK: %[[A:.*]] = arith.addi %[[HX]], %[[HNY]] : i32
// CHECK: %[[B:.*]] = arith.addi %[[A]], %[[HPZ]] : i32
// CHECK: return %[[B]] : i32
tt.func public @caller(%out: !tt.ptr<i32>, %x: i32) {
  %result = tt.call @helper(%x) : (i32) -> i32
  tt.store %out, %result : !tt.ptr<i32>
  tt.return
}

tt.func private @helper(%x: i32) -> i32 attributes {noinline = true} {
  %ny = tt.get_num_programs y : i32
  %pz = tt.get_program_id z : i32
  %a = arith.addi %x, %ny : i32
  %b = arith.addi %a, %pz : i32
  tt.return %b : i32
}
