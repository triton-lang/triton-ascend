// RUN: triton-opt %s -graph-optimize='compile-on-910-95=true' --verify-each | FileCheck %s
// RUN: triton-opt %s -graph-optimize='compile-on-910-95=false' --verify-each | FileCheck %s --check-prefix=OTHER
// RUN: triton-opt %s -graph-optimize='compile-on-910-95=true' --verify-each | triton-opt -graph-optimize='compile-on-910-95=true' --verify-each | FileCheck %s

// A shared constant, input, mask, or condition cannot connect separate data
// computations. Only the sum and its active XOR/select descendants are split.
// CHECK-LABEL: tt.func @unrelated_values(
// CHECK-COUNT-2: tt.reduce.return {{.*}} : i32
// CHECK: arith.xori {{.*}} : tensor<2xi32>
// CHECK: arith.xori {{.*}} : tensor<2xi32>
// CHECK: %[[UNRELATED_XOR:.*]] = arith.xori {{.*}} : tensor<2xi64>
// CHECK: %[[UNRELATED_CMP:.*]] = arith.cmpi sgt, {{.*}} : tensor<2xi64>
// CHECK: %[[UNRELATED_LOW:.*]] = arith.trunci {{.*}} : tensor<2xi64> to tensor<2xi32>
// CHECK: %[[UNRELATED_MUL:.*]] = arith.muli {{.*}} : tensor<2x2xi64>
// CHECK: %[[UNRELATED_SELECT:.*]] = arith.select {{.*}} : tensor<2xi1>, tensor<2xi64>
// CHECK: %[[SHARED_INPUT_XOR:.*]] = arith.xori {{.*}} : tensor<2x2xi64>
// CHECK: tt.return {{.*}}, %[[UNRELATED_XOR]], %[[UNRELATED_CMP]], %[[UNRELATED_LOW]], %[[UNRELATED_MUL]], %[[UNRELATED_SELECT]], %[[SHARED_INPUT_XOR]]
// OTHER-LABEL: tt.func @unrelated_values(
// OTHER: tt.reduce.return {{.*}} : i64
tt.func @unrelated_values(%data: tensor<2x2xi64>, %unrelated: tensor<2xi64>, %other: tensor<2xi64>, %spare: tensor<2x2xi64>, %condition: tensor<2xi1>) -> (tensor<2xi64>, tensor<2xi64>, tensor<2xi1>, tensor<2xi32>, tensor<2x2xi64>, tensor<2xi64>, tensor<2x2xi64>) {
  %zero = arith.constant dense<0> : tensor<2xi64>
  %shared = arith.constant dense<4294967299> : tensor<2xi64>
  %range = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %wide = arith.extui %range : tensor<2xi32> to tensor<2xi64>
  %row = tt.expand_dims %wide {axis = 0 : i32} : tensor<2xi64> -> tensor<1x2xi64>
  %mask = tt.broadcast %row : tensor<1x2xi64> -> tensor<2x2xi64>
  %product = arith.muli %data, %mask : tensor<2x2xi64>
  %sum = "tt.reduce"(%product) <{axis = 1 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %s = arith.addi %a, %b : i64
    tt.reduce.return %s : i64
  }) : (tensor<2x2xi64>) -> tensor<2xi64>
  %active_xor = arith.xori %sum, %shared : tensor<2xi64>
  %active = arith.select %condition, %active_xor, %zero : tensor<2xi1>, tensor<2xi64>
  %connected_cmp = arith.cmpi sgt, %sum, %other : tensor<2xi64>
  %unrelated_xor = arith.xori %unrelated, %shared : tensor<2xi64>
  %unrelated_cmp = arith.cmpi sgt, %unrelated, %other : tensor<2xi64>
  %unrelated_low = arith.trunci %unrelated : tensor<2xi64> to tensor<2xi32>
  %unrelated_mul = arith.muli %spare, %mask : tensor<2x2xi64>
  %unrelated_select = arith.select %connected_cmp, %unrelated, %zero : tensor<2xi1>, tensor<2xi64>
  %shared_input_xor = arith.xori %data, %spare : tensor<2x2xi64>
  tt.return %active, %unrelated_xor, %unrelated_cmp, %unrelated_low, %unrelated_mul, %unrelated_select, %shared_input_xor : tensor<2xi64>, tensor<2xi64>, tensor<2xi1>, tensor<2xi32>, tensor<2x2xi64>, tensor<2xi64>, tensor<2x2xi64>
}

// Unsupported multiplication, a reorderable reshape, and an unsupported shift
// stop the component. Supported arithmetic beyond each boundary stays i64.
// CHECK-LABEL: tt.func @rejected_branches(
// CHECK-COUNT-2: tt.reduce.return {{.*}} : i32
// CHECK: %[[UNKNOWN_PRODUCT:.*]] = arith.muli {{.*}} : tensor<2xi64>
// CHECK: %[[TAIL:.*]] = arith.xori %[[UNKNOWN_PRODUCT]], {{.*}} : tensor<2xi64>
// CHECK: arith.cmpi sgt, %[[TAIL]], {{.*}} : tensor<2xi64>
// CHECK: arith.trunci %[[TAIL]] : tensor<2xi64> to tensor<2xi32>
// CHECK: %[[REORDER:.*]] = tt.reshape {{.*}} allow_reorder : tensor<2x1xi64> -> tensor<2xi64>
// CHECK: arith.xori %[[REORDER]], {{.*}} : tensor<2xi64>
// CHECK: %[[SHIFT:.*]] = arith.shrui {{.*}} : tensor<2xi64>
// CHECK: arith.xori %[[SHIFT]], {{.*}} : tensor<2xi64>
// OTHER-LABEL: tt.func @rejected_branches(
// OTHER: tt.reduce.return {{.*}} : i64
tt.func @rejected_branches(%data: tensor<2x2xi64>, %unknown: tensor<2xi64>, %other: tensor<2xi64>) -> (tensor<2xi64>, tensor<2xi1>, tensor<2xi32>, tensor<2xi64>, tensor<2xi64>) {
  %seventeen = arith.constant dense<17> : tensor<2xi64>
  %range = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %wide = arith.extui %range : tensor<2xi32> to tensor<2xi64>
  %row = tt.expand_dims %wide {axis = 0 : i32} : tensor<2xi64> -> tensor<1x2xi64>
  %mask = tt.broadcast %row : tensor<1x2xi64> -> tensor<2x2xi64>
  %product = arith.muli %data, %mask : tensor<2x2xi64>
  %sum = "tt.reduce"(%product) <{axis = 1 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %s = arith.addi %a, %b : i64
    tt.reduce.return %s : i64
  }) : (tensor<2x2xi64>) -> tensor<2xi64>
  %unknown_product = arith.muli %sum, %unknown : tensor<2xi64>
  %tail = arith.xori %unknown_product, %other : tensor<2xi64>
  %tail_cmp = arith.cmpi sgt, %tail, %other : tensor<2xi64>
  %tail_low = arith.trunci %tail : tensor<2xi64> to tensor<2xi32>
  %expanded = tt.expand_dims %sum {axis = 1 : i32} : tensor<2xi64> -> tensor<2x1xi64>
  %reorder = tt.reshape %expanded allow_reorder : tensor<2x1xi64> -> tensor<2xi64>
  %reorder_xor = arith.xori %reorder, %other : tensor<2xi64>
  %shift = arith.shrui %sum, %seventeen : tensor<2xi64>
  %shift_xor = arith.xori %shift, %other : tensor<2xi64>
  tt.return %tail, %tail_cmp, %tail_low, %reorder_xor, %shift_xor : tensor<2xi64>, tensor<2xi1>, tensor<2xi32>, tensor<2xi64>, tensor<2xi64>
}

// Values shared across sibling regions are split at their original definition.
// Each selected branch repacks for scf.yield; the scf.if result is a boundary.
// CHECK-LABEL: tt.func @sibling_regions(
// CHECK-COUNT-2: tt.reduce.return {{.*}} : i32
// CHECK: scf.if {{.*}} -> (tensor<2xi64>) {
// CHECK-COUNT-2: arith.xori {{.*}} : tensor<2xi32>
// CHECK: %[[THEN_PACK:.*]] = arith.ori {{.*}} : tensor<2xi64>
// CHECK: scf.yield %[[THEN_PACK]] : tensor<2xi64>
// CHECK: } else {
// CHECK-COUNT-2: arith.xori {{.*}} : tensor<2xi32>
// CHECK: %[[ELSE_PACK:.*]] = arith.ori {{.*}} : tensor<2xi64>
// CHECK: scf.yield %[[ELSE_PACK]] : tensor<2xi64>
// CHECK: %[[AFTER:.*]] = arith.xori {{.*}} : tensor<2xi64>
// CHECK: tt.return %[[AFTER]]
// OTHER-LABEL: tt.func @sibling_regions(
// OTHER: tt.reduce.return {{.*}} : i64
tt.func @sibling_regions(%data: tensor<2x2xi64>, %other: tensor<2xi64>, %condition: i1) -> tensor<2xi64> {
  %range = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %wide = arith.extui %range : tensor<2xi32> to tensor<2xi64>
  %row = tt.expand_dims %wide {axis = 0 : i32} : tensor<2xi64> -> tensor<1x2xi64>
  %mask = tt.broadcast %row : tensor<1x2xi64> -> tensor<2x2xi64>
  %product = arith.muli %data, %mask : tensor<2x2xi64>
  %sum = "tt.reduce"(%product) <{axis = 1 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %s = arith.addi %a, %b : i64
    tt.reduce.return %s : i64
  }) : (tensor<2x2xi64>) -> tensor<2xi64>
  %branch = scf.if %condition -> (tensor<2xi64>) {
    %value = arith.xori %sum, %other : tensor<2xi64>
    scf.yield %value : tensor<2xi64>
  } else {
    %value = arith.xori %other, %sum : tensor<2xi64>
    scf.yield %value : tensor<2xi64>
  }
  %after = arith.xori %branch, %other : tensor<2xi64>
  tt.return %after : tensor<2xi64>
}

// A shared iter_arg is a boundary and cannot link the independent accumulation
// to the selected sum. The existing two-i64 loop interface remains unchanged.
// CHECK-LABEL: tt.func @loop_scope(
// CHECK: scf.for {{.*}} iter_args(%[[STATE:[^ ]+]] = {{.*}}, %[[ACC:[^ ]+]] = {{.*}}) -> (tensor<2x2xi64>, tensor<2x2xi64>) {
// CHECK: %[[LOW:.*]] = arith.trunci %[[STATE]] : tensor<2x2xi64> to tensor<2x2xi32>
// CHECK-NOT: arith.trunci %[[STATE]]
// CHECK: arith.muli %[[LOW]], {{.*}} : tensor<2x2xi32>
// CHECK-COUNT-2: tt.reduce.return {{.*}} : i32
// CHECK: arith.xori %[[LOW]], {{.*}} : tensor<2x2xi32>
// CHECK: %[[INDEPENDENT:.*]] = arith.xori %[[STATE]], %[[ACC]] : tensor<2x2xi64>
// CHECK: scf.yield {{.*}}, %[[INDEPENDENT]] : tensor<2x2xi64>, tensor<2x2xi64>
// CHECK: %[[AFTER:.*]] = arith.xori {{.*}} : tensor<2x2xi64>
// CHECK: tt.return %[[AFTER]]
// OTHER-LABEL: tt.func @loop_scope(
// OTHER: tt.reduce.return {{.*}} : i64
tt.func @loop_scope(%data: tensor<2x2xi64>, %other: tensor<2x2xi64>, %end: index) -> tensor<2x2xi64> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %range = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %wide = arith.extui %range : tensor<2xi32> to tensor<2xi64>
  %row = tt.expand_dims %wide {axis = 0 : i32} : tensor<2xi64> -> tensor<1x2xi64>
  %mask = tt.broadcast %row : tensor<1x2xi64> -> tensor<2x2xi64>
  %result:2 = scf.for %i = %zero to %end step %one iter_args(%state = %data, %acc = %other) -> (tensor<2x2xi64>, tensor<2x2xi64>) {
    %product = arith.muli %state, %mask : tensor<2x2xi64>
    %sum = "tt.reduce"(%product) <{axis = 1 : i32}> ({
    ^bb0(%a: i64, %b: i64):
      %s = arith.addi %a, %b : i64
      tt.reduce.return %s : i64
    }) : (tensor<2x2xi64>) -> tensor<2xi64>
    %column = tt.expand_dims %sum {axis = 1 : i32} : tensor<2xi64> -> tensor<2x1xi64>
    %repeated = tt.broadcast %column : tensor<2x1xi64> -> tensor<2x2xi64>
    %next = arith.xori %state, %repeated : tensor<2x2xi64>
    %independent = arith.xori %state, %acc : tensor<2x2xi64>
    scf.yield %next, %independent : tensor<2x2xi64>, tensor<2x2xi64>
  }
  %after = arith.xori %result#0, %result#1 : tensor<2x2xi64>
  tt.return %after : tensor<2x2xi64>
}
