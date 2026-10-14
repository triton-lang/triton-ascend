// RUN: triton-opt %s -graph-optimize='compile-on-910-95=true' --verify-each | FileCheck %s
// RUN: triton-opt %s -graph-optimize='compile-on-910-95=false' --verify-each | FileCheck %s --check-prefix=OTHER
// RUN: triton-opt %s -graph-optimize='compile-on-910-95=true' --verify-each | triton-opt -graph-optimize='compile-on-910-95=true' --verify-each | FileCheck %s

// The complemented i32 range remains one-hot. The same reduced value feeds
// comparison, low/high extraction, an XOR, and a full-width external user.
// CHECK-LABEL: tt.func @inverse_i32_shared(
// CHECK-COUNT-2: tt.reduce.return {{.*}} : i32
// CHECK: arith.xori {{.*}} : tensor<2xi32>
// CHECK: arith.xori {{.*}} : tensor<2xi32>
// CHECK: arith.cmpi sgt, {{.*}} : tensor<2xi32>
// CHECK: arith.cmpi eq, {{.*}} : tensor<2xi32>
// CHECK: arith.cmpi ugt, {{.*}} : tensor<2xi32>
// CHECK: tt.return {{.*}} : tensor<2xi64>, tensor<2xi64>, tensor<2xi1>, tensor<2xi32>, tensor<2xi32>
// OTHER-LABEL: tt.func @inverse_i32_shared(
// OTHER: arith.muli {{.*}} : tensor<2x2xi64>
// OTHER: tt.reduce.return {{.*}} : i64
// OTHER: arith.cmpi sgt, {{.*}} : tensor<2xi64>
tt.func @inverse_i32_shared(%data: tensor<2x2xi64>, %other: tensor<2xi64>) -> (tensor<2xi64>, tensor<2xi64>, tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) {
  %one = arith.constant dense<1> : tensor<2x2xi32>
  %shift = arith.constant dense<32> : tensor<2xi64>
  %range = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %row = tt.expand_dims %range {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %mask32 = tt.broadcast %row : tensor<1x2xi32> -> tensor<2x2xi32>
  %inverse = arith.subi %one, %mask32 : tensor<2x2xi32>
  %mask = arith.extsi %inverse : tensor<2x2xi32> to tensor<2x2xi64>
  %product = arith.muli %data, %mask : tensor<2x2xi64>
  %sum = "tt.reduce"(%product) <{axis = 1 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %s = arith.addi %a, %b : i64
    tt.reduce.return %s : i64
  }) : (tensor<2x2xi64>) -> tensor<2xi64>
  %xor = arith.xori %sum, %other : tensor<2xi64>
  %greater = arith.cmpi sgt, %sum, %other : tensor<2xi64>
  %low = arith.trunci %sum : tensor<2xi64> to tensor<2xi32>
  %upper = arith.shrui %sum, %shift : tensor<2xi64>
  %high = arith.trunci %upper : tensor<2xi64> to tensor<2xi32>
  tt.return %sum, %xor, %greater, %low, %high : tensor<2xi64>, tensor<2xi64>, tensor<2xi1>, tensor<2xi32>, tensor<2xi32>
}

// Complement after extension and reduction along axis zero are both valid.
// CHECK-LABEL: tt.func @inverse_i64_axis_zero(
// CHECK-COUNT-2: tt.reduce.return {{.*}} : i32
// CHECK-NOT: tt.reduce.return {{.*}} : i64
// CHECK: tt.return {{.*}} : tensor<2xi64>
// OTHER-LABEL: tt.func @inverse_i64_axis_zero(
// OTHER: tt.reduce.return {{.*}} : i64
tt.func @inverse_i64_axis_zero(%data: tensor<2x2xi64>) -> tensor<2xi64> {
  %one = arith.constant dense<1> : tensor<2x2xi64>
  %range = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %wide = arith.extui %range : tensor<2xi32> to tensor<2xi64>
  %column = tt.expand_dims %wide {axis = 1 : i32} : tensor<2xi64> -> tensor<2x1xi64>
  %mask = tt.broadcast %column : tensor<2x1xi64> -> tensor<2x2xi64>
  %inverse = arith.subi %one, %mask : tensor<2x2xi64>
  %product = arith.muli %inverse, %data : tensor<2x2xi64>
  %sum = "tt.reduce"(%product) <{axis = 0 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %s = arith.addi %b, %a : i64
    tt.reduce.return %s : i64
  }) : (tensor<2x2xi64>) -> tensor<2xi64>
  tt.return %sum : tensor<2xi64>
}

// Boolean elements may be [1,1], so their reduction can need a carry.
// CHECK-LABEL: tt.func @keep_unknown_boolean_mask(
// CHECK: arith.muli {{.*}} : tensor<2x2xi64>
// CHECK: tt.reduce.return {{.*}} : i64
// OTHER-LABEL: tt.func @keep_unknown_boolean_mask(
// OTHER: tt.reduce.return {{.*}} : i64
tt.func @keep_unknown_boolean_mask(%data: tensor<2x2xi64>, %predicate: tensor<2x2xi1>) -> tensor<2xi64> {
  %mask = arith.extui %predicate : tensor<2x2xi1> to tensor<2x2xi64>
  %product = arith.muli %data, %mask : tensor<2x2xi64>
  %sum = "tt.reduce"(%product) <{axis = 1 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %s = arith.addi %a, %b : i64
    tt.reduce.return %s : i64
  }) : (tensor<2x2xi64>) -> tensor<2xi64>
  tt.return %sum : tensor<2xi64>
}

// The broadcast-created axis duplicates each mask value. It is not one-hot.
// CHECK-LABEL: tt.func @keep_copied_reduction_axis(
// CHECK: arith.muli {{.*}} : tensor<2x2xi64>
// CHECK: tt.reduce.return {{.*}} : i64
// OTHER-LABEL: tt.func @keep_copied_reduction_axis(
// OTHER: tt.reduce.return {{.*}} : i64
tt.func @keep_copied_reduction_axis(%data: tensor<2x2xi64>) -> tensor<2xi64> {
  %range = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %wide = arith.extui %range : tensor<2xi32> to tensor<2xi64>
  %row = tt.expand_dims %wide {axis = 0 : i32} : tensor<2xi64> -> tensor<1x2xi64>
  %mask = tt.broadcast %row : tensor<1x2xi64> -> tensor<2x2xi64>
  %product = arith.muli %data, %mask : tensor<2x2xi64>
  %sum = "tt.reduce"(%product) <{axis = 0 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %s = arith.addi %a, %b : i64
    tt.reduce.return %s : i64
  }) : (tensor<2x2xi64>) -> tensor<2xi64>
  tt.return %sum : tensor<2xi64>
}

// Scalar reduction results stay at their original width.
// CHECK-LABEL: tt.func @keep_rank_one(
// CHECK: arith.muli {{.*}} : tensor<2xi64>
// CHECK: tt.reduce.return {{.*}} : i64
// CHECK: tt.return {{.*}} : i64
// OTHER-LABEL: tt.func @keep_rank_one(
// OTHER: tt.reduce.return {{.*}} : i64
tt.func @keep_rank_one(%data: tensor<2xi64>) -> i64 {
  %range = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %mask = arith.extui %range : tensor<2xi32> to tensor<2xi64>
  %product = arith.muli %data, %mask : tensor<2xi64>
  %sum = "tt.reduce"(%product) <{axis = 0 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %s = arith.addi %a, %b : i64
    tt.reduce.return %s : i64
  }) : (tensor<2xi64>) -> i64
  tt.return %sum : i64
}

// Dynamic outer dimensions do not enter the dense-constant split path.
// CHECK-LABEL: tt.func @keep_dynamic_shape(
// CHECK: arith.muli {{.*}} : tensor<?x2xi64>
// CHECK: tt.reduce.return {{.*}} : i64
// OTHER-LABEL: tt.func @keep_dynamic_shape(
// OTHER: tt.reduce.return {{.*}} : i64
tt.func @keep_dynamic_shape(%data: tensor<?x2xi64>) -> tensor<?xi64> {
  %range = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %wide = arith.extui %range : tensor<2xi32> to tensor<2xi64>
  %row = tt.expand_dims %wide {axis = 0 : i32} : tensor<2xi64> -> tensor<1x2xi64>
  %mask = tt.broadcast %row : tensor<1x2xi64> -> tensor<?x2xi64>
  %product = arith.muli %data, %mask : tensor<?x2xi64>
  %sum = "tt.reduce"(%product) <{axis = 1 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %s = arith.addi %a, %b : i64
    tt.reduce.return %s : i64
  }) : (tensor<?x2xi64>) -> tensor<?xi64>
  tt.return %sum : tensor<?xi64>
}

// A valid sum selects a component. Unsupported flags must still survive
// on a second multiplication, a second reduction, and a truncation consumer.
// CHECK-LABEL: tt.func @keep_flags_inside_split_function(
// CHECK: arith.muli {{.*}} overflow<nsw> : tensor<2x2xi64>
// CHECK: tt.reduce.return {{.*}} : i64
// CHECK-COUNT-2: tt.reduce.return {{.*}} : i32
// CHECK: arith.addi {{.*}} overflow<nuw> : i64
// CHECK: arith.trunci {{.*}} overflow<nsw> : tensor<2xi64> to tensor<2xi32>
// OTHER-LABEL: tt.func @keep_flags_inside_split_function(
// OTHER: arith.muli {{.*}} overflow<nsw> : tensor<2x2xi64>
// OTHER: arith.addi {{.*}} overflow<nuw> : i64
tt.func @keep_flags_inside_split_function(%data: tensor<2x2xi64>) -> (tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi32>) {
  %range = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %wide = arith.extui %range : tensor<2xi32> to tensor<2xi64>
  %row = tt.expand_dims %wide {axis = 0 : i32} : tensor<2xi64> -> tensor<1x2xi64>
  %mask = tt.broadcast %row : tensor<1x2xi64> -> tensor<2x2xi64>
  %flagged_product = arith.muli %data, %mask overflow<nsw> : tensor<2x2xi64>
  %flagged_mul_sum = "tt.reduce"(%flagged_product) <{axis = 1 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %s = arith.addi %a, %b : i64
    tt.reduce.return %s : i64
  }) : (tensor<2x2xi64>) -> tensor<2xi64>
  %product = arith.muli %data, %mask : tensor<2x2xi64>
  %sum = "tt.reduce"(%product) <{axis = 1 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %s = arith.addi %a, %b : i64
    tt.reduce.return %s : i64
  }) : (tensor<2x2xi64>) -> tensor<2xi64>
  %flagged_add_sum = "tt.reduce"(%product) <{axis = 1 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %s = arith.addi %a, %b overflow<nuw> : i64
    tt.reduce.return %s : i64
  }) : (tensor<2x2xi64>) -> tensor<2xi64>
  %truncated = arith.trunci %sum overflow<nsw> : tensor<2xi64> to tensor<2xi32>
  tt.return %flagged_mul_sum, %sum, %flagged_add_sum, %truncated : tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi32>
}

// Packed loop state remains i64. Its two uses share one split per iteration,
// and the split results are repacked only for the original loop yield.
// CHECK-LABEL: tt.func @loop_boundary(
// CHECK: scf.for {{.*}} iter_args(%[[STATE:[^ ]+]] = {{.*}}) -> (tensor<2x2xi64>) {
// CHECK: %[[LOW:.*]] = arith.trunci %[[STATE]] : tensor<2x2xi64> to tensor<2x2xi32>
// CHECK-NOT: arith.trunci %[[STATE]]
// CHECK: arith.muli %[[LOW]], {{.*}} : tensor<2x2xi32>
// CHECK-COUNT-2: tt.reduce.return {{.*}} : i32
// CHECK: arith.xori %[[LOW]], {{.*}} : tensor<2x2xi32>
// CHECK-NOT: arith.trunci %[[STATE]]
// CHECK: scf.yield {{.*}} : tensor<2x2xi64>
// OTHER-LABEL: tt.func @loop_boundary(
// OTHER: scf.for
// OTHER: arith.muli {{.*}} : tensor<2x2xi64>
// OTHER: tt.reduce.return {{.*}} : i64
tt.func @loop_boundary(%data: tensor<2x2xi64>, %end: index) -> tensor<2x2xi64> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %range = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %wide = arith.extui %range : tensor<2xi32> to tensor<2xi64>
  %row = tt.expand_dims %wide {axis = 0 : i32} : tensor<2xi64> -> tensor<1x2xi64>
  %mask = tt.broadcast %row : tensor<1x2xi64> -> tensor<2x2xi64>
  %result = scf.for %i = %zero to %end step %one iter_args(%state = %data) -> (tensor<2x2xi64>) {
    %product = arith.muli %state, %mask : tensor<2x2xi64>
    %sum = "tt.reduce"(%product) <{axis = 1 : i32}> ({
    ^bb0(%a: i64, %b: i64):
      %s = arith.addi %a, %b : i64
      tt.reduce.return %s : i64
    }) : (tensor<2x2xi64>) -> tensor<2xi64>
    %column = tt.expand_dims %sum {axis = 1 : i32} : tensor<2xi64> -> tensor<2x1xi64>
    %repeated = tt.broadcast %column : tensor<2x1xi64> -> tensor<2x2xi64>
    %next = arith.xori %state, %repeated : tensor<2x2xi64>
    scf.yield %next : tensor<2x2xi64>
  }
  tt.return %result : tensor<2x2xi64>
}
