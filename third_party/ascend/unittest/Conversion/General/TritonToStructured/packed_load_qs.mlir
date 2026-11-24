// This regression test covers the W3-QS packed-weight case.
// The logical matrix is derived from row/column coordinates using the packed
// offset formula used by quantized weights.  The rewrite aims to prove that the
// load can be converted into a compact physical load and then reconstructed to
// the original logical tensor view.
// RUN: triton-opt '--triton-to-structured=enable-packed-load-rewrite=true' --split-input-file %s | FileCheck %s

module {
  tt.func public @w3_qs(%base: !tt.ptr<i8>) -> tensor<16x128xi8> {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %rs = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cs = tt.expand_dims %c {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %rb = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x128xi32>
    %cb = tt.broadcast %cs : tensor<1x128xi32> -> tensor<16x128xi32>
    %two = arith.constant 2 : i32
    %five = arith.constant 5 : i32
    %eight = arith.constant 8 : i32
    %thirtyone = arith.constant 31 : i32
    %thirtytwo = arith.constant 32 : i32
    %two_b = tt.splat %two : i32 -> tensor<16x128xi32>
    %five_b = tt.splat %five : i32 -> tensor<16x128xi32>
    %eight_b = tt.splat %eight : i32 -> tensor<16x128xi32>
    %thirtyone_b = tt.splat %thirtyone : i32 -> tensor<16x128xi32>
    %thirtytwo_b = tt.splat %thirtytwo : i32 -> tensor<16x128xi32>
    %group = arith.shrsi %cb, %five_b : tensor<16x128xi32>
    %in_group = arith.andi %cb, %thirtyone_b : tensor<16x128xi32>
    %in_group_term = arith.shrsi %in_group, %two_b : tensor<16x128xi32>
    %row_term = arith.muli %rb, %thirtytwo_b : tensor<16x128xi32>
    %group_term = arith.muli %group, %eight_b : tensor<16x128xi32>
    %offset0 = arith.addi %row_term, %group_term : tensor<16x128xi32>
    %offset = arith.addi %offset0, %in_group_term : tensor<16x128xi32>
    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<16x128x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %offset : tensor<16x128x!tt.ptr<i8>>, tensor<16x128xi32>
    %v = tt.load %ptr : tensor<16x128x!tt.ptr<i8>>
    tt.return %v : tensor<16x128xi8>
  }
}

// CHECK-LABEL: tt.func public @w3_qs
// CHECK: tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
// CHECK: tt.load
// CHECK: tt.reshape
// CHECK: tt.broadcast
// CHECK: tt.reshape

// -----

module {
  tt.func public @w3_qs_sub_k(%base: !tt.ptr<i8>) -> tensor<16x128xi8> {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %rs = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cs = tt.expand_dims %c {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %rb = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x128xi32>
    %cb = tt.broadcast %cs : tensor<1x128xi32> -> tensor<16x128xi32>
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %three = arith.constant 3 : i32
    %five = arith.constant 5 : i32
    %seven = arith.constant 7 : i32
    %eight = arith.constant 8 : i32
    %thirtyone = arith.constant 31 : i32
    %thirtytwo = arith.constant 32 : i32
    %one_b = tt.splat %one : i32 -> tensor<16x128xi32>
    %two_b = tt.splat %two : i32 -> tensor<16x128xi32>
    %three_b = tt.splat %three : i32 -> tensor<16x128xi32>
    %five_b = tt.splat %five : i32 -> tensor<16x128xi32>
    %seven_b = tt.splat %seven : i32 -> tensor<16x128xi32>
    %eight_b = tt.splat %eight : i32 -> tensor<16x128xi32>
    %thirtyone_b = tt.splat %thirtyone : i32 -> tensor<16x128xi32>
    %thirtytwo_b = tt.splat %thirtytwo : i32 -> tensor<16x128xi32>
    %group = arith.shrsi %cb, %five_b : tensor<16x128xi32>
    %in_group = arith.andi %cb, %thirtyone_b : tensor<16x128xi32>
    %sub = arith.shrsi %in_group, %three_b : tensor<16x128xi32>
    %k = arith.andi %in_group, %seven_b : tensor<16x128xi32>
    %sub_shift = arith.shli %sub, %one_b : tensor<16x128xi32>
    %k_shift = arith.shrsi %k, %two_b : tensor<16x128xi32>
    %in_group_term = arith.addi %sub_shift, %k_shift : tensor<16x128xi32>
    %row_term = arith.muli %rb, %thirtytwo_b : tensor<16x128xi32>
    %group_term = arith.muli %group, %eight_b : tensor<16x128xi32>
    %offset0 = arith.addi %row_term, %group_term : tensor<16x128xi32>
    %offset = arith.addi %offset0, %in_group_term : tensor<16x128xi32>
    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<16x128x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %offset : tensor<16x128x!tt.ptr<i8>>, tensor<16x128xi32>
    %v = tt.load %ptr : tensor<16x128x!tt.ptr<i8>>
    tt.return %v : tensor<16x128xi8>
  }
}

// CHECK-LABEL: tt.func public @w3_qs_sub_k
// CHECK: tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
// CHECK: tt.load
// CHECK: tt.reshape
// CHECK: tt.broadcast
// CHECK: tt.reshape

// -----

module {
  tt.func public @w3_qs_pair0(%base: !tt.ptr<i8>) -> tensor<16x128xi8> {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %rs = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cs = tt.expand_dims %c {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %rb = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x128xi32>
    %cb = tt.broadcast %cs : tensor<1x128xi32> -> tensor<16x128xi32>
    %two = arith.constant 2 : i32
    %three = arith.constant 3 : i32
    %five = arith.constant 5 : i32
    %eight = arith.constant 8 : i32
    %thirtyone = arith.constant 31 : i32
    %thirtytwo = arith.constant 32 : i32
    %two_b = tt.splat %two : i32 -> tensor<16x128xi32>
    %three_b = tt.splat %three : i32 -> tensor<16x128xi32>
    %five_b = tt.splat %five : i32 -> tensor<16x128xi32>
    %eight_b = tt.splat %eight : i32 -> tensor<16x128xi32>
    %thirtyone_b = tt.splat %thirtyone : i32 -> tensor<16x128xi32>
    %thirtytwo_b = tt.splat %thirtytwo : i32 -> tensor<16x128xi32>
    %group = arith.shrsi %cb, %five_b : tensor<16x128xi32>
    %in_group = arith.andi %cb, %thirtyone_b : tensor<16x128xi32>
    %sub = arith.shrsi %in_group, %three_b : tensor<16x128xi32>
    %sub_term = arith.muli %sub, %two_b : tensor<16x128xi32>
    %row_term = arith.muli %rb, %thirtytwo_b : tensor<16x128xi32>
    %group_term = arith.muli %group, %eight_b : tensor<16x128xi32>
    %offset0 = arith.addi %row_term, %group_term : tensor<16x128xi32>
    %offset = arith.addi %offset0, %sub_term : tensor<16x128xi32>
    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<16x128x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %offset : tensor<16x128x!tt.ptr<i8>>, tensor<16x128xi32>
    %v = tt.load %ptr : tensor<16x128x!tt.ptr<i8>>
    tt.return %v : tensor<16x128xi8>
  }
}

// CHECK-LABEL: tt.func public @w3_qs_pair0
// CHECK: tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK: tt.load {{.*}} : tensor<16x16x!tt.ptr<i8>>
// CHECK: tt.expand_dims {{.*}} {axis = 2 : i32} : tensor<16x16xi8> -> tensor<16x16x1xi8>
// CHECK: tt.broadcast {{.*}} : tensor<16x16x1xi8> -> tensor<16x16x8xi8>
// CHECK: tt.reshape {{.*}} : tensor<16x16x8xi8> -> tensor<16x128xi8>

// -----

module {
  tt.func public @w3_qs_pair1(%base: !tt.ptr<i8>) -> tensor<16x128xi8> {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %rs = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cs = tt.expand_dims %c {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %rb = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x128xi32>
    %cb = tt.broadcast %cs : tensor<1x128xi32> -> tensor<16x128xi32>
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %three = arith.constant 3 : i32
    %five = arith.constant 5 : i32
    %eight = arith.constant 8 : i32
    %thirtyone = arith.constant 31 : i32
    %thirtytwo = arith.constant 32 : i32
    %one_b = tt.splat %one : i32 -> tensor<16x128xi32>
    %two_b = tt.splat %two : i32 -> tensor<16x128xi32>
    %three_b = tt.splat %three : i32 -> tensor<16x128xi32>
    %five_b = tt.splat %five : i32 -> tensor<16x128xi32>
    %eight_b = tt.splat %eight : i32 -> tensor<16x128xi32>
    %thirtyone_b = tt.splat %thirtyone : i32 -> tensor<16x128xi32>
    %thirtytwo_b = tt.splat %thirtytwo : i32 -> tensor<16x128xi32>
    %group = arith.shrsi %cb, %five_b : tensor<16x128xi32>
    %in_group = arith.andi %cb, %thirtyone_b : tensor<16x128xi32>
    %sub = arith.shrsi %in_group, %three_b : tensor<16x128xi32>
    %sub_term = arith.muli %sub, %two_b : tensor<16x128xi32>
    %row_term = arith.muli %rb, %thirtytwo_b : tensor<16x128xi32>
    %group_term = arith.muli %group, %eight_b : tensor<16x128xi32>
    %offset0 = arith.addi %row_term, %group_term : tensor<16x128xi32>
    %offset1 = arith.addi %offset0, %sub_term : tensor<16x128xi32>
    %offset = arith.addi %offset1, %one_b : tensor<16x128xi32>
    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<16x128x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %offset : tensor<16x128x!tt.ptr<i8>>, tensor<16x128xi32>
    %v = tt.load %ptr : tensor<16x128x!tt.ptr<i8>>
    tt.return %v : tensor<16x128xi8>
  }
}

// CHECK-LABEL: tt.func public @w3_qs_pair1
// CHECK: arith.constant dense<1> : tensor<16x16xi32>
// CHECK: tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK: tt.load {{.*}} : tensor<16x16x!tt.ptr<i8>>
// CHECK: tt.expand_dims {{.*}} {axis = 2 : i32} : tensor<16x16xi8> -> tensor<16x16x1xi8>
// CHECK: tt.broadcast {{.*}} : tensor<16x16x1xi8> -> tensor<16x16x8xi8>
// CHECK: tt.reshape {{.*}} : tensor<16x16x8xi8> -> tensor<16x128xi8>
