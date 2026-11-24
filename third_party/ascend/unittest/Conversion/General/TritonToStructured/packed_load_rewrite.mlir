// These tests describe the packed-memory layouts used by qweights in the
// Ascend backend.  The intent is to keep the IR shape/offset logic visible to
// future readers: each function constructs a logical 2-D matrix, applies the
// storage formula for the compact layout, and then verifies the rewrite returns
// to the original logical tensor form.
// RUN: triton-opt '--triton-to-structured=enable-packed-load-rewrite=true' --split-input-file %s | FileCheck %s

module {
  tt.func public @w3_qh(%base: !tt.ptr<i8>) -> tensor<16x128xi8> {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %rs = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cs = tt.expand_dims %c {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %rb = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x128xi32>
    %cb = tt.broadcast %cs : tensor<1x128xi32> -> tensor<16x128xi32>
    %five = arith.constant 5 : i32
    %thirtyone = arith.constant 31 : i32
    %four = arith.constant 4 : i32
    %three = arith.constant 3 : i32
    %sixteen = arith.constant 16 : i32
    %five_b = tt.splat %five : i32 -> tensor<16x128xi32>
    %thirtyone_b = tt.splat %thirtyone : i32 -> tensor<16x128xi32>
    %four_b = tt.splat %four : i32 -> tensor<16x128xi32>
    %three_b = tt.splat %three : i32 -> tensor<16x128xi32>
    %sixteen_b = tt.splat %sixteen : i32 -> tensor<16x128xi32>
    %group = arith.shrsi %cb, %five_b : tensor<16x128xi32>
    %in_group = arith.andi %cb, %thirtyone_b : tensor<16x128xi32>
    %sub = arith.shrsi %in_group, %three_b : tensor<16x128xi32>
    %row_term = arith.muli %rb, %sixteen_b : tensor<16x128xi32>
    %group_term = arith.muli %group, %four_b : tensor<16x128xi32>
    %offset0 = arith.addi %row_term, %group_term : tensor<16x128xi32>
    %offset = arith.addi %offset0, %sub : tensor<16x128xi32>
    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<16x128x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %offset : tensor<16x128x!tt.ptr<i8>>, tensor<16x128xi32>
    %v = tt.load %ptr : tensor<16x128x!tt.ptr<i8>>
    tt.return %v : tensor<16x128xi8>
  }
}

// W3-QH: the compact buffer is smaller than the logical matrix, so the pass
// must materialize a range that matches the compact storage and then reshape it
// back into the logical matrix view.
// CHECK-LABEL: tt.func public @w3_qh
// CHECK: tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
// CHECK: tt.load
// CHECK: tt.reshape
// CHECK: tt.broadcast

// -----

module {
  tt.func public @w3_qh_k1024(%base: !tt.ptr<i8>) -> tensor<16x1024xi8> {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
    %rs = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cs = tt.expand_dims %c {axis = 0 : i32} : tensor<1024xi32> -> tensor<1x1024xi32>
    %rb = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x1024xi32>
    %cb = tt.broadcast %cs : tensor<1x1024xi32> -> tensor<16x1024xi32>
    %five = arith.constant 5 : i32
    %thirtyone = arith.constant 31 : i32
    %four = arith.constant 4 : i32
    %three = arith.constant 3 : i32
    %onehundredtwentyeight = arith.constant 128 : i32
    %five_b = tt.splat %five : i32 -> tensor<16x1024xi32>
    %thirtyone_b = tt.splat %thirtyone : i32 -> tensor<16x1024xi32>
    %four_b = tt.splat %four : i32 -> tensor<16x1024xi32>
    %three_b = tt.splat %three : i32 -> tensor<16x1024xi32>
    %row_b = tt.splat %onehundredtwentyeight : i32 -> tensor<16x1024xi32>
    %group = arith.shrsi %cb, %five_b : tensor<16x1024xi32>
    %in_group = arith.andi %cb, %thirtyone_b : tensor<16x1024xi32>
    %sub = arith.shrsi %in_group, %three_b : tensor<16x1024xi32>
    %row_term = arith.muli %rb, %row_b : tensor<16x1024xi32>
    %group_term = arith.muli %group, %four_b : tensor<16x1024xi32>
    %offset0 = arith.addi %row_term, %group_term : tensor<16x1024xi32>
    %offset = arith.addi %offset0, %sub : tensor<16x1024xi32>
    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<16x1024x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %offset : tensor<16x1024x!tt.ptr<i8>>, tensor<16x1024xi32>
    %v = tt.load %ptr : tensor<16x1024x!tt.ptr<i8>>
    tt.return %v : tensor<16x1024xi8>
  }
}

// W3-QH with a larger width keeps the same rewrite pattern but changes the
// compact physical size.  The important part is that the pass still recognizes
// the same offset formula for the larger packed matrix.
// CHECK-LABEL: tt.func public @w3_qh_k1024
// CHECK: tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32>

// -----

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

// W3-QS: 8 bytes per 32-element group, broadcast across 4 consecutive elements.
// CHECK-LABEL: tt.func public @w3_qs
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

// -----

module {
  tt.func public @w4(%base: !tt.ptr<i8>) -> tensor<16x128xi8> {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %rs = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cs = tt.expand_dims %c {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %rb = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x128xi32>
    %cb = tt.broadcast %cs : tensor<1x128xi32> -> tensor<16x128xi32>
    %five = arith.constant 5 : i32
    %fifteen = arith.constant 15 : i32
    %sixteen = arith.constant 16 : i32
    %sixtyfour = arith.constant 64 : i32
    %five_b = tt.splat %five : i32 -> tensor<16x128xi32>
    %fifteen_b = tt.splat %fifteen : i32 -> tensor<16x128xi32>
    %sixteen_b = tt.splat %sixteen : i32 -> tensor<16x128xi32>
    %sixtyfour_b = tt.splat %sixtyfour : i32 -> tensor<16x128xi32>
    %group = arith.shrsi %cb, %five_b : tensor<16x128xi32>
    %in_group = arith.andi %cb, %fifteen_b : tensor<16x128xi32>
    %row_term = arith.muli %rb, %sixtyfour_b : tensor<16x128xi32>
    %group_term = arith.muli %group, %sixteen_b : tensor<16x128xi32>
    %offset0 = arith.addi %row_term, %group_term : tensor<16x128xi32>
    %offset = arith.addi %offset0, %in_group : tensor<16x128xi32>
    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<16x128x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %offset : tensor<16x128x!tt.ptr<i8>>, tensor<16x128xi32>
    %v = tt.load %ptr : tensor<16x128x!tt.ptr<i8>>
    tt.return %v : tensor<16x128xi8>
  }
}

// W4 packs the logical matrix by a different formula; this branch confirms the
// rewrite still recognizes the alternative mapping and emits the compact load
// required for the physical buffer.
// CHECK-LABEL: tt.func public @w4
// CHECK: tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
// CHECK: tt.load

// -----

module {
  tt.func public @w4_g64_strided(%base: !tt.ptr<i8>) -> tensor<16x128xi8> {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %rs = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cs = tt.expand_dims %c {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %rb = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x128xi32>
    %cb = tt.broadcast %cs : tensor<1x128xi32> -> tensor<16x128xi32>
    %six = arith.constant 6 : i32
    %thirtyone = arith.constant 31 : i32
    %thirtytwo = arith.constant 32 : i32
    %sixtyfour = arith.constant 64 : i32
    %six_b = tt.splat %six : i32 -> tensor<16x128xi32>
    %thirtyone_b = tt.splat %thirtyone : i32 -> tensor<16x128xi32>
    %thirtytwo_b = tt.splat %thirtytwo : i32 -> tensor<16x128xi32>
    %sixtyfour_b = tt.splat %sixtyfour : i32 -> tensor<16x128xi32>
    %group = arith.shrsi %cb, %six_b : tensor<16x128xi32>
    %in_group = arith.andi %cb, %thirtyone_b : tensor<16x128xi32>
    %row_term = arith.muli %rb, %sixtyfour_b : tensor<16x128xi32>
    %group_term = arith.muli %group, %thirtytwo_b : tensor<16x128xi32>
    %offset0 = arith.addi %row_term, %group_term : tensor<16x128xi32>
    %offset = arith.addi %offset0, %in_group : tensor<16x128xi32>
    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<16x128x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %offset : tensor<16x128x!tt.ptr<i8>>, tensor<16x128xi32>
    %v = tt.load %ptr : tensor<16x128x!tt.ptr<i8>>
    tt.return %v : tensor<16x128xi8>
  }
}

// Generalized packed load test: group size G=64 with 32 physical bytes per group (W4 strided).
// Logical shape 16x128 -> 2 groups per row, 64 physical bytes per row -> 1024 bytes total.
// CHECK-LABEL: tt.func public @w4_g64_strided
// CHECK: tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
// CHECK: tt.load
// CHECK: tt.reshape
// CHECK: tt.broadcast
// CHECK: tt.reshape

// -----

module {
  tt.func public @irregular(%base: !tt.ptr<i8>) -> tensor<32xi8> {
    %offset = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<32x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %offset : tensor<32x!tt.ptr<i8>>, tensor<32xi32>
    %v = tt.load %ptr : tensor<32x!tt.ptr<i8>>
    tt.return %v : tensor<32xi8>
  }
}

// The last negative case verifies that non-packed loads are intentionally left
// alone.
// CHECK-LABEL: tt.func public @irregular
// CHECK: tt.load
// CHECK-NOT: tt.make_range {end = 256 : i32, start = 0 : i32}

// -----

module {
  tt.func public @multi_scalar_stride_and_dynamic_k(
      %base: !tt.ptr<i8>, %s0: i32, %s1: i32, %k_idx: i32) -> tensor<16x128xi8> {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %rs = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cs = tt.expand_dims %c {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %rb = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x128xi32>
    %cb = tt.broadcast %cs : tensor<1x128xi32> -> tensor<16x128xi32>

    %c4 = arith.constant 4 : i32
    %c128 = arith.constant 128 : i32
    %c2_b = arith.constant dense<2> : tensor<16x128xi32>

    %s1_4 = arith.muli %s1, %c4 : i32
    %row_stride = arith.addi %s0, %s1_4 : i32
    %row_stride_b = tt.splat %row_stride : i32 -> tensor<16x128xi32>
    %row_term = arith.muli %rb, %row_stride_b : tensor<16x128xi32>

    %k_base = arith.muli %k_idx, %c128 : i32
    %k_base_b = tt.splat %k_base : i32 -> tensor<16x128xi32>
    %k_offs = arith.addi %k_base_b, %cb : tensor<16x128xi32>
    %col_term = arith.shrsi %k_offs, %c2_b : tensor<16x128xi32>

    %offset = arith.addi %row_term, %col_term : tensor<16x128xi32>
    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<16x128x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %offset : tensor<16x128x!tt.ptr<i8>>, tensor<16x128xi32>
    %v = tt.load %ptr {cache = 2 : i32, evict = 1 : i32, isVolatile = true} : tensor<16x128x!tt.ptr<i8>>
    tt.return %v : tensor<16x128xi8>
  }
}

// Symbolic multi-scalar outer stride + dynamic aligned K-tile offset + volatile/cache preservation.
// CHECK-LABEL: tt.func public @multi_scalar_stride_and_dynamic_k
// CHECK: tt.load {{.*}} cacheModifier = ca {isVolatile = true} : tensor<16x32x!tt.ptr<i8>>
// CHECK: tt.expand_dims {{.*}} {axis = 2 : i32} : tensor<16x32xi8> -> tensor<16x32x1xi8>
// CHECK: tt.broadcast {{.*}} : tensor<16x32x1xi8> -> tensor<16x32x4xi8>
// CHECK: tt.reshape {{.*}} : tensor<16x32x4xi8> -> tensor<16x128xi8>

// -----

module {
  tt.func public @outer_runtime_mask_and_other(
      %base: !tt.ptr<i8>, %valid_rows: i32) -> tensor<16x128xi8> {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %rs = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cs = tt.expand_dims %c {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %rb = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x128xi32>
    %cb = tt.broadcast %cs : tensor<1x128xi32> -> tensor<16x128xi32>

    %vr_s = tt.splat %valid_rows : i32 -> tensor<16x1xi32>
    %rmask_2d = arith.cmpi slt, %rs, %vr_s : tensor<16x1xi32>
    %mask = tt.broadcast %rmask_2d : tensor<16x1xi1> -> tensor<16x128xi1>
    %other = arith.constant dense<0> : tensor<16x128xi8>

    %c32 = arith.constant dense<32> : tensor<16x128xi32>
    %c2 = arith.constant dense<2> : tensor<16x128xi32>
    %row_term = arith.muli %rb, %c32 : tensor<16x128xi32>
    %col_term = arith.shrsi %cb, %c2 : tensor<16x128xi32>
    %offset = arith.addi %row_term, %col_term : tensor<16x128xi32>

    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<16x128x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %offset : tensor<16x128x!tt.ptr<i8>>, tensor<16x128xi32>
    %v = tt.load %ptr, %mask, %other : tensor<16x128x!tt.ptr<i8>>
    tt.return %v : tensor<16x128xi8>
  }
}

// Outer runtime mask + scalar-splat other projected onto compact 16x32 load.
// CHECK-LABEL: tt.func public @outer_runtime_mask_and_other
// CHECK: tt.broadcast {{.*}} : tensor<16x1xi1> -> tensor<16x32xi1>
// CHECK: tt.load {{.*}}, {{.*}}, {{.*}} : tensor<16x32x!tt.ptr<i8>>
// CHECK: tt.expand_dims {{.*}} {axis = 2 : i32} : tensor<16x32xi8> -> tensor<16x32x1xi8>
// CHECK: tt.broadcast {{.*}} : tensor<16x32x1xi8> -> tensor<16x32x4xi8>
// CHECK: tt.reshape {{.*}} : tensor<16x32x4xi8> -> tensor<16x128xi8>

// -----

module {
  tt.func public @packed_3d_tensor(%base: !tt.ptr<i8>) -> tensor<4x16x128xi8> {
    %b = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>

    %b1 = tt.expand_dims %b {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %b2 = tt.expand_dims %b1 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
    %bb = tt.broadcast %b2 : tensor<4x1x1xi32> -> tensor<4x16x128xi32>

    %r0 = tt.expand_dims %r {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %r2 = tt.expand_dims %r0 {axis = 2 : i32} : tensor<1x16xi32> -> tensor<1x16x1xi32>
    %rb = tt.broadcast %r2 : tensor<1x16x1xi32> -> tensor<4x16x128xi32>

    %c0 = tt.expand_dims %c {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %c1 = tt.expand_dims %c0 {axis = 1 : i32} : tensor<1x128xi32> -> tensor<1x1x128xi32>
    %cb = tt.broadcast %c1 : tensor<1x1x128xi32> -> tensor<4x16x128xi32>

    %c512 = arith.constant dense<512> : tensor<4x16x128xi32>
    %c32 = arith.constant dense<32> : tensor<4x16x128xi32>
    %c2 = arith.constant dense<2> : tensor<4x16x128xi32>

    %b_term = arith.muli %bb, %c512 : tensor<4x16x128xi32>
    %r_term = arith.muli %rb, %c32 : tensor<4x16x128xi32>
    %c_term = arith.shrsi %cb, %c2 : tensor<4x16x128xi32>
    %off0 = arith.addi %b_term, %r_term : tensor<4x16x128xi32>
    %off = arith.addi %off0, %c_term : tensor<4x16x128xi32>

    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<4x16x128x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %off : tensor<4x16x128x!tt.ptr<i8>>, tensor<4x16x128xi32>
    %v = tt.load %ptr : tensor<4x16x128x!tt.ptr<i8>>
    tt.return %v : tensor<4x16x128xi8>
  }
}

// 3D contiguous packed load collapses to 1D 2048-byte physical load + 4D broadcast + 3D reshape.
// CHECK-LABEL: tt.func public @packed_3d_tensor
// CHECK: tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32>
// CHECK: tt.load {{.*}} : tensor<2048x!tt.ptr<i8>>
// CHECK: tt.reshape {{.*}} : tensor<2048xi8> -> tensor<4x16x32xi8>
// CHECK: tt.expand_dims {{.*}} {axis = 3 : i32} : tensor<4x16x32xi8> -> tensor<4x16x32x1xi8>
// CHECK: tt.broadcast {{.*}} : tensor<4x16x32x1xi8> -> tensor<4x16x32x4xi8>
// CHECK: tt.reshape {{.*}} : tensor<4x16x32x4xi8> -> tensor<4x16x128xi8>

// -----

module {
  tt.func public @polynomial_runtime_stride_and_unsigned_narrow(
      %base: !tt.ptr<i8>, %pid_n: i32, %k_idx: i32, %ng_total: i32, %N: i32) -> tensor<16x256xi8> {
    %c16 = arith.constant 16 : i32
    %n_base = arith.muli %pid_n, %c16 : i32
    %n_base_v = tt.splat %n_base : i32 -> tensor<16xi32>
    %rows = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %n_idx = arith.addi %n_base_v, %rows : tensor<16xi32>
    %n_col = tt.expand_dims %n_idx {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %n_bcast = tt.broadcast %n_col : tensor<16x1xi32> -> tensor<16x256xi32>

    %N_col = tt.splat %N : i32 -> tensor<16x1xi32>
    %n_mask_col = arith.cmpi slt, %n_col, %N_col : tensor<16x1xi32>
    %mask = tt.broadcast %n_mask_col : tensor<16x1xi1> -> tensor<16x256xi1>
    %other = arith.constant dense<0> : tensor<16x256xi8>

    %cols = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %c256_v = arith.constant dense<256> : tensor<256xi32>
    %grp_1d = arith.divsi %cols, %c256_v : tensor<256xi32>
    %rem_1d = arith.remsi %cols, %c256_v : tensor<256xi32>
    %u8_1d = arith.trunci %rem_1d : tensor<256xi32> to tensor<256xi8>
    %u8_row = tt.expand_dims %u8_1d {axis = 0 : i32} : tensor<256xi8> -> tensor<1x256xi8>
    %u8_b = tt.broadcast %u8_row : tensor<1x256xi8> -> tensor<16x256xi8>
    %c1_u8 = arith.constant dense<1> : tensor<16x256xi8>
    %byte_u8 = arith.shrui %u8_b, %c1_u8 : tensor<16x256xi8>
    %byte_i32 = arith.extui %byte_u8 : tensor<16x256xi8> to tensor<16x256xi32>

    %grp_row = tt.expand_dims %grp_1d {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %grp_b = tt.broadcast %grp_row : tensor<1x256xi32> -> tensor<16x256xi32>
    %k_b = tt.splat %k_idx : i32 -> tensor<16x256xi32>
    %k_grp = arith.addi %k_b, %grp_b : tensor<16x256xi32>

    %ng_b = tt.splat %ng_total : i32 -> tensor<16x256xi32>
    %row_blk = arith.muli %n_bcast, %ng_b : tensor<16x256xi32>
    %blk = arith.addi %row_blk, %k_grp : tensor<16x256xi32>
    %c128_b = arith.constant dense<128> : tensor<16x256xi32>
    %blk_bytes = arith.muli %blk, %c128_b : tensor<16x256xi32>
    %off = arith.addi %blk_bytes, %byte_i32 : tensor<16x256xi32>

    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<16x256x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %off : tensor<16x256x!tt.ptr<i8>>, tensor<16x256xi32>
    %v = tt.load %ptr, %mask, %other : tensor<16x256x!tt.ptr<i8>>
    tt.return %v : tensor<16x256xi8>
  }
}

// Polynomial scalar base `(pid_n * 16 * ng_total) * 128` + runtime outer stride `ng_total * 128`
// + unsigned narrow `uint8` `[0..255]` (`trunci` + `shrui` + `extui`) coalesces to `16x128`.
// CHECK-LABEL: tt.func public @polynomial_runtime_stride_and_unsigned_narrow
// CHECK: tt.broadcast {{.*}} : tensor<16x1xi1> -> tensor<16x128xi1>
// CHECK: tt.addptr %arg0, {{.*}} : !tt.ptr<i8>, i32
// CHECK: tt.load {{.*}}, {{.*}}, {{.*}} : tensor<16x128x!tt.ptr<i8>>
// CHECK: tt.expand_dims {{.*}} {axis = 2 : i32} : tensor<16x128xi8> -> tensor<16x128x1xi8>
// CHECK: tt.broadcast {{.*}} : tensor<16x128x1xi8> -> tensor<16x128x2xi8>
// CHECK: tt.reshape {{.*}} : tensor<16x128x2xi8> -> tensor<16x256xi8>



