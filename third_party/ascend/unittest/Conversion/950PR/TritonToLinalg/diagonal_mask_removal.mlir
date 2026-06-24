// RUN: triton-opt %s --triton-to-unstructure='compile-on-910-95=true force-simt-template=true' \
// RUN:                --triton-to-linalg='compile-on-910-95=true' --split-input-file \
// RUN: | FileCheck %s

// -----
// Positive case: forward cumulative sum with K=+1 (sub-diagonal left shift).
// Pattern: reduce(axis=1, select(cmpi eq(row, col+1), broadcast(expand_dims(scan(x),0)), 0))
// Should be replaced by: subf(scan_result, scan_input).
// CHECK-LABEL: func.func @diagonal_shift_forward_k_plus1
// CHECK-NOT: tt.reduce
// CHECK: arith.subf
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @diagonal_shift_forward_k_plus1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                  %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

    // cumulative sum (forward, axis=0 on 1-D => trivially axis=0)
    %scan = tt.scan %x {axis = 0 : i32, reverse = false} : (tensor<16xf32>) -> tensor<16xf32> {
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.scan.return %s : f32
    }

    // expand_dims axis=0 -> broadcast to 16x16
    %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
    %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>

    // Build row indices (expand on axis=1, broadcast)
    %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>

    // Build col+1 indices (expand on axis=0, add 1, broadcast)
    %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %one = arith.constant dense<1> : tensor<1x16xi32>
    %col_plus1 = arith.addi %col_expd, %one : tensor<1x16xi32>
    %col_bc = tt.broadcast %col_plus1 : tensor<1x16xi32> -> tensor<16x16xi32>

    // cmpi eq (row == col + 1)
    %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>

    // select(cmp, broadcast_scan, zeros)
    %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>

    // reduce sum axis=1
    %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x16xf32>) -> tensor<16xf32>

    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Negative: reduce axis is NOT 1 (axis=0). The pass must not fire.
// CHECK-LABEL: func.func @diagonal_shift_wrong_reduce_axis
// CHECK: tt.reduce
// CHECK-NOT: arith.subf
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @diagonal_shift_wrong_reduce_axis(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>
    %scan = tt.scan %x {axis = 0 : i32, reverse = false} : (tensor<16xf32>) -> tensor<16xf32> {
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.scan.return %s : f32
    }
    %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
    %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>
    %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>
    %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %one = arith.constant dense<1> : tensor<1x16xi32>
    %col_plus1 = arith.addi %col_expd, %one : tensor<1x16xi32>
    %col_bc = tt.broadcast %col_plus1 : tensor<1x16xi32> -> tensor<16x16xi32>
    %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
    %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>
    // Wrong axis: reduce on axis=0 instead of 1
    %red = "tt.reduce"(%sel) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x16xf32>) -> tensor<16xf32>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Negative: K=+2 (not ±1). The pass only fires for unit diagonal shifts.
// CHECK-LABEL: func.func @diagonal_shift_k_too_large
// CHECK: tt.reduce
// CHECK-NOT: arith.subf
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @diagonal_shift_k_too_large(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                              %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>
    %scan = tt.scan %x {axis = 0 : i32, reverse = false} : (tensor<16xf32>) -> tensor<16xf32> {
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.scan.return %s : f32
    }
    %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
    %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>
    %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>
    %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %two = arith.constant dense<2> : tensor<1x16xi32>
    %col_plus2 = arith.addi %col_expd, %two : tensor<1x16xi32>
    %col_bc = tt.broadcast %col_plus2 : tensor<1x16xi32> -> tensor<16x16xi32>
    %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
    %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>
    %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x16xf32>) -> tensor<16xf32>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Negative: scan direction mismatch. K=+1 requires forward scan (reverse=false),
// but here scan is reverse=true. The pass must NOT fire.
// CHECK-LABEL: func.func @diagonal_shift_scan_direction_mismatch
// CHECK: tt.reduce
// CHECK-NOT: arith.subf
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @diagonal_shift_scan_direction_mismatch(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                          %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>
    // reverse=true but K=+1 expects reverse=false
    %scan = tt.scan %x {axis = 0 : i32, reverse = true} : (tensor<16xf32>) -> tensor<16xf32> {
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.scan.return %s : f32
    }
    %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
    %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>
    %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>
    %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %one = arith.constant dense<1> : tensor<1x16xi32>
    %col_plus1 = arith.addi %col_expd, %one : tensor<1x16xi32>
    %col_bc = tt.broadcast %col_plus1 : tensor<1x16xi32> -> tensor<16x16xi32>
    %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
    %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>
    %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x16xf32>) -> tensor<16xf32>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Negative: false value in select is NOT zero. Pass requires isZeroFloat to
// confirm the false branch contributes nothing.
// CHECK-LABEL: func.func @diagonal_shift_nonzero_false_value
// CHECK: tt.reduce
// CHECK-NOT: arith.subf
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @diagonal_shift_nonzero_false_value(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>
    %scan = tt.scan %x {axis = 0 : i32, reverse = false} : (tensor<16xf32>) -> tensor<16xf32> {
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.scan.return %s : f32
    }
    %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
    %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>
    %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>
    %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %one = arith.constant dense<1> : tensor<1x16xi32>
    %col_plus1 = arith.addi %col_expd, %one : tensor<1x16xi32>
    %col_bc = tt.broadcast %col_plus1 : tensor<1x16xi32> -> tensor<16x16xi32>
    %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
    // false value is 1.0, NOT zero — pass must not fire
    %ones = arith.constant dense<1.000000e+00> : tensor<16x16xf32>
    %sel = arith.select %cmp, %bcast, %ones : tensor<16x16xi1>, tensor<16x16xf32>
    %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x16xf32>) -> tensor<16xf32>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Negative: combine region is mulf (multiply), not addf. The cumulative sum
// identity only applies to additive reductions.
// CHECK-LABEL: func.func @diagonal_shift_mulf_combine
// CHECK: tt.reduce
// CHECK-NOT: arith.subf
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @diagonal_shift_mulf_combine(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                               %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>
    %scan = tt.scan %x {axis = 0 : i32, reverse = false} : (tensor<16xf32>) -> tensor<16xf32> {
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.scan.return %s : f32
    }
    %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
    %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>
    %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>
    %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %one = arith.constant dense<1> : tensor<1x16xi32>
    %col_plus1 = arith.addi %col_expd, %one : tensor<1x16xi32>
    %col_bc = tt.broadcast %col_plus1 : tensor<1x16xi32> -> tensor<16x16xi32>
    %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
    %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>
    // Reduce with MULF, not ADDF
    %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.mulf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x16xf32>) -> tensor<16xf32>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Negative: non-square broadcast (16x8). The pass requires NxN square.
// CHECK-LABEL: func.func @diagonal_shift_non_square
// CHECK: tt.reduce
// CHECK-NOT: arith.subf
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @diagonal_shift_non_square(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                             %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %range8 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %src_ptr = tt.addptr %src_base, %range8 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %x = tt.load %src_ptr : tensor<8x!tt.ptr<f32>>
    %scan = tt.scan %x {axis = 0 : i32, reverse = false} : (tensor<8xf32>) -> tensor<8xf32> {
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.scan.return %s : f32
    }
    // expand_dims to [1,8], broadcast to [16,8] (non-square!)
    %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
    %bcast = tt.broadcast %expd : tensor<1x8xf32> -> tensor<16x8xf32>
    %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x8xi32>
    %col_range = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %one = arith.constant dense<1> : tensor<1x8xi32>
    %col_plus1 = arith.addi %col_expd, %one : tensor<1x8xi32>
    %col_bc = tt.broadcast %col_plus1 : tensor<1x8xi32> -> tensor<16x8xi32>
    %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x8xi32>
    %zeros = arith.constant dense<0.000000e+00> : tensor<16x8xf32>
    %sel = arith.select %cmp, %bcast, %zeros : tensor<16x8xi1>, tensor<16x8xf32>
    %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x8xf32>) -> tensor<16xf32>
    %range16 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %range16 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Negative: predicate is slt (lower-triangular), NOT eq (diagonal).
// CHECK-LABEL: func.func @diagonal_shift_slt_predicate
// CHECK: tt.reduce
// CHECK-NOT: arith.subf
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @diagonal_shift_slt_predicate(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>
    %scan = tt.scan %x {axis = 0 : i32, reverse = false} : (tensor<16xf32>) -> tensor<16xf32> {
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.scan.return %s : f32
    }
    %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
    %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>
    %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>
    %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<16x16xi32>
    // slt (row < col) instead of eq — NOT a diagonal pattern
    %cmp = arith.cmpi slt, %row_bc, %col_bc : tensor<16x16xi32>
    %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>
    %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x16xf32>) -> tensor<16xf32>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Negative: scan combine is NOT addf (uses maxf instead).
// CHECK-LABEL: func.func @diagonal_shift_scan_not_addf
// CHECK: tt.reduce
// CHECK-NOT: arith.subf
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @diagonal_shift_scan_not_addf(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>
    // scan uses maxf, not addf
    %scan = tt.scan %x {axis = 0 : i32, reverse = false} : (tensor<16xf32>) -> tensor<16xf32> {
    ^bb0(%a: f32, %b: f32):
      %s = arith.maximumf %a, %b : f32
      tt.scan.return %s : f32
    }
    %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
    %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>
    %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>
    %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %one = arith.constant dense<1> : tensor<1x16xi32>
    %col_plus1 = arith.addi %col_expd, %one : tensor<1x16xi32>
    %col_bc = tt.broadcast %col_plus1 : tensor<1x16xi32> -> tensor<16x16xi32>
    %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
    %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>
    %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x16xf32>) -> tensor<16xf32>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Positive case: reverse scan with K=-1 (super-diagonal right shift via subi).
// K < 0 requires reverse=true. Pattern: row == col - 1.
// CHECK-LABEL: func.func @diagonal_shift_reverse_k_minus1
// CHECK-NOT: tt.reduce
// CHECK: arith.subf
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @diagonal_shift_reverse_k_minus1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                   %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

    // reverse cumulative sum
    %scan = tt.scan %x {axis = 0 : i32, reverse = true} : (tensor<16xf32>) -> tensor<16xf32> {
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.scan.return %s : f32
    }

    %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
    %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>

    %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>

    %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    // col - 1 means K = -1
    %one = arith.constant dense<1> : tensor<1x16xi32>
    %col_minus1 = arith.subi %col_expd, %one : tensor<1x16xi32>
    %col_bc = tt.broadcast %col_minus1 : tensor<1x16xi32> -> tensor<16x16xi32>

    %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
    %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>

    %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x16xf32>) -> tensor<16xf32>

    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
