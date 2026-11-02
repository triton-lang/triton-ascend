// RUN: triton-opt %s --triton-to-linalg --verify-each --split-input-file | FileCheck %s

// A full 2x256x16 static offset tile from the failing CI shape. Its five
// sanitizer checks cover two multiplies, two additions and reverse indexing.
// The device still stores the i32 offsets, while all check-only i64 tensors
// disappear before bufferization can allocate UB for them.
// CHECK-LABEL: func.func @safe_static_3d_offsets(
// CHECK-SAME: {{.*}} {
// CHECK-NOT: i64
// CHECK-NOT: call @triton_assert
// CHECK: bufferization.materialize_in_destination
// CHECK-NOT: i64
// CHECK-NOT: call @triton_assert
// CHECK: return
tt.func @safe_static_3d_offsets(%out: !tt.ptr<i32>) {
  %x = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %y = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %z = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %x1 = tt.expand_dims %x {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %x2 = tt.expand_dims %x1 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
  %xx = tt.broadcast %x2 : tensor<2x1x1xi32> -> tensor<2x256x16xi32>
  %y1 = tt.expand_dims %y {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
  %y2 = tt.expand_dims %y1 {axis = 2 : i32} : tensor<1x256xi32> -> tensor<1x256x1xi32>
  %yy = tt.broadcast %y2 : tensor<1x256x1xi32> -> tensor<2x256x16xi32>
  %z1 = tt.expand_dims %z {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %z2 = tt.expand_dims %z1 {axis = 0 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
  %zz = tt.broadcast %z2 : tensor<1x1x16xi32> -> tensor<2x256x16xi32>
  %min = arith.constant dense<-2147483648> : tensor<2x256x16xi64>
  %max = arith.constant dense<2147483647> : tensor<2x256x16xi64>
  %stride_x = arith.constant dense<4096> : tensor<2x256x16xi32>
  %stride_y = arith.constant dense<16> : tensor<2x256x16xi32>
  %last = arith.constant dense<8191> : tensor<2x256x16xi32>
  %xoff_lhs = arith.extsi %xx : tensor<2x256x16xi32> to tensor<2x256x16xi64>
  %xoff_rhs = arith.extsi %stride_x : tensor<2x256x16xi32> to tensor<2x256x16xi64>
  %xoff_wide = arith.muli %xoff_lhs, %xoff_rhs : tensor<2x256x16xi64>
  %xoff_upper = arith.cmpi sle, %xoff_wide, %max : tensor<2x256x16xi64>
  %xoff_lower = arith.cmpi sge, %xoff_wide, %min : tensor<2x256x16xi64>
  %xoff_ok = arith.andi %xoff_upper, %xoff_lower : tensor<2x256x16xi1>
  tt.assert %xoff_ok, "int32 overflow detected for operation mul" {tt.auto_overflow_assert} : tensor<2x256x16xi1>
  %xoff = arith.muli %xx, %stride_x : tensor<2x256x16xi32>
  %yoff_lhs = arith.extsi %yy : tensor<2x256x16xi32> to tensor<2x256x16xi64>
  %yoff_rhs = arith.extsi %stride_y : tensor<2x256x16xi32> to tensor<2x256x16xi64>
  %yoff_wide = arith.muli %yoff_lhs, %yoff_rhs : tensor<2x256x16xi64>
  %yoff_upper = arith.cmpi sle, %yoff_wide, %max : tensor<2x256x16xi64>
  %yoff_lower = arith.cmpi sge, %yoff_wide, %min : tensor<2x256x16xi64>
  %yoff_ok = arith.andi %yoff_upper, %yoff_lower : tensor<2x256x16xi1>
  tt.assert %yoff_ok, "int32 overflow detected for operation mul" {tt.auto_overflow_assert} : tensor<2x256x16xi1>
  %yoff = arith.muli %yy, %stride_y : tensor<2x256x16xi32>
  %xy_lhs = arith.extsi %xoff : tensor<2x256x16xi32> to tensor<2x256x16xi64>
  %xy_rhs = arith.extsi %yoff : tensor<2x256x16xi32> to tensor<2x256x16xi64>
  %xy_wide = arith.addi %xy_lhs, %xy_rhs : tensor<2x256x16xi64>
  %xy_upper = arith.cmpi sle, %xy_wide, %max : tensor<2x256x16xi64>
  %xy_lower = arith.cmpi sge, %xy_wide, %min : tensor<2x256x16xi64>
  %xy_ok = arith.andi %xy_upper, %xy_lower : tensor<2x256x16xi1>
  tt.assert %xy_ok, "int32 overflow detected for operation add" {tt.auto_overflow_assert} : tensor<2x256x16xi1>
  %xy = arith.addi %xoff, %yoff : tensor<2x256x16xi32>
  %xyz_lhs = arith.extsi %xy : tensor<2x256x16xi32> to tensor<2x256x16xi64>
  %xyz_rhs = arith.extsi %zz : tensor<2x256x16xi32> to tensor<2x256x16xi64>
  %xyz_wide = arith.addi %xyz_lhs, %xyz_rhs : tensor<2x256x16xi64>
  %xyz_upper = arith.cmpi sle, %xyz_wide, %max : tensor<2x256x16xi64>
  %xyz_lower = arith.cmpi sge, %xyz_wide, %min : tensor<2x256x16xi64>
  %xyz_ok = arith.andi %xyz_upper, %xyz_lower : tensor<2x256x16xi1>
  tt.assert %xyz_ok, "int32 overflow detected for operation add" {tt.auto_overflow_assert} : tensor<2x256x16xi1>
  %xyz = arith.addi %xy, %zz : tensor<2x256x16xi32>
  %reverse_lhs = arith.extsi %last : tensor<2x256x16xi32> to tensor<2x256x16xi64>
  %reverse_rhs = arith.extsi %xyz : tensor<2x256x16xi32> to tensor<2x256x16xi64>
  %reverse_wide = arith.subi %reverse_lhs, %reverse_rhs : tensor<2x256x16xi64>
  %reverse_upper = arith.cmpi sle, %reverse_wide, %max : tensor<2x256x16xi64>
  %reverse_lower = arith.cmpi sge, %reverse_wide, %min : tensor<2x256x16xi64>
  %reverse_ok = arith.andi %reverse_upper, %reverse_lower : tensor<2x256x16xi1>
  tt.assert %reverse_ok, "int32 overflow detected for operation sub" {tt.auto_overflow_assert} : tensor<2x256x16xi1>
  %reverse = arith.subi %last, %xyz : tensor<2x256x16xi32>
  %base = tt.splat %out : !tt.ptr<i32> -> tensor<2x256x16x!tt.ptr<i32>>
  %ptrs = tt.addptr %base, %xyz : tensor<2x256x16x!tt.ptr<i32>>, tensor<2x256x16xi32>
  tt.store %ptrs, %reverse : tensor<2x256x16x!tt.ptr<i32>>
  tt.return
}

// -----

// Inclusive signed limits and a negative multiplication are safe for every lane.
// CHECK-LABEL: func.func @safe_signed_boundaries(
// CHECK-SAME: {{.*}} {
// CHECK-NOT: call @triton_assert
// CHECK-NOT: i64
// CHECK: return
tt.func @safe_signed_boundaries() {
  %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %w = arith.extsi %r : tensor<4xi32> to tensor<4xi64>
  %min = arith.constant dense<-2147483648> : tensor<4xi64>
  %max = arith.constant dense<2147483647> : tensor<4xi64>
  %near_max = arith.constant dense<2147483644> : tensor<4xi64>
  %abs_min = arith.constant dense<2147483648> : tensor<4xi64>
  %negative = arith.constant dense<-1024> : tensor<4xi64>
  %at_max = arith.addi %w, %near_max : tensor<4xi64>
  %at_max_upper = arith.cmpi sle, %at_max, %max : tensor<4xi64>
  %at_max_lower = arith.cmpi sge, %at_max, %min : tensor<4xi64>
  %at_max_ok = arith.andi %at_max_upper, %at_max_lower : tensor<4xi1>
  tt.assert %at_max_ok, "int32 overflow detected for operation add" {tt.auto_overflow_assert} : tensor<4xi1>
  %at_min = arith.subi %w, %abs_min : tensor<4xi64>
  %at_min_upper = arith.cmpi sle, %at_min, %max : tensor<4xi64>
  %at_min_lower = arith.cmpi sge, %at_min, %min : tensor<4xi64>
  %at_min_ok = arith.andi %at_min_upper, %at_min_lower : tensor<4xi1>
  tt.assert %at_min_ok, "int32 overflow detected for operation sub" {tt.auto_overflow_assert} : tensor<4xi1>
  %negative_product = arith.muli %w, %negative : tensor<4xi64>
  %negative_product_upper = arith.cmpi sle, %negative_product, %max : tensor<4xi64>
  %negative_product_lower = arith.cmpi sge, %negative_product, %min : tensor<4xi64>
  %negative_product_ok = arith.andi %negative_product_upper, %negative_product_lower : tensor<4xi1>
  tt.assert %negative_product_ok, "int32 overflow detected for operation mul" {tt.auto_overflow_assert} : tensor<4xi1>
  tt.return
}

// -----

// uint32's upper bound must remain positive after widening, including values
// above INT32_MAX. Both lanes are representable, with the last at UINT32_MAX.
// CHECK-LABEL: func.func @safe_unsigned_boundary(
// CHECK-SAME: {{.*}} {
// CHECK-NOT: call @triton_assert
// CHECK-NOT: i64
// CHECK: return
tt.func @safe_unsigned_boundary() {
  %r = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %w = arith.extui %r : tensor<2xi32> to tensor<2xi64>
  %min = arith.constant dense<0> : tensor<2xi64>
  %max = arith.constant dense<4294967295> : tensor<2xi64>
  %base = arith.constant dense<4294967294> : tensor<2xi64>
  %sum = arith.addi %w, %base : tensor<2xi64>
  %upper = arith.cmpi sle, %sum, %max : tensor<2xi64>
  %lower = arith.cmpi sge, %sum, %min : tensor<2xi64>
  %ok = arith.andi %upper, %lower : tensor<2xi1>
  tt.assert %ok, "int32 overflow detected for operation add" {tt.auto_overflow_assert} : tensor<2xi1>
  tt.return
}

// -----

// Runtime values cannot be declared safe from their storage type alone.
// CHECK-LABEL: func.func @dynamic_check_remains(
// CHECK: arith.extsi
// CHECK: arith.addi
// CHECK: call @triton_assert
// CHECK: return
tt.func @dynamic_check_remains(%x: i32, %y: i32) {
  %min = arith.constant -2147483648 : i64
  %max = arith.constant 2147483647 : i64
  %wx = arith.extsi %x : i32 to i64
  %wy = arith.extsi %y : i32 to i64
  %sum = arith.addi %wx, %wy : i64
  %upper = arith.cmpi sle, %sum, %max : i64
  %lower = arith.cmpi sge, %sum, %min : i64
  %ok = arith.andi %upper, %lower : i1
  tt.assert %ok, "int32 overflow detected for operation add" {tt.auto_overflow_assert} : i1
  tt.return
}

// -----

// One overflowing lane must preserve the entire tensor assertion.
// CHECK-LABEL: func.func @one_lane_overflows(
// CHECK: call @triton_assert
// CHECK: return
tt.func @one_lane_overflows() {
  %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %w = arith.extsi %r : tensor<4xi32> to tensor<4xi64>
  %min = arith.constant dense<-2147483648> : tensor<4xi64>
  %max = arith.constant dense<2147483647> : tensor<4xi64>
  %base = arith.constant dense<2147483645> : tensor<4xi64>
  %sum = arith.addi %w, %base : tensor<4xi64>
  %upper = arith.cmpi sle, %sum, %max : tensor<4xi64>
  %lower = arith.cmpi sge, %sum, %min : tensor<4xi64>
  %ok = arith.andi %upper, %lower : tensor<4xi1>
  tt.assert %ok, "int32 overflow detected for operation add" {tt.auto_overflow_assert} : tensor<4xi1>
  tt.return
}

// -----

// Lower-bound overflow is retained even though all upper-bound checks succeed.
// CHECK-LABEL: func.func @negative_overflow_remains(
// CHECK: call @triton_assert
// CHECK: return
tt.func @negative_overflow_remains() {
  %r = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %w = arith.extsi %r : tensor<2xi32> to tensor<2xi64>
  %min = arith.constant dense<-2147483648> : tensor<2xi64>
  %max = arith.constant dense<2147483647> : tensor<2xi64>
  %base = arith.constant dense<2147483649> : tensor<2xi64>
  %difference = arith.subi %w, %base : tensor<2xi64>
  %upper = arith.cmpi sle, %difference, %max : tensor<2xi64>
  %lower = arith.cmpi sge, %difference, %min : tensor<2xi64>
  %ok = arith.andi %upper, %lower : tensor<2xi1>
  tt.assert %ok, "int32 overflow detected for operation sub" {tt.auto_overflow_assert} : tensor<2xi1>
  tt.return
}

// -----

// The source i32 bit pattern -1 widens to UINT32_MAX, not -1.
// CHECK-LABEL: func.func @unsigned_overflow_remains(
// CHECK: call @triton_assert
// CHECK: return
tt.func @unsigned_overflow_remains() {
  %r = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %max32 = arith.constant dense<-1> : tensor<2xi32>
  %w = arith.extui %r : tensor<2xi32> to tensor<2xi64>
  %base = arith.extui %max32 : tensor<2xi32> to tensor<2xi64>
  %min = arith.constant dense<0> : tensor<2xi64>
  %max = arith.constant dense<4294967295> : tensor<2xi64>
  %sum = arith.addi %w, %base : tensor<2xi64>
  %upper = arith.cmpi sle, %sum, %max : tensor<2xi64>
  %lower = arith.cmpi sge, %sum, %min : tensor<2xi64>
  %ok = arith.andi %upper, %lower : tensor<2xi1>
  tt.assert %ok, "int32 overflow detected for operation add" {tt.auto_overflow_assert} : tensor<2xi1>
  tt.return
}

// -----

// A statically overflowing check must still reach the device assertion.
// CHECK-LABEL: func.func @constant_overflow_remains(
// CHECK: call @triton_assert
// CHECK: return
tt.func @constant_overflow_remains() {
  %overflow = arith.constant 2147483648 : i64
  %max = arith.constant 2147483647 : i64
  %ok = arith.cmpi sle, %overflow, %max : i64
  tt.assert %ok, "int32 overflow detected for operation add" {tt.auto_overflow_assert} : i1
  tt.return
}

// -----

// An overflowing narrow operation cannot acquire a mathematical range from
// nsw/nuw flags. The second lane exceeds the signed i32 range.
// CHECK-LABEL: func.func @wrapping_narrow_op_remains(
// CHECK: call @triton_assert
// CHECK: return
tt.func @wrapping_narrow_op_remains() {
  %r = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %max = arith.constant dense<2147483647> : tensor<2xi32>
  %wrapped = arith.addi %r, %max overflow<nsw, nuw> : tensor<2xi32>
  %w = arith.extsi %wrapped : tensor<2xi32> to tensor<2xi64>
  %zero = arith.constant dense<0> : tensor<2xi64>
  %ok = arith.cmpi sge, %w, %zero : tensor<2xi64>
  tt.assert %ok, "int32 overflow detected for operation add" {tt.auto_overflow_assert} : tensor<2xi1>
  tt.return
}

// -----

// The message does not identify compiler-generated assertions. A user assertion
// with the same text is preserved, even when its condition is statically true.
// CHECK-LABEL: func.func @user_assert_remains(
// CHECK: call @triton_assert
// CHECK: return
tt.func @user_assert_remains() {
  %true = arith.constant true
  tt.assert %true, "int32 overflow detected for operation add" : i1
  tt.return
}

// -----

// Signed make_range bounds stay negative instead of becoming unsigned values.
// CHECK-LABEL: func.func @safe_negative_range(
// CHECK-SAME: {{.*}} {
// CHECK-NOT: call @triton_assert
// CHECK-NOT: i64
// CHECK: return
tt.func @safe_negative_range() {
  %r = tt.make_range {start = -4 : i32, end = 0 : i32} : tensor<4xi32>
  %w = arith.extsi %r : tensor<4xi32> to tensor<4xi64>
  %zero = arith.constant dense<0> : tensor<4xi64>
  %negative = arith.cmpi slt, %w, %zero : tensor<4xi64>
  tt.assert %negative, "int32 overflow detected for operation sub" {tt.auto_overflow_assert} : tensor<4xi1>
  tt.return
}

// -----

// i1's set bit is 1 when zero-extended and -1 when sign-extended.
// CHECK-LABEL: func.func @safe_boolean_extensions(
// CHECK-SAME: {{.*}} {
// CHECK-NOT: call @triton_assert
// CHECK: return
tt.func @safe_boolean_extensions() {
  %true = arith.constant true
  %one = arith.constant 1 : i32
  %minus_one = arith.constant -1 : i32
  %unsigned = arith.extui %true : i1 to i32
  %signed = arith.extsi %true : i1 to i32
  %unsigned_ok = arith.cmpi eq, %unsigned, %one : i32
  %signed_ok = arith.cmpi eq, %signed, %minus_one : i32
  %ok = arith.andi %unsigned_ok, %signed_ok : i1
  tt.assert %ok, "int32 overflow detected for operation add" {tt.auto_overflow_assert} : i1
  tt.return
}

// -----

// Signed arithmetic -1 + 1 fits i32, but the corresponding unsigned operation
// overflows. The nuw flag therefore introduces a poison dependency, which must
// not be treated as a proven-safe sanitizer condition.
// CHECK-LABEL: func.func @nuw_signed_fit_remains(
// CHECK: call @triton_assert
// CHECK: return
tt.func @nuw_signed_fit_remains() {
  %r = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %minus_one = arith.constant dense<-1> : tensor<2xi32>
  %sum = arith.addi %r, %minus_one overflow<nuw> : tensor<2xi32>
  %w = arith.extsi %sum : tensor<2xi32> to tensor<2xi64>
  %min = arith.constant dense<-1> : tensor<2xi64>
  %ok = arith.cmpi sge, %w, %min : tensor<2xi64>
  tt.assert %ok, "int32 overflow detected for operation add" {tt.auto_overflow_assert} : tensor<2xi1>
  tt.return
}

// -----

// nsw truncation of 3 to i1 is poison rather than a known true condition.
// CHECK-LABEL: func.func @nsw_truncation_remains(
// CHECK: call @triton_assert
// CHECK: return
tt.func @nsw_truncation_remains() {
  %three = arith.constant 3 : i32
  %narrow = "arith.trunci"(%three) {overflowFlags = #arith.overflow<nsw>} : (i32) -> i1
  tt.assert %narrow, "int32 overflow detected for operation add" {tt.auto_overflow_assert} : i1
  tt.return
}
