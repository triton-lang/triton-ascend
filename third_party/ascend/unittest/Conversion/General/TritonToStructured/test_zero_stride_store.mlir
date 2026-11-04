// RUN: triton-opt --triton-to-structured --verify-each --split-input-file %s | FileCheck %s
// RUN: triton-opt '--triton-to-structured=optimize-dynamic-offset=true' --verify-each --split-input-file %s | FileCheck %s

// Zero-stride load axes may be removed and broadcast back. Stores must keep
// the pointer, value and mask iteration domains together. These cases use no
// model-specific shapes, names, compiler options or source fingerprints.

// CHECK-LABEL: tt.func public @copy_broadcast_i32
// CHECK: tt.store {{.*}} : tensor<8x!tt.ptr<i32>>
module {
  tt.func public @copy_broadcast_i32(%in: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
    %ip = tt.splat %in : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %v = tt.load %ip : tensor<1x!tt.ptr<i32>>
    %value = tt.broadcast %v : tensor<1xi32> -> tensor<8xi32>
    %op = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    tt.store %op, %value : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: tt.func public @constant_f32
// CHECK: tt.store {{.*}} : tensor<16x!tt.ptr<f32>>
module {
  tt.func public @constant_f32(%out: !tt.ptr<f32>) {
    %op = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %value = arith.constant dense<3.5> : tensor<16xf32>
    tt.store %op, %value : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Lane zero is masked off. Taking the first lane would silently drop this
// store. Leaving the full mask with its full pointer tensor preserves it.
// CHECK-LABEL: tt.func public @masked_i64
// CHECK: %[[MASK:.*]] = arith.constant dense<[false, false, true, false, false, false, false, false]> : tensor<8xi1>
// CHECK: tt.store {{.*}}, %[[MASK]] : tensor<8x!tt.ptr<i64>>
module {
  tt.func public @masked_i64(%out: !tt.ptr<i64>) {
    %op = tt.splat %out : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
    %value = arith.constant dense<41> : tensor<8xi64>
    %mask = arith.constant dense<[false, false, true, false, false, false, false, false]> : tensor<8xi1>
    tt.store %op, %value, %mask : tensor<8x!tt.ptr<i64>>
    tt.return
  }
}

// -----

// Lane-dependent values are legal when only one aliasing lane is active.
// CHECK-LABEL: tt.func public @masked_lane_values
// CHECK: %[[VALUES:.*]] = tt.make_range
// CHECK: tt.store {{.*}}, %[[VALUES]], {{.*}} : tensor<8x!tt.ptr<i32>>
module {
  tt.func public @masked_lane_values(%out: !tt.ptr<i32>) {
    %op = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %value = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %mask = arith.constant dense<[false, false, false, true, false, false, false, false]> : tensor<8xi1>
    tt.store %op, %value, %mask : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: tt.func public @two_zero_stride_axes
// CHECK: tt.store {{.*}} : tensor<2x8x!tt.ptr<f16>>
module {
  tt.func public @two_zero_stride_axes(%out: !tt.ptr<f16>) {
    %op = tt.splat %out : !tt.ptr<f16> -> tensor<2x8x!tt.ptr<f16>>
    %value = arith.constant dense<2.0> : tensor<2x8xf16>
    tt.store %op, %value : tensor<2x8x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// Nonzero-stride memory operations must still use normal linearization.
// CHECK-LABEL: tt.func public @contiguous_control
// CHECK: tt.store {{.*}} : tensor<8x!tt.ptr<i32>>
module {
  tt.func public @contiguous_control(%out: !tt.ptr<i32>) {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ptr = tt.addptr %base, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %ptr, %range : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// If mask analysis cannot linearize the remainder expression, leave the
// complete original pointer/value/mask domain intact.
// CHECK-LABEL: tt.func public @remainder_masked
// CHECK: tt.store {{.*}} : tensor<8x!tt.ptr<i32>>
module {
  tt.func public @remainder_masked(%out: !tt.ptr<i32>) {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %four = arith.constant dense<4> : tensor<8xi32>
    %offsets = arith.remsi %range, %four : tensor<8xi32>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ptr = tt.addptr %base, %offsets : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %mask = arith.cmpi slt, %range, %four : tensor<8xi32>
    tt.store %ptr, %range, %mask : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: tt.func public @quotient_masked
// CHECK: tt.store {{.*}} : tensor<8x!tt.ptr<i32>>
module {
  tt.func public @quotient_masked(%out: !tt.ptr<i32>) {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %four = arith.constant dense<4> : tensor<8xi32>
    %offsets = arith.divsi %range, %four : tensor<8xi32>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ptr = tt.addptr %base, %offsets : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %value = arith.constant dense<23> : tensor<8xi32>
    %mask = arith.cmpi slt, %range, %four : tensor<8xi32>
    tt.store %ptr, %value, %mask : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: tt.func public @quotient_store
// CHECK: tt.store {{.*}} : tensor<2x4x!tt.ptr<i32>>
module {
  tt.func public @quotient_store(%out: !tt.ptr<i32>) {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %four = arith.constant dense<4> : tensor<8xi32>
    %offsets = arith.divsi %range, %four : tensor<8xi32>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ptr = tt.addptr %base, %offsets : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %value = arith.constant dense<23> : tensor<8xi32>
    tt.store %ptr, %value : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}
