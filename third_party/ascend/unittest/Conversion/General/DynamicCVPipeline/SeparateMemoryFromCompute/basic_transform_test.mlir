// RUN: triton-opt --gm-load-multi-buffer %s | FileCheck %s

// ============================================================================
// TC-B01: Basic double buffer (depth=2) transformation
// Verifies a single Q load with gm_load_bufferable is correctly rewritten
// into a 2-slot double-buffer structure with Producer/Consumer split.
// ============================================================================

// CHECK-LABEL: func.func @tc_b01_double_buffer_basic
func.func @tc_b01_double_buffer_basic(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg1: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %c28_i32 = arith.constant 28 : i32
  %c65536_i32 = arith.constant 65536 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index

  // Two buffer slots allocated before loop
  // CHECK: %[[Q_SLOT0:.*]] = memref.alloc() : memref<128x128xf16>
  // CHECK: %[[Q_SLOT1:.*]] = memref.alloc() : memref<128x128xf16>

  // Trip count computed via ceildivui
  // CHECK: arith.ceildivui

  // scf.for gets 4 extra iter_args: 2 flags + prod_counter + cons_counter
  // CHECK: scf.for {{.+}} iter_args({{.+}}) -> (tensor<128x128xf32>, i1, i1, index, index)

  // Producer fills slots via memref.copy
  // CHECK: memref.copy {{.*}}, %[[Q_SLOT0]]
  // CHECK: memref.copy {{.*}}, %[[Q_SLOT1]]

  // Consumer uses remui + select for FIFO slot selection
  // CHECK: arith.remui {{.*}}, %c2
  // CHECK: bufferization.to_tensor %[[Q_SLOT0]]
  // CHECK: bufferization.to_tensor %[[Q_SLOT1]]
  // CHECK: arith.select

  // Consumer body uses selected tensor for matmul
  // CHECK: linalg.matmul

  %result = scf.for %arg16 = %c0_i32 to %c65536_i32 step %c28_i32
    iter_args(%acc = %arg1) -> tensor<128x128xf32> : i32 {
    %iv_idx = arith.index_cast %arg16 : i32 to index
    %row_off = arith.muli %iv_idx, %c128 : index
    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %q_src_rc, %q_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %q_tensor = bufferization.to_tensor %q_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16>
    %qk = linalg.matmul {input_precision = "ieee"} ins(%q_tensor, %q_tensor : tensor<128x128xf16>, tensor<128x128xf16>) outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %qk : tensor<128x128xf32>
  }
  return %result : tensor<128x128xf32>
}

// ============================================================================
// TC-B02: No gm_load_bufferable marker -> Pass-through
// Verifies that without the gm_load_bufferable attribute, the pass does
// not transform the IR at all.
// ============================================================================

// CHECK-LABEL: func.func @tc_b02_no_marker_pass_through
func.func @tc_b02_no_marker_pass_through(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}
) {
  %c28_i32 = arith.constant 28 : i32
  %c65536_i32 = arith.constant 65536 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index

  // No transformation without marker
  // CHECK-NOT: arith.ceildivui
  // CHECK-NOT: arith.remui
  // CHECK-NOT: arith.select
  // CHECK: scf.for {{.*}} : i32 {
  // CHECK: memref.copy

  scf.for %arg16 = %c0_i32 to %c65536_i32 step %c28_i32 : i32 {
    %iv_idx = arith.index_cast %arg16 : i32 to index
    %row_off = arith.muli %iv_idx, %c128 : index
    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %q_src_rc, %q_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %q_tensor = bufferization.to_tensor %q_alloc restrict writable : memref<128x128xf16>
  }
  return
}

// ============================================================================
// TC-B03: Trip count <= depth -> skip transformation
// When the loop trip count (compile-time known) is <= buffer depth (2),
// the pass should skip transformation since there is no pipeline benefit.
// ============================================================================

// CHECK-LABEL: func.func @tc_b03_trip_count_too_small
func.func @tc_b03_trip_count_too_small(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}
) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index

  // Trip count = ceildiv(2 - 0, 1) = 2 = depth -> skip
  // CHECK-NOT: arith.ceildivui
  // CHECK-NOT: arith.remui
  // CHECK: scf.for %{{.*}} = %c0 to %c2 step %c1
  // CHECK: memref.copy

  scf.for %iv = %c0 to %c2 step %c1 : index {
    %row_off = arith.muli %iv, %c128 : index
    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %q_src_rc, %q_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %q_tensor = bufferization.to_tensor %q_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16>
  }
  return
}

// ============================================================================
// TC-B03b: Trip count = depth + 1 -> should transform (minimal valid TC)
// Verifies that when trip count is just above depth, transformation happens.
// ============================================================================

// CHECK-LABEL: func.func @tc_b03b_trip_count_minimal_valid
func.func @tc_b03b_trip_count_minimal_valid(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}
) {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index

  // Trip count = 3 > depth(2) -> should transform
  // CHECK: arith.ceildivui
  // CHECK: arith.remui {{.*}}, %c2

  scf.for %iv = %c0 to %c3 step %c1 : index {
    %row_off = arith.muli %iv, %c128 : index
    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %q_src_rc, %q_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %q_tensor = bufferization.to_tensor %q_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16>
  }
  return
}

// ============================================================================
// TC-B13: No backing alloc in chain -> pass should skip
// When the gm_load_bufferable to_tensor has no associated memref.alloc
// in its dependency chain, the pass cannot create buffer slots and must skip.
// ============================================================================

// CHECK-LABEL: func.func @tc_b13_no_backing_alloc_skip
func.func @tc_b13_no_backing_alloc_skip(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}
) {
  %c28_i32 = arith.constant 28 : i32
  %c65536_i32 = arith.constant 65536 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index

  // No memref.alloc in chain -> pass skips this load
  // CHECK-NOT: arith.ceildivui
  // CHECK-NOT: arith.remui
  // CHECK: scf.for {{.*}} : i32 {
  // CHECK: bufferization.to_tensor {{.*}} {gm_load_bufferable}

  scf.for %arg16 = %c0_i32 to %c65536_i32 step %c28_i32 : i32 {
    %iv_idx = arith.index_cast %arg16 : i32 to index
    %row_off = arith.muli %iv_idx, %c128 : index
    // No alloc - direct reinterpret_cast used as to_tensor source
    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_tensor = bufferization.to_tensor %q_src_rc restrict writable {gm_load_bufferable} : memref<128x128xf16, strided<[128, 1], offset: ?>>
  }
  return
}

// ============================================================================
// TC-B15: No marked loads at all -> entire pass is a no-op
// Verifies the pass handles an empty scf.for with no gm_load_bufferable ops.
// ============================================================================

// CHECK-LABEL: func.func @tc_b15_no_marked_loads_noop
func.func @tc_b15_no_marked_loads_noop() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index

  // CHECK-NOT: arith.ceildivui
  // CHECK-NOT: arith.remui
  // CHECK: scf.for

  scf.for %iv = %c0 to %c10 step %c1 : index {
    %val = arith.addi %iv, %c1 : index
  }
  return
}
