// RUN: triton-opt --gm-load-multi-buffer %s | FileCheck %s

// ============================================================================
// TC-B04: Producer IV projection correctness
// Verifies that the Producer uses projected IV (lb + prod_counter * step)
// instead of the current loop IV to compute prefetch addresses.
// ============================================================================

// CHECK-LABEL: func.func @tc_b04_iv_projection
func.func @tc_b04_iv_projection(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg1: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %c28_i32 = arith.constant 28 : i32
  %c65536_i32 = arith.constant 65536 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index

  // Producer computes virtual IV: lb_i32 + prod_counter_i32 * step_i32
  // CHECK: scf.if {{.*}} -> (i1, index) {
  // CHECK: arith.index_cast {{.*}} : index to i32
  // CHECK: arith.muli {{.*}}, %c28_i32 : i32
  // CHECK: arith.addi %c0_i32, {{.*}} : i32
  // CHECK: arith.index_cast {{.*}} : i32 to index
  // CHECK: memref.copy

  %result = scf.for %arg16 = %c0_i32 to %c65536_i32 step %c28_i32
    iter_args(%acc = %arg1) -> tensor<128x128xf32> : i32 {
    %iv_idx = arith.index_cast %arg16 : i32 to index
    %row_off = arith.muli %iv_idx, %c128 : index
    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %q_src_rc, %q_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %q_tensor = bufferization.to_tensor %q_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16> to tensor<128x128xf16>
    %qk = linalg.matmul {input_precision = "ieee"} ins(%q_tensor, %q_tensor : tensor<128x128xf16>, tensor<128x128xf16>) outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %qk : tensor<128x128xf32>
  }
  return %result : tensor<128x128xf32>
}

// ============================================================================
// TC-B04b: IV projection with runtime lower bound
// Verifies Producer IV projection when the loop lower bound is a runtime
// value (e.g., %offset_init), not a compile-time constant.
// ============================================================================

// CHECK-LABEL: func.func @tc_b04b_iv_projection_runtime_lb
func.func @tc_b04b_iv_projection_runtime_lb(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %offset_init: i32,
  %arg1: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %c28_i32 = arith.constant 28 : i32
  %c65536_i32 = arith.constant 65536 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index

  // Producer should use %offset_init as lb for projection
  // CHECK: scf.if {{.*}} -> (i1, index) {
  // CHECK: arith.index_cast {{.*}} : index to i32
  // CHECK: arith.muli {{.*}}, %c28_i32 : i32
  // CHECK: arith.addi {{.*}}, {{.*}} : i32
  // CHECK: memref.copy

  %result = scf.for %arg16 = %offset_init to %c65536_i32 step %c28_i32
    iter_args(%acc = %arg1) -> tensor<128x128xf32> : i32 {
    %iv_idx = arith.index_cast %arg16 : i32 to index
    %row_off = arith.muli %iv_idx, %c128 : index
    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %q_src_rc, %q_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %q_tensor = bufferization.to_tensor %q_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16> to tensor<128x128xf16>
    %qk = linalg.matmul {input_precision = "ieee"} ins(%q_tensor, %q_tensor : tensor<128x128xf16>, tensor<128x128xf16>) outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %qk : tensor<128x128xf32>
  }
  return %result : tensor<128x128xf32>
}

// ============================================================================
// TC-B06: Preserve original iter_args and extend with control args
// When the scf.for already has original iter_args (e.g., acc, lse), the
// 4 control iter_args (2 flags + prod + cons) are appended at the end.
// ============================================================================

// CHECK-LABEL: func.func @tc_b06_preserve_original_iter_args
func.func @tc_b06_preserve_original_iter_args(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %acc_init: tensor<128x128xf32>,
  %lse_init: tensor<128xf32>
) -> (tensor<128x128xf32>, tensor<128xf32>) {
  %c28_i32 = arith.constant 28 : i32
  %c65536_i32 = arith.constant 65536 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index

  // Original 2 iter_args preserved, 4 control args appended = 6 total
  // CHECK: scf.for {{.+}} iter_args({{.+}}) -> (tensor<128x128xf32>, tensor<128xf32>, i1, i1, index, index)

  %result:2 = scf.for %arg16 = %c0_i32 to %c65536_i32 step %c28_i32
    iter_args(%acc = %acc_init, %lse = %lse_init)
    -> (tensor<128x128xf32>, tensor<128xf32>) : i32 {
    %iv_idx = arith.index_cast %arg16 : i32 to index
    %row_off = arith.muli %iv_idx, %c128 : index
    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %q_src_rc, %q_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %q_tensor = bufferization.to_tensor %q_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16> to tensor<128x128xf16>
    %acc_new = linalg.matmul {input_precision = "ieee"} ins(%q_tensor, %q_tensor : tensor<128x128xf16>, tensor<128x128xf16>) outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %acc_new, %lse : tensor<128x128xf32>, tensor<128xf32>
  }
  return %result#0, %result#1 : tensor<128x128xf32>, tensor<128xf32>
}

// ============================================================================
// TC-B07: Dead DMA elimination
// Q load is marked (gm_load_bufferable) and gets multi-buffered.
// K load is unmarked and remains as normal alloc+copy in the consumer body.
// The Q load's memref.copy should be eliminated from the consumer body.
// ============================================================================

// CHECK-LABEL: func.func @tc_b07_dead_dma_elimination
func.func @tc_b07_dead_dma_elimination(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg1: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg2: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %c28_i32 = arith.constant 28 : i32
  %c65536_i32 = arith.constant 65536 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index

  // Q gets 2 buffer slots; K stays as normal alloc+copy in consumer
  // CHECK: %[[Q_SLOT0:.*]] = memref.alloc() : memref<128x128xf16>
  // CHECK: %[[Q_SLOT1:.*]] = memref.alloc() : memref<128x128xf16>

  // Consumer: Q selected via arith.select, then K loaded normally, then matmul
  // CHECK: arith.select
  // CHECK: memref.alloc() : memref<128x128xf16>
  // CHECK: memref.copy
  // CHECK: linalg.matmul

  %result = scf.for %arg16 = %c0_i32 to %c65536_i32 step %c28_i32
    iter_args(%acc = %arg2) -> tensor<128x128xf32> : i32 {
    %iv_idx = arith.index_cast %arg16 : i32 to index
    %row_off = arith.muli %iv_idx, %c128 : index

    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %q_src_rc, %q_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %q_tensor = bufferization.to_tensor %q_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16> to tensor<128x128xf16>

    %k_src_rc = memref.reinterpret_cast %arg1 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %k_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %k_src_rc, %k_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %k_tensor = bufferization.to_tensor %k_alloc restrict writable : memref<128x128xf16> to tensor<128x128xf16>

    %qk = linalg.matmul {input_precision = "ieee"} ins(%q_tensor, %k_tensor : tensor<128x128xf16>, tensor<128x128xf16>) outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %qk : tensor<128x128xf32>
  }
  return %result : tensor<128x128xf32>
}

// ============================================================================
// TC-B16: Producer flag state machine initialization
// Verifies that flag iter_args are initialized to false and counter
// iter_args are initialized to 0 (c0 index).
// ============================================================================

// CHECK-LABEL: func.func @tc_b16_flag_counter_init
func.func @tc_b16_flag_counter_init(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg1: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %c28_i32 = arith.constant 28 : i32
  %c65536_i32 = arith.constant 65536 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index
  %false = arith.constant false
  %c0 = arith.constant 0 : index

  // Flags init to %false, counters init to %c0
  // CHECK: %false = arith.constant false
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: scf.for {{.*}} iter_args({{.*}} = %false, {{.*}} = %false, {{.*}} = %[[C0]], {{.*}} = %[[C0]])

  %result = scf.for %arg16 = %c0_i32 to %c65536_i32 step %c28_i32
    iter_args(%acc = %arg1) -> tensor<128x128xf32> : i32 {
    %iv_idx = arith.index_cast %arg16 : i32 to index
    %row_off = arith.muli %iv_idx, %c128 : index
    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %q_src_rc, %q_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %q_tensor = bufferization.to_tensor %q_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16> to tensor<128x128xf16>
    %qk = linalg.matmul {input_precision = "ieee"} ins(%q_tensor, %q_tensor : tensor<128x128xf16>, tensor<128x128xf16>) outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %qk : tensor<128x128xf32>
  }
  return %result : tensor<128x128xf32>
}

// ============================================================================
// TC-B17: Producer condition check structure
// Verifies the Producer's scf.if condition: flag_empty AND has_more_trips.
// ============================================================================

// CHECK-LABEL: func.func @tc_b17_producer_condition_structure
func.func @tc_b17_producer_condition_structure(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg1: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %c28_i32 = arith.constant 28 : i32
  %c65536_i32 = arith.constant 65536 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index

  // Producer checks: flag == false (empty) AND prod_counter < trip_count
  // CHECK: arith.cmpi eq, {{.*}}, %false
  // CHECK: arith.cmpi ult, {{.*}}, {{.*}}
  // CHECK: arith.andi

  %result = scf.for %arg16 = %c0_i32 to %c65536_i32 step %c28_i32
    iter_args(%acc = %arg1) -> tensor<128x128xf32> : i32 {
    %iv_idx = arith.index_cast %arg16 : i32 to index
    %row_off = arith.muli %iv_idx, %c128 : index
    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %q_src_rc, %q_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %q_tensor = bufferization.to_tensor %q_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16> to tensor<128x128xf16>
    %qk = linalg.matmul {input_precision = "ieee"} ins(%q_tensor, %q_tensor : tensor<128x128xf16>, tensor<128x128xf16>) outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %qk : tensor<128x128xf32>
  }
  return %result : tensor<128x128xf32>
}
