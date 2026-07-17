// RUN: triton-opt --gm-load-multi-buffer %s | FileCheck %s

// ============================================================================
// TC-B10: Multiple loads in same loop - each gets independent state
// When multiple loads have gm_load_bufferable in the same scf.for,
// each load group gets its own buffer slots, flags, and counters.
// ============================================================================

// CHECK-LABEL: func.func @tc_b10_multiple_loads_same_loop
func.func @tc_b10_multiple_loads_same_loop(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg1: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg2: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %c28_i32 = arith.constant 28 : i32
  %c65536_i32 = arith.constant 65536 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index

  // Each load gets 2 slots = 4 allocs total before the loop
  // CHECK: memref.alloc() : memref<128x128xf16>
  // CHECK: memref.alloc() : memref<128x128xf16>
  // CHECK: memref.alloc() : memref<128x128xf16>
  // CHECK: memref.alloc() : memref<128x128xf16>

  // Each load gets independent control: 2*(2 flags + prod + cons) = 8 control iter_args
  // CHECK: scf.for {{.+}} iter_args({{.+}}) -> (tensor<128x128xf32>, i1, i1, index, index, i1, i1, index, index)

  // Two separate select chains (one per load group)
  // CHECK: arith.select
  // CHECK: arith.select
  // CHECK: linalg.matmul

  %result = scf.for %arg16 = %c0_i32 to %c65536_i32 step %c28_i32
    iter_args(%acc = %arg2) -> tensor<128x128xf32> : i32 {
    %iv_idx = arith.index_cast %arg16 : i32 to index
    %row_off = arith.muli %iv_idx, %c128 : index

    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %q_src_rc, %q_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %q_tensor = bufferization.to_tensor %q_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16>

    %k_src_rc = memref.reinterpret_cast %arg1 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %k_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %k_src_rc, %k_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %k_tensor = bufferization.to_tensor %k_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16>

    %qk = linalg.matmul {input_precision = "ieee"} ins(%q_tensor, %k_tensor : tensor<128x128xf16>, tensor<128x128xf16>) outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %qk : tensor<128x128xf32>
  }
  return %result : tensor<128x128xf32>
}

// ============================================================================
// TC-B11: Nested loops - inner loop transformed first
// When only the inner loop has a marked load, buffer slots are allocated
// inside the outer loop body (before the inner loop).
// ============================================================================

// CHECK-LABEL: func.func @tc_b11_nested_loops_inner_first
func.func @tc_b11_nested_loops_inner_first(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg1: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %c28_i32 = arith.constant 28 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c8192_i32 = arith.constant 8192 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index

  // Outer loop (no marked load in outer body)
  // CHECK: scf.for {{.*}} : i32 {

  // Inner loop has marked load - buffer slots allocated inside outer body
  // CHECK: memref.alloc() : memref<128x128xf16>
  // CHECK: memref.alloc() : memref<128x128xf16>
  // CHECK: scf.for {{.+}} iter_args({{.+}}) -> (tensor<128x128xf32>, i1, i1, index, index)

  %outer_result = scf.for %outer_iv = %c0_i32 to %c1024_i32 step %c28_i32
    iter_args(%outer_acc = %arg1) -> tensor<128x128xf32> : i32 {

    %inner_result = scf.for %inner_iv = %c0_i32 to %c8192_i32 step %c128_i32
      iter_args(%inner_acc = %outer_acc) -> tensor<128x128xf32> : i32 {
      %iv_idx = arith.index_cast %inner_iv : i32 to index
      %row_off = arith.muli %iv_idx, %c128 : index
      %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
      %q_alloc = memref.alloc () : memref<128x128xf16>
      memref.copy %q_src_rc, %q_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
      %q_tensor = bufferization.to_tensor %q_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16>
      %qk = linalg.matmul {input_precision = "ieee"} ins(%q_tensor, %q_tensor : tensor<128x128xf16>, tensor<128x128xf16>) outs(%inner_acc : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield %qk : tensor<128x128xf32>
    }
    scf.yield %inner_result : tensor<128x128xf32>
  }
  return %outer_result : tensor<128x128xf32>
}

// ============================================================================
// TC-B12: Linear iter_arg delta projection
// When the loop has an iter_arg that follows a linear delta pattern
// (iter_arg_yield = addi(iter_arg, loop_invariant)), the pass should
// recognize it and correctly project the address calculation.
// ============================================================================

// CHECK-LABEL: func.func @tc_b12_linear_iter_arg_delta
func.func @tc_b12_linear_iter_arg_delta(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %offset_init: i32,
  %arg1: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %c28_i32 = arith.constant 28 : i32
  %c65536_i32 = arith.constant 65536 : i32
  %c128 = arith.constant 128 : index
  %c128_i32 = arith.constant 128 : i32

  // iter_arg with linear delta gets multi-buffer + projected delta
  // CHECK: scf.for {{.+}} iter_args({{.+}}) -> (i32, i1, i1, index, index)

  %result = scf.for %iv = %offset_init to %c65536_i32 step %c28_i32
    iter_args(%offset = %offset_init) -> i32 : i32 {
    %offset_idx = arith.index_cast %offset : i32 to index
    %row_off = arith.muli %offset_idx, %c128 : index
    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %q_src_rc, %q_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %q_tensor = bufferization.to_tensor %q_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16>
    %acc_dummy = tensor.empty() : tensor<128x128xf32>
    %qk = linalg.matmul {input_precision = "ieee"} ins(%q_tensor, %q_tensor : tensor<128x128xf16>, tensor<128x128xf16>) outs(%acc_dummy : tensor<128x128xf32>) -> tensor<128x128xf32>

    // Linear delta: offset_new = offset + 128
    %offset_new = arith.addi %offset, %c128_i32 : i32
    scf.yield %offset_new : i32
  }

  // Use the result to prevent dead code elimination
  %final_idx = arith.index_cast %result : i32 to index
  %final_row = arith.muli %final_idx, %c128 : index
  %final_view = memref.reinterpret_cast %arg0 to offset: [%final_row], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
  %final_alloc = memref.alloc() : memref<128x128xf16>
  memref.copy %final_view, %final_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
  %final_tensor = bufferization.to_tensor %final_alloc restrict writable : memref<128x128xf16>
  %ext = arith.extf %final_tensor : tensor<128x128xf16> to tensor<128x128xf32>
  %sum = arith.addf %ext, %arg1 : tensor<128x128xf32>
  return %sum : tensor<128x128xf32>
}

// ============================================================================
// TC-B18: Multiple marked loads + original iter_args combined
// Combines TC-B06 and TC-B10: multiple marked loads in a loop that already
// has original iter_args. Verifies iter_args layout:
// [orig_args...] + [control_group_1...] + [control_group_2...]
// ============================================================================

// CHECK-LABEL: func.func @tc_b18_multi_load_with_orig_iter_args
func.func @tc_b18_multi_load_with_orig_iter_args(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg1: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %acc_init: tensor<128x128xf32>,
  %lse_init: tensor<128xf32>
) -> (tensor<128x128xf32>, tensor<128xf32>) {
  %c28_i32 = arith.constant 28 : i32
  %c65536_i32 = arith.constant 65536 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index

  // 4 allocs total: 2 slots per load group
  // CHECK: memref.alloc() : memref<128x128xf16>
  // CHECK: memref.alloc() : memref<128x128xf16>
  // CHECK: memref.alloc() : memref<128x128xf16>
  // CHECK: memref.alloc() : memref<128x128xf16>

  // iter_args: 2 orig + 4*2 control = 10 total
  // CHECK: scf.for {{.+}} iter_args({{.+}}) -> (tensor<128x128xf32>, tensor<128xf32>, i1, i1, index, index, i1, i1, index, index)

  %result:2 = scf.for %arg16 = %c0_i32 to %c65536_i32 step %c28_i32
    iter_args(%acc = %acc_init, %lse = %lse_init)
    -> (tensor<128x128xf32>, tensor<128xf32>) : i32 {
    %iv_idx = arith.index_cast %arg16 : i32 to index
    %row_off = arith.muli %iv_idx, %c128 : index

    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %q_src_rc, %q_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %q_tensor = bufferization.to_tensor %q_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16>

    %k_src_rc = memref.reinterpret_cast %arg1 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %k_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %k_src_rc, %k_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %k_tensor = bufferization.to_tensor %k_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16>

    %acc_new = linalg.matmul {input_precision = "ieee"} ins(%q_tensor, %k_tensor : tensor<128x128xf16>, tensor<128x128xf16>) outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %acc_new, %lse : tensor<128x128xf32>, tensor<128xf32>
  }
  return %result#0, %result#1 : tensor<128x128xf32>, tensor<128xf32>
}

// ============================================================================
// TC-B19: Different data types (bf16)
// Verifies the pass handles bf16 tensors correctly, not just f16.
// ============================================================================

// CHECK-LABEL: func.func @tc_b19_bf16_data_type
func.func @tc_b19_bf16_data_type(
  %arg0: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg1: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %c28_i32 = arith.constant 28 : i32
  %c65536_i32 = arith.constant 65536 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index

  // bf16 buffer slots
  // CHECK: %[[S0:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK: %[[S1:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK: arith.remui {{.*}}, %c2

  %result = scf.for %arg16 = %c0_i32 to %c65536_i32 step %c28_i32
    iter_args(%acc = %arg1) -> tensor<128x128xf32> : i32 {
    %iv_idx = arith.index_cast %arg16 : i32 to index
    %row_off = arith.muli %iv_idx, %c128 : index
    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[128, 1], offset: ?>>
    %q_alloc = memref.alloc () : memref<128x128xbf16>
    memref.copy %q_src_rc, %q_alloc : memref<128x128xbf16, strided<[128, 1], offset: ?>> to memref<128x128xbf16>
    %q_tensor = bufferization.to_tensor %q_alloc restrict writable {gm_load_bufferable} : memref<128x128xbf16>
    %qk = linalg.matmul {input_precision = "ieee"} ins(%q_tensor, %q_tensor : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %qk : tensor<128x128xf32>
  }
  return %result : tensor<128x128xf32>
}

// ============================================================================
// TC-B20: Nested loops with marked load in outer loop only
// When only the outer loop has a marked load (and the inner loop has none),
// the outer loop gets transformed while the inner loop stays unchanged.
// ============================================================================

// CHECK-LABEL: func.func @tc_b20_outer_loop_marked_only
func.func @tc_b20_outer_loop_marked_only(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg1: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %c28_i32 = arith.constant 28 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index

  // Outer loop has marked load -> gets buffer slots before outer loop
  // CHECK: %[[S0:.*]] = memref.alloc() : memref<128x128xf16>
  // CHECK: %[[S1:.*]] = memref.alloc() : memref<128x128xf16>
  // CHECK: scf.for {{.+}} iter_args({{.+}}) -> (tensor<128x128xf32>, i1, i1, index, index)

  // Inner loop has no marked load -> stays as-is
  // CHECK: scf.for {{.*}} : i32 {

  %outer_result = scf.for %outer_iv = %c0_i32 to %c1024_i32 step %c28_i32
    iter_args(%outer_acc = %arg1) -> tensor<128x128xf32> : i32 {
    %iv_idx = arith.index_cast %outer_iv : i32 to index
    %row_off = arith.muli %iv_idx, %c128 : index
    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %q_src_rc, %q_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %q_tensor = bufferization.to_tensor %q_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16>
    %qk = linalg.matmul {input_precision = "ieee"} ins(%q_tensor, %q_tensor : tensor<128x128xf16>, tensor<128x128xf16>) outs(%outer_acc : tensor<128x128xf32>) -> tensor<128x128xf32>

    // Inner loop with no marked load
    %inner_result = scf.for %inner_iv = %c0_i32 to %c4_i32 step %c1_i32
      iter_args(%inner_acc = %qk) -> tensor<128x128xf32> : i32 {
      %add = arith.addf %inner_acc, %inner_acc : tensor<128x128xf32>
      scf.yield %add : tensor<128x128xf32>
    }
    scf.yield %inner_result : tensor<128x128xf32>
  }
  return %outer_result : tensor<128x128xf32>
}

// ============================================================================
// TC-B21: Index-typed IV (not i32)
// Verifies the pass handles scf.for with index-typed induction variable
// correctly, where the address calculation chain uses index arithmetic.
// ============================================================================

// CHECK-LABEL: func.func @tc_b21_index_iv_type
func.func @tc_b21_index_iv_type(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg1: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %c0 = arith.constant 0 : index
  %c2344 = arith.constant 2344 : index
  %c28 = arith.constant 28 : index
  %c128 = arith.constant 128 : index

  // Index-typed IV: address chain uses index arithmetic directly
  // CHECK: %[[S0:.*]] = memref.alloc() : memref<128x128xf16>
  // CHECK: %[[S1:.*]] = memref.alloc() : memref<128x128xf16>
  // CHECK: arith.remui {{.*}}, %c2

  %result = scf.for %iv = %c0 to %c2344 step %c28
    iter_args(%acc = %arg1) -> tensor<128x128xf32> : index {
    %row_off = arith.muli %iv, %c128 : index
    %q_src_rc = memref.reinterpret_cast %arg0 to offset: [%row_off], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_alloc = memref.alloc () : memref<128x128xf16>
    memref.copy %q_src_rc, %q_alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %q_tensor = bufferization.to_tensor %q_alloc restrict writable {gm_load_bufferable} : memref<128x128xf16>
    %qk = linalg.matmul {input_precision = "ieee"} ins(%q_tensor, %q_tensor : tensor<128x128xf16>, tensor<128x128xf16>) outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %qk : tensor<128x128xf32>
  }
  return %result : tensor<128x128xf32>
}
