// RUN: triton-opt --split-input-file --mark-vload-multi-buffer %s | FileCheck %s --check-prefix=MARK
// RUN: triton-opt --split-input-file --separate-memory-from-compute %s | FileCheck %s --check-prefix=DEPTH

// MARK-LABEL: func.func @qkv_semantics_only_marks_direct_matmul_rhs
func.func @qkv_semantics_only_marks_direct_matmul_rhs(
  %v: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %q: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %k: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}) {
  %c0 = arith.constant 0 : index

  // MARK: %[[Q_ALLOC:.*]] = memref.alloc() : memref<128x128xbf16>
  // MARK-NEXT: memref.copy {{.*}}, %[[Q_ALLOC]]
  // MARK: %[[K_ALLOC:.*]] = memref.alloc() : memref<128x128xbf16>
  // MARK-NEXT: memref.copy {{.*}}, %[[K_ALLOC]]
  // MARK: %[[V_ALLOC:.*]] = memref.alloc() : memref<128x128xbf16>
  // MARK-NEXT: annotation.mark %[[V_ALLOC]] {hivm.multi_buffer = 3 : i32} : memref<128x128xbf16>
  // MARK-NEXT: memref.copy {{.*}}, %[[V_ALLOC]]

  %q_src = memref.reinterpret_cast %q to offset: [%c0], sizes: [128, 128], strides: [128, 1] :
    memref<?xbf16> to memref<128x128xbf16, strided<[128, 1], offset: ?>>
  %q_alloc = memref.alloc() : memref<128x128xbf16>
  memref.copy %q_src, %q_alloc : memref<128x128xbf16, strided<[128, 1], offset: ?>> to memref<128x128xbf16>
  %q_tensor = bufferization.to_tensor %q_alloc restrict writable : memref<128x128xbf16>

  %k_src = memref.reinterpret_cast %k to offset: [%c0], sizes: [128, 128], strides: [128, 1] :
    memref<?xbf16> to memref<128x128xbf16, strided<[128, 1], offset: ?>>
  %k_alloc = memref.alloc() : memref<128x128xbf16>
  memref.copy %k_src, %k_alloc : memref<128x128xbf16, strided<[128, 1], offset: ?>> to memref<128x128xbf16>
  %k_tensor = bufferization.to_tensor %k_alloc restrict writable : memref<128x128xbf16>
  %k_empty = tensor.empty() : tensor<128x128xbf16>
  %k_transposed = linalg.transpose ins(%k_tensor : tensor<128x128xbf16>) outs(%k_empty : tensor<128x128xbf16>) permutation = [1, 0]
  %qk_acc = tensor.empty() : tensor<128x128xf32>
  %qk = linalg.matmul {input_precision = "ieee"} ins(%q_tensor, %k_transposed : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%qk_acc : tensor<128x128xf32>) -> tensor<128x128xf32>

  %v_src = memref.reinterpret_cast %v to offset: [%c0], sizes: [128, 128], strides: [128, 1] :
    memref<?xbf16> to memref<128x128xbf16, strided<[128, 1], offset: ?>>
  %v_alloc = memref.alloc() : memref<128x128xbf16>
  memref.copy %v_src, %v_alloc : memref<128x128xbf16, strided<[128, 1], offset: ?>> to memref<128x128xbf16>
  %v_tensor = bufferization.to_tensor %v_alloc restrict writable : memref<128x128xbf16>
  %out_acc = tensor.empty() : tensor<128x128xf32>
  %out = linalg.matmul {input_precision = "ieee"} ins(%q_tensor, %v_tensor : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%out_acc : tensor<128x128xf32>) -> tensor<128x128xf32>
  return
}

// -----

// MARK-LABEL: func.func @marks_direct_matmul_rhs_for_generic_shape_and_type
func.func @marks_direct_matmul_rhs_for_generic_shape_and_type(
  %lhs: tensor<16x64xf16>,
  %v: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}) {
  %c0 = arith.constant 0 : index

  // MARK: %[[V_ALLOC:.*]] = memref.alloc() : memref<64x32xf16>
  // MARK-NEXT: annotation.mark %[[V_ALLOC]] {hivm.multi_buffer = 3 : i32} : memref<64x32xf16>

  %v_src = memref.reinterpret_cast %v to offset: [%c0], sizes: [64, 32], strides: [32, 1] :
    memref<?xf16> to memref<64x32xf16, strided<[32, 1], offset: ?>>
  %v_alloc = memref.alloc() : memref<64x32xf16>
  memref.copy %v_src, %v_alloc : memref<64x32xf16, strided<[32, 1], offset: ?>> to memref<64x32xf16>
  %v_tensor = bufferization.to_tensor %v_alloc restrict writable : memref<64x32xf16>
  %acc = tensor.empty() : tensor<16x32xf32>
  %out = linalg.matmul {input_precision = "ieee"} ins(%lhs, %v_tensor : tensor<16x64xf16>, tensor<64x32xf16>) outs(%acc : tensor<16x32xf32>) -> tensor<16x32xf32>
  return
}

// -----

// MARK-LABEL: func.func @ignore_rhs_matmul_without_gm_copy
func.func @ignore_rhs_matmul_without_gm_copy(%lhs: tensor<16x16xf16>) {
  // MARK: memref.alloc() : memref<16x16xf16>
  // MARK-NEXT: bufferization.to_tensor
  // MARK-NOT: annotation.mark
  // MARK: return

  %alloc = memref.alloc() : memref<16x16xf16>
  %rhs = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
  %acc = tensor.empty() : tensor<16x16xf32>
  %out = linalg.matmul {input_precision = "ieee"} ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x16xf16>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
  return
}

// -----

// MARK-LABEL: func.func @ignore_non_gm_copy_source
func.func @ignore_non_gm_copy_source(
  %lhs: tensor<16x16xf16>,
  %src: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}) {
  %c0 = arith.constant 0 : index

  // MARK: memref.alloc() : memref<16x16xf16>
  // MARK-NEXT: memref.copy
  // MARK-NOT: annotation.mark
  // MARK: return

  %view = memref.reinterpret_cast %src to offset: [%c0], sizes: [16, 16], strides: [16, 1] :
    memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
  %alloc = memref.alloc() : memref<16x16xf16>
  memref.copy %view, %alloc : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<16x16xf16>
  %rhs = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
  %acc = tensor.empty() : tensor<16x16xf32>
  %out = linalg.matmul {input_precision = "ieee"} ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x16xf16>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
  return
}

// -----

module attributes {ssbuffer.load_store_buf_count = 3 : i32} {
  // DEPTH-LABEL: func.func @depth_3_marks_direct_matmul_rhs
  func.func @depth_3_marks_direct_matmul_rhs(
    %lhs: tensor<16x16xf16>,
    %v: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}) {
    %c0 = arith.constant 0 : index

    // DEPTH: %[[V_ALLOC:.*]] = memref.alloc() : memref<16x16xf16>
    // DEPTH-NEXT: annotation.mark %[[V_ALLOC]] {hivm.multi_buffer = 3 : i32} : memref<16x16xf16>

    %view = memref.reinterpret_cast %v to offset: [%c0], sizes: [16, 16], strides: [16, 1] :
      memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
    %alloc = memref.alloc() : memref<16x16xf16>
    memref.copy %view, %alloc : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<16x16xf16>
    %rhs = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %acc = tensor.empty() : tensor<16x16xf32>
    %out = linalg.matmul {input_precision = "ieee"} ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x16xf16>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    return
  }
}

// -----

module attributes {ssbuffer.load_store_buf_count = 2 : i32} {
  // DEPTH-LABEL: func.func @depth_2_does_not_run_vload_marker
  func.func @depth_2_does_not_run_vload_marker(
    %lhs: tensor<16x16xf16>,
    %v: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}) {
    %c0 = arith.constant 0 : index

    // DEPTH: memref.alloc() : memref<16x16xf16>
    // DEPTH-NOT: annotation.mark
    // DEPTH-NOT: gm_load_bufferable
    // DEPTH: return

    %view = memref.reinterpret_cast %v to offset: [%c0], sizes: [16, 16], strides: [16, 1] :
      memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
    %alloc = memref.alloc() : memref<16x16xf16>
    memref.copy %view, %alloc : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<16x16xf16>
    %rhs = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %acc = tensor.empty() : tensor<16x16xf32>
    %out = linalg.matmul {input_precision = "ieee"} ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x16xf16>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    return
  }
}

// -----

module attributes {ssbuffer.load_store_buf_count = 4 : i32} {
  // DEPTH-LABEL: func.func @depth_4_does_not_run_vload_marker
  func.func @depth_4_does_not_run_vload_marker(
    %lhs: tensor<16x16xf16>,
    %v: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}) {
    %c0 = arith.constant 0 : index

    // DEPTH: memref.alloc() : memref<16x16xf16>
    // DEPTH-NOT: annotation.mark
    // DEPTH-NOT: gm_load_bufferable
    // DEPTH: return

    %view = memref.reinterpret_cast %v to offset: [%c0], sizes: [16, 16], strides: [16, 1] :
      memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
    %alloc = memref.alloc() : memref<16x16xf16>
    memref.copy %view, %alloc : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<16x16xf16>
    %rhs = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %acc = tensor.empty() : tensor<16x16xf32>
    %out = linalg.matmul {input_precision = "ieee"} ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x16xf16>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    return
  }
}
