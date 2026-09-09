// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=REJECT
// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=3" 2>/dev/null | FileCheck %s --check-prefix=BAD-UNROLL
// RUN: triton-opt %s --debug-only=cv-split-scheduling "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>&1 >/dev/null | FileCheck %s --check-prefix=ACCEPT
// RUN: triton-opt %s --debug-only=cv-split-scheduling "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG
// RUN: triton-opt %s --debug-only=cv-split-scheduling "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" 2>&1 >/dev/null | FileCheck %s --check-prefix=FACTOR2
// RUN: triton-opt %s --debug-only=cv-split-scheduling "--cv_split_scheduling=compile-on-910-95=true unroll-factor=8" 2>&1 >/dev/null | FileCheck %s --check-prefix=FACTOR8

// Pre-check rejection must happen before unrolling or scope construction.
// REJECT-NOT: scope.scope
// REJECT-LABEL: func.func @no_loop
// REJECT-NEXT: return
// DIAG: [cv-split] Function: no_loop
// DIAG-NEXT: [cv-split] Pre-check rejected function, skip
func.func @no_loop() {
  return
}

// REJECT-LABEL: func.func @loop_without_matmul
// REJECT: scf.for
// REJECT-NOT: scope.scope
// DIAG: [cv-split] Function: loop_without_matmul
// DIAG-NEXT: [cv-split] Pre-check rejected function, skip
func.func @loop_without_matmul() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %iv = %c0 to %c16 step %c1 {
    arith.addi %iv, %c1 : index
  }
  return
}

// ACCEPT: [cv-split] Function: single_loop_without_store
// ACCEPT-NEXT: [cv-split] Pre-check accepted candidate loop
// FACTOR2: [cv-split] Function: single_loop_without_store
// FACTOR2-NEXT: [cv-split] Pre-check accepted candidate loop
// FACTOR8: [cv-split] Function: single_loop_without_store
// FACTOR8-NEXT: [cv-split] Pre-check accepted candidate loop
func.func @single_loop_without_store() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %lhs = tensor.empty() : tensor<32x16xf16>
  %rhs = tensor.empty() : tensor<16x16xf16>
  %mat_init = tensor.empty() : tensor<32x16xf32>
  scf.for %iv = %c0 to %c16 step %c1 {
    %matmul = linalg.matmul ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
        outs(%mat_init : tensor<32x16xf32>) -> tensor<32x16xf32>
    arith.addi %iv, %c1 : index
  }
  return
}

// An enclosing loop is supported when it contains exactly one innermost
// candidate loop.
// ACCEPT: [cv-split] Function: one_nested_candidate
// ACCEPT-NEXT: [cv-split] Pre-check accepted candidate loop
func.func @one_nested_candidate(%lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %lhs = tensor.empty() : tensor<32x16xf16>
  %rhs = tensor.empty() : tensor<16x16xf16>
  %mat_init = tensor.empty() : tensor<32x16xf32>
  scf.for %outer = %lb to %ub step %step {
    scf.for %inner = %c0 to %c16 step %c1 {
      %matmul = linalg.matmul ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%mat_init : tensor<32x16xf32>) -> tensor<32x16xf32>
      arith.addi %inner, %c1 : index
    }
  }
  return
}

// A store in the enclosing loop is outside the selected inner candidate and
// must not make pre-check reject the function.
// ACCEPT: [cv-split] Function: outer_store_after_inner_candidate
// ACCEPT-NEXT: [cv-split] Pre-check accepted candidate loop
func.func @outer_store_after_inner_candidate(
    %lb: index, %ub: index, %step: index, %value: i32,
    %dst: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %lhs = tensor.empty() : tensor<32x16xf16>
  %rhs = tensor.empty() : tensor<16x16xf16>
  %mat_init = tensor.empty() : tensor<32x16xf32>
  scf.for %outer = %lb to %ub step %step {
    scf.for %inner = %c0 to %c16 step %c1 {
      %matmul = linalg.matmul ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%mat_init : tensor<32x16xf32>) -> tensor<32x16xf32>
      arith.addi %inner, %c1 : index
    }
    memref.store %value, %dst[%outer] : memref<?xi32>
  }
  return
}

// REJECT-LABEL: func.func @multiple_innermost_loops
// REJECT: scf.for
// REJECT: scf.for
// DIAG: [cv-split] Function: multiple_innermost_loops
// DIAG-NEXT: [cv-split] Pre-check rejected function, skip
func.func @multiple_innermost_loops(%lb: index, %ub: index, %step: index) {
  scf.for %first = %lb to %ub step %step {
  }
  scf.for %second = %lb to %ub step %step {
  }
  return
}

// REJECT-LABEL: func.func @store_in_candidate
// REJECT: scf.for
// REJECT: memref.store
// BAD-UNROLL-LABEL: func.func @store_in_candidate
// BAD-UNROLL: scf.for
// BAD-UNROLL-NOT: scope.scope
// DIAG: [cv-split] Function: store_in_candidate
// DIAG-NEXT: [cv-split] Pre-check rejected function, skip
func.func @store_in_candidate(%value: i32, %dst: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %iv = %c0 to %c16 step %c1 {
    memref.store %value, %dst[%c0] : memref<1xi32>
  }
  return
}

// A static tensor defined outside the candidate may be captured directly by
// operations inside it, matching the loop-invariant Q tile in FA.
// ACCEPT: [cv-split] Function: static_external_tensor
// ACCEPT-NEXT: [cv-split] Pre-check accepted candidate loop
func.func @static_external_tensor() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %tile = tensor.empty() : tensor<4x8xf32>
  %lhs = tensor.empty() : tensor<32x16xf16>
  %rhs = tensor.empty() : tensor<16x16xf16>
  %mat_init = tensor.empty() : tensor<32x16xf32>
  scf.for %iv = %c0 to %c16 step %c1 {
    %matmul = linalg.matmul ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
        outs(%mat_init : tensor<32x16xf32>) -> tensor<32x16xf32>
    %sum = arith.addf %tile, %tile : tensor<4x8xf32>
  }
  return
}

// Static tensor init arguments, body block arguments, yields, and loop results
// are all supported.
// ACCEPT: [cv-split] Function: static_tensor_iter_args
// ACCEPT-NEXT: [cv-split] Pre-check accepted candidate loop
func.func @static_tensor_iter_args() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %init = tensor.empty() : tensor<4x8xf32>
  %lhs = tensor.empty() : tensor<32x16xf16>
  %rhs = tensor.empty() : tensor<16x16xf16>
  %mat_init = tensor.empty() : tensor<32x16xf32>
  %result = scf.for %iv = %c0 to %c16 step %c1
      iter_args(%arg = %init) -> tensor<4x8xf32> {
    %matmul = linalg.matmul ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
        outs(%mat_init : tensor<32x16xf32>) -> tensor<32x16xf32>
    %next = arith.addf %arg, %arg : tensor<4x8xf32>
    scf.yield %next : tensor<4x8xf32>
  }
  return
}

// Copies into loop-local staging allocations are not output stores and remain
// supported.
// ACCEPT: [cv-split] Function: local_memref_copy
// ACCEPT-NEXT: [cv-split] Pre-check accepted candidate loop
func.func @local_memref_copy(%src: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %lhs = tensor.empty() : tensor<32x16xf16>
  %rhs = tensor.empty() : tensor<16x16xf16>
  %mat_init = tensor.empty() : tensor<32x16xf32>
  scf.for %iv = %c0 to %c16 step %c1 {
    %matmul = linalg.matmul ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
        outs(%mat_init : tensor<32x16xf32>) -> tensor<32x16xf32>
    %dst = memref.alloc() : memref<8xf32>
    memref.copy %src, %dst : memref<8xf32> to memref<8xf32>
  }
  return
}

// A nested computation region such as linalg.reduce is not control-flow
// branching and remains supported.
// ACCEPT: [cv-split] Function: linalg_reduce_region
// ACCEPT-NEXT: [cv-split] Pre-check accepted candidate loop
func.func @linalg_reduce_region() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %input = tensor.empty() : tensor<4x8xf32>
  %init = tensor.empty() : tensor<4xf32>
  %lhs = tensor.empty() : tensor<32x16xf16>
  %rhs = tensor.empty() : tensor<16x16xf16>
  %mat_init = tensor.empty() : tensor<32x16xf32>
  scf.for %iv = %c0 to %c16 step %c1 {
    %matmul = linalg.matmul ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
        outs(%mat_init : tensor<32x16xf32>) -> tensor<32x16xf32>
    %reduced = linalg.reduce ins(%input : tensor<4x8xf32>)
        outs(%init : tensor<4xf32>) dimensions = [1]
        (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  }
  return
}

// REJECT-LABEL: func.func @dynamic_upper_bound
// REJECT: scf.for
// DIAG: [cv-split] Function: dynamic_upper_bound
// DIAG-NEXT: [cv-split] Pre-check rejected function, skip
func.func @dynamic_upper_bound(%ub: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %ub step %c1 {
    arith.addi %iv, %c1 : index
  }
  return
}

// Ten iterations leave a factor-4 remainder.
// REJECT-LABEL: func.func @non_divisible_trip_count
// REJECT: scf.for
// DIAG: [cv-split] Function: non_divisible_trip_count
// DIAG-NEXT: [cv-split] Pre-check rejected function, skip
func.func @non_divisible_trip_count() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.for %iv = %c0 to %c10 step %c1 {
    arith.addi %iv, %c1 : index
  }
  return
}

// Three iterations are fewer than the factor-4 unroll width.
// REJECT-LABEL: func.func @trip_count_smaller_than_factor
// REJECT: scf.for
// DIAG: [cv-split] Function: trip_count_smaller_than_factor
// DIAG-NEXT: [cv-split] Pre-check rejected function, skip
func.func @trip_count_smaller_than_factor() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %iv = %c0 to %c3 step %c1 {
    arith.addi %iv, %c1 : index
  }
  return
}

// REJECT-LABEL: func.func @dynamic_ranked_tensor
// REJECT: tensor<?x8xf32>
// DIAG: [cv-split] Function: dynamic_ranked_tensor
// DIAG-NEXT: [cv-split] Pre-check rejected function, skip
func.func @dynamic_ranked_tensor(%tile: tensor<?x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %iv = %c0 to %c16 step %c1 {
    %dim = tensor.dim %tile, %c0 : tensor<?x8xf32>
  }
  return
}

// REJECT-LABEL: func.func @unranked_tensor
// REJECT: tensor<*xf32>
// DIAG: [cv-split] Function: unranked_tensor
// DIAG-NEXT: [cv-split] Pre-check rejected function, skip
func.func @unranked_tensor(%tile: tensor<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %iv = %c0 to %c16 step %c1 {
    %dim = tensor.dim %tile, %c0 : tensor<*xf32>
  }
  return
}

// REJECT-LABEL: func.func @branching_inside_candidate
// REJECT: scf.if
// DIAG: [cv-split] Function: branching_inside_candidate
// DIAG-NEXT: [cv-split] Pre-check rejected function, skip
func.func @branching_inside_candidate(%condition: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %iv = %c0 to %c16 step %c1 {
    scf.if %condition {
      arith.addi %iv, %c1 : index
    }
  }
  return
}

// REJECT-LABEL: func.func @while_inside_candidate
// REJECT: scf.while
// DIAG: [cv-split] Function: while_inside_candidate
// DIAG-NEXT: [cv-split] Pre-check rejected function, skip
func.func @while_inside_candidate() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %iv = %c0 to %c16 step %c1 {
    %result = scf.while (%arg = %c0) : (index) -> index {
      %condition = arith.cmpi slt, %arg, %c1 : index
      scf.condition(%condition) %arg : index
    } do {
    ^bb0(%arg : index):
      %next = arith.addi %arg, %c1 : index
      scf.yield %next : index
    }
  }
  return
}

// REJECT-LABEL: func.func @insert_slice_inside_candidate
// REJECT: tensor.insert_slice
// DIAG: [cv-split] Function: insert_slice_inside_candidate
// DIAG-NEXT: [cv-split] Pre-check rejected function, skip
func.func @insert_slice_inside_candidate() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %src = tensor.empty() : tensor<1x8xf32>
  %dst = tensor.empty() : tensor<4x8xf32>
  scf.for %iv = %c0 to %c16 step %c1 {
    %updated = tensor.insert_slice %src into %dst[0, 0] [1, 8] [1, 1]
        : tensor<1x8xf32> into tensor<4x8xf32>
  }
  return
}

// REJECT-LABEL: func.func @materialize_inside_candidate
// REJECT: bufferization.materialize_in_destination
// DIAG: [cv-split] Function: materialize_inside_candidate
// DIAG-NEXT: [cv-split] Pre-check rejected function, skip
func.func @materialize_inside_candidate(%dst: memref<4x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %src = tensor.empty() : tensor<4x8xf32>
  scf.for %iv = %c0 to %c16 step %c1 {
    bufferization.materialize_in_destination %src in writable %dst
        : (tensor<4x8xf32>, memref<4x8xf32>) -> ()
  }
  return
}

// Buffer-semantics matmul is unsupported because PV unfusing constructs tensor
// results and must be rejected before the candidate loop is mutated.
// REJECT-LABEL: func.func @memref_matmul_out
// REJECT: linalg.matmul
// DIAG: [cv-split] Function: memref_matmul_out
// DIAG-NEXT: [cv-split] Pre-check rejected function, skip
func.func @memref_matmul_out(
    %lhs: memref<4x4xf32>, %rhs: memref<4x4xf32>,
    %out: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %iv = %c0 to %c16 step %c1 {
    linalg.matmul ins(%lhs, %rhs : memref<4x4xf32>, memref<4x4xf32>)
        outs(%out : memref<4x4xf32>)
  }
  return
}

// Tensor-semantics matmul with a ranked tensor destination is supported.
// ACCEPT: [cv-split] Function: ranked_tensor_matmul_out
// ACCEPT-NEXT: [cv-split] Pre-check accepted candidate loop
func.func @ranked_tensor_matmul_out() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %lhs = tensor.empty() : tensor<32x16xf32>
  %rhs = tensor.empty() : tensor<16x16xf32>
  %out = tensor.empty() : tensor<32x16xf32>
  scf.for %iv = %c0 to %c16 step %c1 {
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<32x16xf32>, tensor<16x16xf32>)
        outs(%out : tensor<32x16xf32>) -> tensor<32x16xf32>
  }
  return
}

// ROW_SPLIT requires each of the two vector cores to receive a whole number
// of 16-row NZ blocks, so the original matmul M dimension must divide by 32.
// REJECT-LABEL: func.func @row_split_unaligned_matmul
// REJECT: linalg.matmul
// DIAG: [cv-split] Function: row_split_unaligned_matmul
// DIAG-NEXT: [cv-split] Pre-check rejected function, skip
func.func @row_split_unaligned_matmul() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %lhs = tensor.empty() : tensor<16x16xf32>
  %rhs = tensor.empty() : tensor<16x16xf32>
  %out = tensor.empty() : tensor<16x16xf32>
  scf.for %iv = %c0 to %c16 step %c1 {
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<16x16xf32>, tensor<16x16xf32>)
        outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
  }
  return
}

// Matmul tile dimensions must be multiples of the hardware NZ block size.
// REJECT-LABEL: func.func @unaligned_tensor_matmul
// REJECT: linalg.matmul
// DIAG: [cv-split] Function: unaligned_tensor_matmul
// DIAG-NEXT: [cv-split] Pre-check rejected function, skip
func.func @unaligned_tensor_matmul() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %lhs = tensor.empty() : tensor<16x8xf32>
  %rhs = tensor.empty() : tensor<8x16xf32>
  %out = tensor.empty() : tensor<16x16xf32>
  scf.for %iv = %c0 to %c16 step %c1 {
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<16x8xf32>, tensor<8x16xf32>)
        outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
  }
  return
}
