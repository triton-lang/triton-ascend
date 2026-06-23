// RUN: (triton-opt --add_multi_buffer_inner_scope %s 2>&1 || echo "PASS") | FileCheck %s
// CHECK: Dynamic-shape tensor cross-block dep is unsupported
// CHECK: Memref type cross-block dep is unsupported
// CHECK: PASS

// T-unsupported-deps: Cross-block deps of types the multi-buffer path
// can't handle trigger pass failure with detailed logging.
//
// Setup:
//   - Block 8 (producer) defines:
//       * %src = arith.addf ... : tensor<64xf16>          (static-shape tensor)
//       * %dyn = arith.minsi %5, %c64 : index              (dynamic dim operand)
//       * %extracted_slice = tensor.extract_slice %src[0] [%dyn] [1]
//             : tensor<64xf16> to tensor<?xf16>            (DYNAMIC-shape tensor)
//       * %alloc9 = memref.alloc() : memref<64xf16, ...>
//       * %subview_22 = memref.subview %alloc9[0] [%dyn] [1]
//             : memref<64xf16, ...> to memref<?xf16, ...>  (MEMREF type)
//   - Block 9 (consumer) uses both via materialize_in_destination:
//       * bufferization.materialize_in_destination %extracted_slice
//             in writable %subview_22 : (tensor<?xf16>, memref<?xf16, ...>) -> ()
//
// Without including result-less ops in groupOpsBySsbufferId, materialize_in_destination
// is silently skipped, the deps are never collected, and the pass produces a
// wrong output. With the fix, both unsupported dep types are detected and the
// pass fails with a clear error message for each.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_unsupported_deps(%arg0: memref<?xf16>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f16
    %cst1 = arith.constant 1.0 : f16
    %empty = tensor.empty() : tensor<64xf16>
    %empty64 = tensor.empty() : tensor<64x64xf16>
    scope.scope : () -> () {
      %src = linalg.fill {ssbuffer.block_id = 7 : i32} ins(%cst1 : f16)
        outs(%empty : tensor<64xf16>) -> tensor<64xf16>
      %loop = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32
        iter_args(%arg = %src) -> (tensor<64xf16>) : i32 {
        %c5 = arith.constant {ssbuffer.block_id = 8 : i32} 5 : index
        // %dyn is the dynamic dimension fed into extract_slice / subview
        %dyn = arith.minsi %c5, %c64 {ssbuffer.block_id = 8 : i32} : index

        // Static-shape tensor produced in block 8 (the multi-buffer candidate)
        %22 = arith.addf %arg, %arg {ssbuffer.block_id = 8 : i32} : tensor<64xf16>

        // Dynamic-shape tensor dep from block 8 -> block 9
        %extracted_slice = tensor.extract_slice %22[0] [%dyn] [1]
          {ssbuffer.block_id = 8 : i32} : tensor<64xf16> to tensor<?xf16>

        // Memref dep from block 8 -> block 9
        %alloc8 = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<64xf16>
        %subview_22 = memref.subview %alloc8[0] [%dyn] [1]
          {ssbuffer.block_id = 8 : i32} : memref<64xf16> to memref<?xf16, strided<[1]>>

        // Consumer in block 9 uses both deps
        bufferization.materialize_in_destination %extracted_slice in writable %subview_22
          {ssbuffer.block_id = 9 : i32} : (tensor<?xf16>, memref<?xf16, strided<[1]>>) -> ()

        // Carry the static-shape tensor across iterations for the multi-buffer path
        %next = arith.addf %22, %arg {ssbuffer.block_id = 7 : i32} : tensor<64xf16>
        scf.yield %next : tensor<64xf16>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}