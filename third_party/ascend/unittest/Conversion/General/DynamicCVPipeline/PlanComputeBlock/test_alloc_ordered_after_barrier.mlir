// RUN: triton-opt --debug-only=memory-effects-tracker --plan-compute-block %s 2>&1 | FileCheck %s

// An unknown op (here gpu.barrier, which exposes no memory-effect interface)
// acts as a full memory barrier. A buffer allocated after such an op must not
// move across it, so the barrier must show up in the allocation's Preds.

module {
  // Case 1: barrier and allocation live in the same block. The ordering edge is
  // barrier -> alloc.
  // CHECK-LABEL: func.func @alloc_after_barrier_same_block
  // CHECK:      Analyzing op {{.*}} = memref.alloc()
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] gpu.barrier
  func.func @alloc_after_barrier_same_block(%arg0: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %v = memref.load %arg0[%c0] : memref<4xf32>
    gpu.barrier
    %alloc = memref.alloc() : memref<4x4xf32>
    return
  }

  // Case 2: the barrier is nested inside scf.if. The ordering edge must NOT
  // cross the block boundary: it is emitted from the parent-block ancestor
  // (scf.if, which is itself unknown because it contains the barrier), and NOT
  // from the gpu.barrier inside the nested region.
  // CHECK-LABEL: func.func @alloc_after_nested_barrier
  // CHECK:      Analyzing op {{.*}} = memref.alloc()
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] scf.if
  func.func @alloc_after_nested_barrier(%cond: i1) {
    scf.if %cond {
      gpu.barrier
    }
    %alloc = memref.alloc() : memref<4x4xf32>
    return
  }
}
