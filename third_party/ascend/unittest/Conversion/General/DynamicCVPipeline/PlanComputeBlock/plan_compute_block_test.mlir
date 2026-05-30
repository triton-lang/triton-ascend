// RUN: triton-opt --plan-compute-block %s | FileCheck %s

// ============================================================================
// TC-01: Single Block Comprehensive
// ============================================================================
// A single function/block covers the general non-control-flow behavior:
// - matmul classification as CUBE and independent cube group partitioning.
// - upstream CUBE propagation through fill, tensor.empty, transpose, to_tensor,
//   memref.alloc/copy, and no-result memref linalg.fill.
// - downstream CUBE propagation through extract_slice, materialize_in_destination,
//   direct hivm.hir.store, and extract_slice -> hivm.hir.store.
// - VECTOR partitioning after the current PlanVectorBlock pass.
// - one producer used by both CUBE and VECTOR consumers.
// - CUBE -> VECTOR transfer with multiple consumers and reorder-sensitive users.
// - vector accumulator legalization for matmul outs.
//   vector->cube, multi-dot partitioning, willCreateCycle, and reorder cases.

// CHECK-LABEL: func.func @tc01_single_block_comprehensive(
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[V19:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: arith.extf {{.*}} {ssbuffer.block_id = [[V19]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: arith.truncf {{.*}} {ssbuffer.block_id = [[V19]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: linalg.matmul {ssbuffer.block_id = [[C12:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CUBE alloc/to_tensor/matmul chain (block_id = 9)
// CHECK: memref.alloc() {ssbuffer.block_id = [[C9:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = [[C9]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.matmul {ssbuffer.block_id = [[C9]] : i32, ssbuffer.core_type = "CUBE"}
// More CUBE matmuls (block_ids = 8, 6)
// CHECK: linalg.matmul {ssbuffer.block_id = [[C8:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.matmul {ssbuffer.block_id = [[C6:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// VECTOR addf after matmul (block_id = 20)
// CHECK: arith.addf {{.*}}, %arg4 {ssbuffer.block_id = [[V20:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CUBE alloc/to_tensor/matmul chain (block_id = 10)
// CHECK: memref.alloc() {ssbuffer.block_id = [[C10:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = [[C10]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.matmul {ssbuffer.block_id = [[C10]] : i32, ssbuffer.core_type = "CUBE"}
// VECTOR ops (block_id = 21)
// CHECK: arith.extf {{.*}} {ssbuffer.block_id = [[V21:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[V21]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: arith.truncf {{.*}} {ssbuffer.block_id = [[V21]] : i32, ssbuffer.core_type = "VECTOR"}
// CUBE matmuls (block_ids = 11, 7, 4, 3, 2, 1, 5)
// CHECK: linalg.matmul {ssbuffer.block_id = [[C11:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.matmul {ssbuffer.block_id = [[C7:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: tensor.empty() {ssbuffer.block_id = [[C4:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.transpose {{.*}} {ssbuffer.block_id = [[C4]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.matmul {ssbuffer.block_id = [[C4]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.matmul {ssbuffer.block_id = [[C3:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: arith.constant {ssbuffer.block_id = [[C2:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = [[C2]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.fill {ssbuffer.block_id = [[C2]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.matmul {ssbuffer.block_id = [[C2]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: arith.constant {ssbuffer.block_id = [[C1:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: memref.alloc() {ssbuffer.block_id = [[C1]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: memref.alloc() {ssbuffer.block_id = [[C1]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: memref.copy {{.*}} {ssbuffer.block_id = [[C1]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: memref.copy {{.*}} {ssbuffer.block_id = [[C1]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = [[C1]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = [[C1]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: tensor.empty() {ssbuffer.block_id = [[C1]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.transpose {{.*}} {ssbuffer.block_id = [[C1]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: tensor.empty() {ssbuffer.block_id = [[C1]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.fill {ssbuffer.block_id = [[C1]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.matmul {ssbuffer.block_id = [[C1]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: tensor.extract_slice {{.*}} {ssbuffer.block_id = [[C1]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: bufferization.materialize_in_destination {{.*}} {ssbuffer.block_id = [[C1]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: hivm.hir.store {{.*}} {ssbuffer.block_id = [[C1]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: hivm.hir.store {{.*}} {ssbuffer.block_id = [[C1]] : i32, ssbuffer.core_type = "CUBE"}
// VECTOR ops (block_id = 22)
// CHECK: arith.constant {ssbuffer.block_id = [[V22:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: tensor.extract_slice {{.*}} {ssbuffer.block_id = [[V22]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: hivm.hir.store {{.*}} {ssbuffer.block_id = [[V22]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: hivm.hir.store {{.*}} {ssbuffer.block_id = [[V22]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: linalg.fill {ssbuffer.block_id = [[V22]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[V22]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: arith.mulf {{.*}} {ssbuffer.block_id = [[V22]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[V22]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = [[V22]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[V22]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[V22]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[V22]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[V22]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[V22]] : i32, ssbuffer.core_type = "VECTOR"}
// CUBE alloc/fill/to_tensor/matmul (block_id = 5)
// CHECK: memref.alloc() {ssbuffer.block_id = [[C5:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.fill {ssbuffer.block_id = [[C5]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = [[C5]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.matmul {ssbuffer.block_id = [[C5]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: return {{.*}}
func.func @tc01_single_block_comprehensive(
    %ma: memref<4x4xf16>,
    %mb: memref<4x4xf16>,
    %shared_src: memref<4x4xf32>,
    %fa: tensor<4x4xf32>,
    %fb: tensor<4x4xf32>,
    %ta: tensor<4x4xf16>,
    %tb: tensor<4x4xf16>,
    %dst0: tensor<4x4xf32>,
    %dst1: tensor<4x4xf32>,
    %memdst: memref<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf16>) {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32

  // Load path: memref copy -> to_tensor -> transpose -> matmul.
  %alloc_a = memref.alloc() : memref<4x4xf16>
  %alloc_b = memref.alloc() : memref<4x4xf16>
  memref.copy %ma, %alloc_a : memref<4x4xf16> to memref<4x4xf16>
  memref.copy %mb, %alloc_b : memref<4x4xf16> to memref<4x4xf16>
  %load_a = bufferization.to_tensor %alloc_a restrict writable : memref<4x4xf16>
  %load_b = bufferization.to_tensor %alloc_b restrict writable : memref<4x4xf16>
  %tr_init = tensor.empty() : tensor<4x4xf16>
  %load_bt = linalg.transpose ins(%load_b : tensor<4x4xf16>) outs(%tr_init : tensor<4x4xf16>) permutation = [1, 0]
  %mm0_empty = tensor.empty() : tensor<4x4xf32>
  %mm0_fill = linalg.fill ins(%zero : f32) outs(%mm0_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %mm0 = linalg.matmul ins(%load_a, %load_bt : tensor<4x4xf16>, tensor<4x4xf16>) outs(%mm0_fill : tensor<4x4xf32>) -> tensor<4x4xf32>

  // Downstream CUBE users and CUBE -> VECTOR transfer.
  %slice0 = tensor.extract_slice %mm0[0, 0] [4, 4] [1, 1] : tensor<4x4xf32> to tensor<4x4xf32>
  bufferization.materialize_in_destination %mm0 in writable %memdst : (tensor<4x4xf32>, memref<4x4xf32>) -> ()
  %store0 = hivm.hir.store ins(%mm0 : tensor<4x4xf32>) outs(%dst0 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %store1 = hivm.hir.store ins(%slice0 : tensor<4x4xf32>) outs(%dst1 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %vec_fill = linalg.fill ins(%one : f32) outs(%dst0 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %vec_from_cube0 = arith.addf %mm0, %vec_fill : tensor<4x4xf32>
  %vec_from_cube1 = arith.mulf %vec_from_cube0, %store0 : tensor<4x4xf32>
  %vec_from_cube2 = arith.addf %vec_from_cube1, %store1 : tensor<4x4xf32>

  // Same producer has both CUBE and VECTOR consumers.
  %shared_tensor = bufferization.to_tensor %shared_src restrict writable : memref<4x4xf32>
  %shared_fill = linalg.fill ins(%two : f32) outs(%dst1 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %shared_mm = linalg.matmul ins(%shared_tensor, %fb : tensor<4x4xf32>, tensor<4x4xf32>) outs(%shared_fill : tensor<4x4xf32>) -> tensor<4x4xf32>
  %shared_vec = arith.addf %shared_tensor, %fa : tensor<4x4xf32>

  // Vector accumulator legalization: matmul outs is VECTOR, pass should split it.
  %vec_acc = arith.addf %fa, %fb : tensor<4x4xf32>
  %legalized_mm = linalg.matmul ins(%ta, %tb : tensor<4x4xf16>, tensor<4x4xf16>) outs(%vec_acc : tensor<4x4xf32>) -> tensor<4x4xf32>

  // Transpose compute-input boundary: extf/truncf stay VECTOR, transpose is CUBE.
  %wide = arith.extf %tb : tensor<4x4xf16> to tensor<4x4xf32>
  %narrow = arith.truncf %wide : tensor<4x4xf32> to tensor<4x4xf16>
  %tr2_init = tensor.empty() : tensor<4x4xf16>
  %computed_t = linalg.transpose ins(%narrow : tensor<4x4xf16>) outs(%tr2_init : tensor<4x4xf16>) permutation = [1, 0]
  %tr_mm = linalg.matmul ins(%ta, %computed_t : tensor<4x4xf16>, tensor<4x4xf16>) outs(%dst0 : tensor<4x4xf32>) -> tensor<4x4xf32>

  // No-result fill with memref outs must be classified with the downstream CUBE use.
  %scratch = memref.alloc() : memref<4x4xf32>
  linalg.fill ins(%zero : f32) outs(%scratch : memref<4x4xf32>)
  %scratch_tensor = bufferization.to_tensor %scratch restrict writable : memref<4x4xf32>
  %scratch_mm = linalg.matmul ins(%scratch_tensor, %fa : tensor<4x4xf32>, tensor<4x4xf32>) outs(%dst1 : tensor<4x4xf32>) -> tensor<4x4xf32>

  // VF findCandidates/updateCandidates: VECTOR chains are blocked by
  // CUBE matmul barriers, then new VECTOR candidates appear downstream.
  %vf0 = arith.addf %fa, %fb : tensor<4x4xf32>
  %vf1 = arith.mulf %vf0, %fa : tensor<4x4xf32>
  %vf2 = arith.truncf %vf1 : tensor<4x4xf32> to tensor<4x4xf16>
  %vf_mm0 = linalg.matmul ins(%vf2, %ta : tensor<4x4xf16>, tensor<4x4xf16>) outs(%dst0 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %vf3 = arith.addf %vf_mm0, %fb : tensor<4x4xf32>
  %vf4 = arith.mulf %vf3, %fa : tensor<4x4xf32>
  %vf5 = arith.truncf %vf4 : tensor<4x4xf32> to tensor<4x4xf16>
  %vf_mm1 = linalg.matmul ins(%vf5, %tb : tensor<4x4xf16>, tensor<4x4xf16>) outs(%vf_mm0 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %vf6 = arith.addf %vf_mm1, %vf3 : tensor<4x4xf32>

  // cutError variants: one VECTOR prefix is kept because it feeds a CUBE
  // barrier, while an independent VECTOR suffix must be cut and scheduled later.
  %cut_a = arith.addf %fa, %fb : tensor<4x4xf32>
  %cut_b = arith.mulf %cut_a, %fa : tensor<4x4xf32>
  %cut_c = arith.addf %cut_b, %cut_b : tensor<4x4xf32>
  %cut_cast = arith.truncf %cut_c : tensor<4x4xf32> to tensor<4x4xf16>
  %cut_mm = linalg.matmul ins(%cut_cast, %ta : tensor<4x4xf16>, tensor<4x4xf16>) outs(%dst1 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %cut_e = arith.addf %fa, %fa : tensor<4x4xf32>
  %cut_f = arith.addf %cut_mm, %cut_e : tensor<4x4xf32>

  // multiple dot
  // anchors, a cube result feeding another cube dot, and vector side users.
  %dep_alloc0 = memref.alloc() : memref<4x4xf16>
  %dep_alloc1 = memref.alloc() : memref<4x4xf16>
  %dep_t0 = bufferization.to_tensor %dep_alloc0 restrict writable : memref<4x4xf16>
  %dep_t1 = bufferization.to_tensor %dep_alloc1 restrict writable : memref<4x4xf16>
  %dep_pre0 = arith.subf %fa, %fb : tensor<4x4xf32>
  %dep_pre1 = arith.mulf %dep_pre0, %fa : tensor<4x4xf32>
  %dep_cast0 = arith.truncf %dep_pre1 : tensor<4x4xf32> to tensor<4x4xf16>
  %dep_mm0 = linalg.matmul ins(%dep_cast0, %dep_t0 : tensor<4x4xf16>, tensor<4x4xf16>) outs(%dst0 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %dep_cast1 = arith.truncf %dep_mm0 : tensor<4x4xf32> to tensor<4x4xf16>
  %dep_mm1 = linalg.matmul ins(%dep_cast1, %dep_t1 : tensor<4x4xf16>, tensor<4x4xf16>) outs(%dst1 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %dep_side0 = arith.extf %dep_cast0 : tensor<4x4xf16> to tensor<4x4xf32>
  %dep_side1 = arith.addf %dep_side0, %dep_mm1 : tensor<4x4xf32>
  %dep_cast2 = arith.truncf %dep_side1 : tensor<4x4xf32> to tensor<4x4xf16>
  %dep_mm2 = linalg.matmul ins(%dep_cast2, %dep_cast1 : tensor<4x4xf16>, tensor<4x4xf16>) outs(%dep_mm1 : tensor<4x4xf32>) -> tensor<4x4xf32>

  // willCreateCycle for dot fusion: A feeds both a VECTOR path to C and
  // C directly, so A and C cannot be merged if it would create a cycle.
  %cycle_a = arith.truncf %fa : tensor<4x4xf32> to tensor<4x4xf16>
  %cycle_b0 = arith.extf %cycle_a : tensor<4x4xf16> to tensor<4x4xf32>
  %cycle_b1 = arith.addf %cycle_b0, %fb : tensor<4x4xf32>
  %cycle_b2 = arith.truncf %cycle_b1 : tensor<4x4xf32> to tensor<4x4xf16>
  %cycle_c = linalg.matmul ins(%cycle_b2, %cycle_a : tensor<4x4xf16>, tensor<4x4xf16>) outs(%dst0 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %cycle_d = arith.addf %cycle_c, %fa : tensor<4x4xf32>

  return %vec_from_cube2, %shared_mm, %shared_vec, %legalized_mm, %tr_mm, %scratch_mm, %vf6, %cut_f, %dep_mm2, %cycle_b2 : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf16>
}

// -----

// ============================================================================
// TC-02: Nested Blocks / Control Flow
// ============================================================================
// Based on the TC-01 patterns but adds scf.for, scf.if, nested blocks, and
// multi-result yields. Checks focus on propagation through region boundaries.

// CHECK-LABEL: func.func @tc02_nested_control_flow(
// CHECK-NOT: call
// CHECK: arith.constant {ssbuffer.block_id = [[V28:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} 0 : index
// CHECK: arith.constant {ssbuffer.block_id = [[V28]] : i32, ssbuffer.core_type = "VECTOR"} 1 : index
// CHECK: arith.addf %arg6, %arg7 {ssbuffer.block_id = [[V28]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: arith.constant {ssbuffer.block_id = [[V28]] : i32, ssbuffer.core_type = "VECTOR"} dense<1.000000e+00>
// CHECK: arith.constant {ssbuffer.block_id = [[C16:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: memref.alloc() {ssbuffer.block_id = [[C16]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: memref.alloc() {ssbuffer.block_id = [[C16]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: memref.copy {{.*}} {ssbuffer.block_id = [[C16]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: memref.copy {{.*}} {ssbuffer.block_id = [[C16]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = [[C16]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = [[C16]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: tensor.empty() {ssbuffer.block_id = [[C16]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.fill {ssbuffer.block_id = [[C16]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.matmul {ssbuffer.block_id = [[C16]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[V29:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: arith.mulf {{.*}} {ssbuffer.block_id = [[V29]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: scf.if %arg5 -> (tensor<4x4xf32>) {
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[V25:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: scf.yield {ssbuffer.core_type = "VECTOR"}
// CHECK: } else {
// CHECK: scf.yield {ssbuffer.core_type = "VECTOR"}
// CHECK: } {ssbuffer.core_type = "VECTOR"}
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[V30:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: scf.for {{.*}} = {{.*}} to %arg4 step {{.*}} iter_args({{.*}} = {{.*}}, {{.*}} = {{.*}}) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK: scf.if %arg5 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[V23:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: linalg.matmul {ssbuffer.block_id = [[C13:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: scf.yield {ssbuffer.core_type = "CUBE, VECTOR"}
// CHECK: } else {
// CHECK: arith.mulf {{.*}} {ssbuffer.block_id = [[V24:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: scf.yield {ssbuffer.core_type = "CUBE, VECTOR"}
// CHECK: } {ssbuffer.core_type = "CUBE, VECTOR"}
// CHECK: scf.yield {ssbuffer.core_type = "CUBE, CUBE"}
// CHECK: } {ssbuffer.core_type = "CUBE, CUBE"}
// CHECK: scf.for {{.*}} = {{.*}} to %arg4 step {{.*}} iter_args({{.*}} = {{.*}}, {{.*}} = {{.*}}) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK: arith.truncf {{.*}} {ssbuffer.block_id = [[V26:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: tensor.empty() {ssbuffer.block_id = [[V26]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: arith.truncf {{.*}} {ssbuffer.block_id = [[V26]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: tensor.empty() {ssbuffer.block_id = [[V26]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: linalg.matmul {ssbuffer.block_id = [[C15:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.matmul {ssbuffer.block_id = [[C14:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[V27:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[V27]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: scf.yield {ssbuffer.core_type = "VECTOR, VECTOR"}
// CHECK: } {ssbuffer.core_type = "VECTOR, VECTOR"}
// CHECK: return {{.*}}
func.func @tc02_nested_control_flow(
    %ma: memref<4x4xf16>,
    %mb: memref<4x4xf16>,
    %ta: tensor<4x4xf16>,
    %tb: tensor<4x4xf16>,
    %n: index,
    %cond: i1,
    %v0: tensor<4x4xf32>,
    %v1: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
  %zero = arith.constant 0.0 : f32
  %alloc_a = memref.alloc() : memref<4x4xf16>
  %alloc_b = memref.alloc() : memref<4x4xf16>
  memref.copy %ma, %alloc_a : memref<4x4xf16> to memref<4x4xf16>
  memref.copy %mb, %alloc_b : memref<4x4xf16> to memref<4x4xf16>
  %load_a = bufferization.to_tensor %alloc_a restrict writable : memref<4x4xf16>
  %load_b = bufferization.to_tensor %alloc_b restrict writable : memref<4x4xf16>
  %empty = tensor.empty() : tensor<4x4xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %outer_mm = linalg.matmul ins(%load_a, %load_b : tensor<4x4xf16>, tensor<4x4xf16>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
  %outer_vec = arith.addf %outer_mm, %v0 : tensor<4x4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop:2 = scf.for %i = %c0 to %n step %c1 iter_args(%cube_acc = %outer_mm, %vec_acc = %outer_vec) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
    %if:2 = scf.if %cond -> (tensor<4x4xf32>, tensor<4x4xf32>) {
      %inner_mm = linalg.matmul ins(%ta, %tb : tensor<4x4xf16>, tensor<4x4xf16>) outs(%cube_acc : tensor<4x4xf32>) -> tensor<4x4xf32>
      %inner_vec = arith.addf %vec_acc, %v1 : tensor<4x4xf32>
      scf.yield %inner_mm, %inner_vec : tensor<4x4xf32>, tensor<4x4xf32>
    } else {
      %else_vec = arith.mulf %vec_acc, %v1 : tensor<4x4xf32>
      scf.yield %cube_acc, %else_vec : tensor<4x4xf32>, tensor<4x4xf32>
    }
    scf.yield %if#0, %if#1 : tensor<4x4xf32>, tensor<4x4xf32>
  }

  // an if region has VECTOR values produced before the
  // region, used inside the region, and merged back after the region.
  %if_pre0 = arith.addf %v0, %v1 : tensor<4x4xf32>
  %if_pre1 = arith.mulf %v1, %outer_vec : tensor<4x4xf32>
  %region_v = scf.if %cond -> (tensor<4x4xf32>) {
    %inside_v = arith.addf %if_pre0, %v1 : tensor<4x4xf32>
    scf.yield %inside_v : tensor<4x4xf32>
  } else {
    scf.yield %if_pre1 : tensor<4x4xf32>
  }
  %region_after = arith.addf %if_pre1, %region_v : tensor<4x4xf32>

  // two loop-carried values each feed a separate dot and are
  // yielded independently, covering inter-iteration dependencies.
  %cst_tensor = arith.constant dense<1.0> : tensor<4x4xf32>
  %loop2:2 = scf.for %j = %c0 to %n step %c1 iter_args(%arg_a = %loop#0, %arg_b = %outer_mm) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
    %a_cast = arith.truncf %arg_a : tensor<4x4xf32> to tensor<4x4xf16>
    %out_b = tensor.empty() : tensor<4x4xf32>
    %b_mm = linalg.matmul ins(%a_cast, %ta : tensor<4x4xf16>, tensor<4x4xf16>) outs(%out_b : tensor<4x4xf32>) -> tensor<4x4xf32>
    %c_next = arith.addf %b_mm, %cst_tensor : tensor<4x4xf32>
    %d_cast = arith.truncf %arg_b : tensor<4x4xf32> to tensor<4x4xf16>
    %out_e = tensor.empty() : tensor<4x4xf32>
    %e_mm = linalg.matmul ins(%d_cast, %tb : tensor<4x4xf16>, tensor<4x4xf16>) outs(%out_e : tensor<4x4xf32>) -> tensor<4x4xf32>
    %f_next = arith.addf %e_mm, %cst_tensor : tensor<4x4xf32>
    scf.yield %c_next, %f_next : tensor<4x4xf32>, tensor<4x4xf32>
  }
  return %loop#0, %loop#1, %region_after, %loop2#0, %loop2#1 : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// ============================================================================
// TC-03: Edge Cases
// ============================================================================

// CHECK-LABEL: func.func @tc03a_empty_func(
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @tc03a_empty_func() {
  return
}

// CHECK-LABEL: func.func @tc03b_all_vector_no_matmul(
// CHECK: arith.addf %arg0, %arg1 {ssbuffer.block_id = [[V31:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : f32
// CHECK: return {{.*}} : f32
func.func @tc03b_all_vector_no_matmul(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

// CHECK-LABEL: func.func @tc03c_empty_loop(
// CHECK: arith.constant {ssbuffer.block_id = [[V32:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} 0 : index
// CHECK: arith.constant {ssbuffer.block_id = [[V32]] : i32, ssbuffer.core_type = "VECTOR"} 1 : index
// CHECK: scf.for {{.*}} = {{.*}} to %arg0 step {{.*}} iter_args({{.*}} = %arg1) -> (tensor<4x4xf32>) {
// CHECK: scf.yield {ssbuffer.core_type = "VECTOR"}
// CHECK: } {ssbuffer.core_type = "VECTOR"}
// CHECK: return {{.*}} : tensor<4x4xf32>
func.func @tc03c_empty_loop(%n: index, %init: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%v = %init) -> (tensor<4x4xf32>) {
    scf.yield %v : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @tc03d_all_cube_matmul_only(
// CHECK: linalg.matmul {ssbuffer.block_id = [[C17:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: return {{.*}} : tensor<4x4xf32>
func.func @tc03d_all_cube_matmul_only(
    %a: tensor<4x4xf16>,
    %b: tensor<4x4xf16>,
    %acc: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = linalg.matmul ins(%a, %b : tensor<4x4xf16>, tensor<4x4xf16>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @tc03e_memref_fill_no_result(
// CHECK: arith.constant {ssbuffer.block_id = [[V33:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: memref.alloc() {ssbuffer.block_id = [[C18:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.fill {ssbuffer.block_id = [[C18]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = [[C18]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: linalg.matmul {ssbuffer.block_id = [[C18]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: return {{.*}} : tensor<4x4xf32>
func.func @tc03e_memref_fill_no_result(
    %a: tensor<4x4xf32>,
    %dst: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %scratch = memref.alloc() : memref<4x4xf32>
  linalg.fill ins(%zero : f32) outs(%scratch : memref<4x4xf32>)
  %t = bufferization.to_tensor %scratch restrict writable : memref<4x4xf32>
  %r = linalg.matmul ins(%t, %a : tensor<4x4xf32>, tensor<4x4xf32>) outs(%dst : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @tc03f_if_all_vector(
// CHECK: scf.if %arg0 -> (tensor<4x4xf32>) {
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[V34:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: scf.yield {ssbuffer.core_type = "VECTOR"}
// CHECK: } else {
// CHECK: arith.mulf {{.*}} {ssbuffer.block_id = [[V35:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: scf.yield {ssbuffer.core_type = "VECTOR"}
// CHECK: } {ssbuffer.core_type = "VECTOR"}
// CHECK: return {{.*}} : tensor<4x4xf32>
func.func @tc03f_if_all_vector(%cond: i1, %x: tensor<4x4xf32>, %y: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = scf.if %cond -> (tensor<4x4xf32>) {
    %a = arith.addf %x, %y : tensor<4x4xf32>
    scf.yield %a : tensor<4x4xf32>
  } else {
    %b = arith.mulf %x, %y : tensor<4x4xf32>
    scf.yield %b : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}
