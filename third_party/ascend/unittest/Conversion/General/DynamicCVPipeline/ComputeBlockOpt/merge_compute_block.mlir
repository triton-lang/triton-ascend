// RUN: triton-opt --allow-unregistered-dialect --merge-compute-block %s | FileCheck %s

// Test merge-compute-block pass: merges adjacent VECTOR blocks between CUBE blocks,
// cloning packed CUBE load scopes that downstream blocks depend on.
//
// Pattern: C1(6,CUBE) → V1(32,VECTOR) → C2(8,CUBE) → V2(33,VECTOR) → C3(10,CUBE) → C4(12,CUBE)
// After pass:
//   - V2(33) merged into V1(32): all block_id 33 → 32
//   - C3(10) gets a cloned ssbuffer.load scope from C2(8)

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.intra_buf_count = 3 : i32, ssbuffer.inter_core_buf_count = 2 : i32} {
  // CHECK-LABEL: func.func @_swa_bwd_dkdv_kernel(
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.subBlock = 32 : i32}
  // CHECK: math.exp {{.*}} {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.subBlock = 32 : i32}
  // CHECK: arith.truncf {{.*}} {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.subBlock = 32 : i32}
  // CHECK: %[[ORIGINAL:.*]] = scope.scope : () -> tensor<64x64xbf16> {
  // CHECK: } {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE", ssbuffer.load}
  // CHECK: linalg.matmul {{.*}} ins({{.*}}, %[[ORIGINAL]] :
  // CHECK: %[[CLONE:.*]] = scope.scope : () -> tensor<64x64xbf16> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x64xbf16>
  // CHECK: scf.if {{.*}} {
  // CHECK: linalg.fill ins({{.*}}) outs(%[[ALLOC]] :
  // CHECK: arith.index_cast
  // CHECK: arith.muli
  // CHECK: arith.addi
  // CHECK: memref.reinterpret_cast
  // CHECK: memref.subview
  // CHECK: memref.subview
  // CHECK: memref.copy
  // CHECK: bufferization.to_tensor %[[ALLOC]] restrict writable :
  // CHECK: scope.return
  // CHECK: } {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE", ssbuffer.load}
  // CHECK: linalg.matmul {{.*}} ins(%[[CLONE]], %[[CLONE]] :
  // CHECK: arith.subf {{.*}} {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.subBlock = 33 : i32}
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.subBlock = 33 : i32}
  // CHECK: arith.truncf {{.*}} {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.subBlock = 33 : i32}
  func.func @_swa_bwd_dkdv_kernel(
    %arg0: memref<?xbf16> {tt.tensor_kind = 0 : i32},
    %arg1: memref<?xbf16> {tt.tensor_kind = 0 : i32},
    %arg2: i32 {tt.divisibility = 16 : i32}
  ) attributes {global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst_bf16 = arith.constant 0.000000e+00 : bf16
    %cst_f32 = arith.constant 0.000000e+00 : f32
    %cst_scale = arith.constant 0.0883883461 : f32
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32

    %empty_2d_bf16 = tensor.empty() : tensor<64x64xbf16>

    // VECTOR block constants
    %cst_v = arith.constant {ssbuffer.block_id = 49 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %empty_v = tensor.empty() {ssbuffer.block_id = 49 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    %scale = linalg.fill {ssbuffer.block_id = 49 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_scale : f32) outs(%empty_v : tensor<64x64xf32>) -> tensor<64x64xf32>

    // CUBE block constants
    %cst_bf16_cube = arith.constant {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : bf16
    %cst_f32_cube = arith.constant {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %empty_cube_f32 = tensor.empty() {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
    %cube_init = linalg.fill {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_f32_cube : f32) outs(%empty_cube_f32 : tensor<64x64xf32>) -> tensor<64x64xf32>

    scf.for %iv = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
      // Shared index computations used by C1 and C4
      %iv_idx = arith.index_cast %iv {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %offset64 = arith.muli %iv_idx, %c64 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : index
      %stride_idx = arith.index_cast %arg2 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %cond_c2 = arith.cmpi slt, %iv_idx, %c64 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : index

      // ==========================================
      // C1: Block 6 (CUBE) — produces %tensor_c1 used by V1(32) and C4(12)
      // ==========================================
      %tensor_c1 = scope.scope : () -> tensor<64x64xbf16> {
        %alloc_c1 = memref.alloc() : memref<64x64xbf16>
        %cond_c1 = arith.cmpi slt, %iv_idx, %c64 : index
        scf.if %cond_c1 {
          linalg.fill ins(%cst_bf16 : bf16) outs(%alloc_c1 : memref<64x64xbf16>)
        } {hivm.unlikely_condition}
        %reint_c1 = memref.reinterpret_cast %arg0 to offset: [%offset64], sizes: [64, 64], strides: [%stride_idx, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
        %subv_c1_src = memref.subview %reint_c1[0, 0] [64, 64] [1, 1] : memref<64x64xbf16, strided<[?, 1], offset: ?>> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
        %subv_c1_dst = memref.subview %alloc_c1[0, 0] [64, 64] [1, 1] : memref<64x64xbf16> to memref<64x64xbf16, strided<[64, 1]>>
        memref.copy %subv_c1_src, %subv_c1_dst : memref<64x64xbf16, strided<[?, 1], offset: ?>> to memref<64x64xbf16, strided<[64, 1]>>
        %loaded_c1 = bufferization.to_tensor %alloc_c1 restrict writable : memref<64x64xbf16> to tensor<64x64xbf16>
        scope.return %loaded_c1 : tensor<64x64xbf16>
      } {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE", ssbuffer.load}
      %matmul_c1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%tensor_c1, %tensor_c1 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%cube_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // ==========================================
      // V1: Block 32 (VECTOR) — CUBE pred=6, CUBE succ=8, edge 32→33 via %v1_exp
      // ==========================================
      %v1_mul = arith.mulf %matmul_c1, %scale {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %v1_exp = math.exp %v1_mul {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %v1_trunc = arith.truncf %v1_exp {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xbf16>

      // ==========================================
      // C2: Block 8 (CUBE) — has copy chain with dedicated arith ops that C3(10) depends on
      // ==========================================
      %tensor_c2 = scope.scope : () -> tensor<64x64xbf16> {
        %alloc_c2 = memref.alloc() : memref<64x64xbf16>
        scf.if %cond_c2 {
          linalg.fill ins(%cst_bf16 : bf16) outs(%alloc_c2 : memref<64x64xbf16>)
        } {hivm.unlikely_condition}
        // Dedicated arith ops in C2 for the reinterpret_cast offset computation
        %c2_stride = arith.index_cast %arg2 : i32 to index
        %c2_mul = arith.muli %iv_idx, %c2_stride : index
        %c2_offset = arith.addi %offset64, %c2_mul : index
        %reint_c2 = memref.reinterpret_cast %arg1 to offset: [%c2_offset], sizes: [64, 64], strides: [%c2_stride, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
        %subv_c2_src = memref.subview %reint_c2[0, 0] [64, 64] [1, 1] : memref<64x64xbf16, strided<[?, 1], offset: ?>> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
        %subv_c2_dst = memref.subview %alloc_c2[0, 0] [64, 64] [1, 1] : memref<64x64xbf16> to memref<64x64xbf16, strided<[64, 1]>>
        memref.copy %subv_c2_src, %subv_c2_dst : memref<64x64xbf16, strided<[?, 1], offset: ?>> to memref<64x64xbf16, strided<[64, 1]>>
        %loaded_c2 = bufferization.to_tensor %alloc_c2 restrict writable : memref<64x64xbf16> to tensor<64x64xbf16>
        scope.return %loaded_c2 : tensor<64x64xbf16>
      } {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE", ssbuffer.load}
      %transposed_v1 = linalg.transpose ins(%v1_trunc : tensor<64x64xbf16>) outs(%empty_2d_bf16 : tensor<64x64xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"}
      %matmul_c2 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%transposed_v1, %tensor_c2 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%cube_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // ==========================================
      // C3: Block 10 (CUBE) — uses %tensor_c2 from C2, triggers clone from C2
      // After pass: cloned load scope, including address math, padding and copy, from C2
      // ==========================================
      %matmul_c3 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%tensor_c2, %tensor_c2 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%cube_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // ==========================================
      // V2: Block 33 (VECTOR) — uses %v1_exp (edge 32→33), %matmul_c3 (edge 10→33)
      // After pass: merged into V1(32), all block_id 33 → 32
      // ==========================================
      %v2_sub = arith.subf %matmul_c3, %v1_exp {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %v2_mul = arith.mulf %v1_exp, %v2_sub {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %v2_mul2 = arith.mulf %v2_mul, %scale {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %v2_trunc = arith.truncf %v2_mul2 {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xbf16>

      // ==========================================
      // C4: Block 12 (CUBE) — uses %v2_trunc (edge 33→12) and %tensor_c1 (edge 6→12)
      // ==========================================
      %transposed_v2 = linalg.transpose ins(%v2_trunc : tensor<64x64xbf16>) outs(%empty_2d_bf16 : tensor<64x64xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"}
      %matmul_c4 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%transposed_v2, %tensor_c1 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%cube_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // Store output to prevent DCE
      %reint_out = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [64, 64], strides: [%stride_idx, 1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
      %trunc_c4 = arith.truncf %matmul_c4 {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xbf16>
      bufferization.materialize_in_destination %trunc_c4 in writable %reint_out {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : (tensor<64x64xbf16>, memref<64x64xbf16, strided<[?, 1], offset: ?>>) -> ()
    }
    return
  }

  // Test case 2: C3(10) consumes BOTH %matmul_c2 (C2's L0C result) and
  // %tensor_c2 (C2's packed load). Cloning the load cannot remove the matmul
  // dependency, so the cross-Cube clone still leaves a cycle via
  // %matmul_c2 (C2 -> C3), so the clone attempt fails and must be rolled back
  // without leaving a residual cloned load scope in block 10.
  // CHECK-LABEL: func.func @_swa_bwd_dkdv_kernel_l0c(
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: %[[LOAD:.*]] = scope.scope : () -> tensor<64x64xbf16> {
  // CHECK: } {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE", ssbuffer.load}
  // CHECK: %[[MM:.*]] = linalg.matmul {{.*}}ssbuffer.block_id = 8 : i32{{.*}} ins({{.*}}, %[[LOAD]] :
  // CHECK-NOT: scope.scope
  // CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 10 : i32{{.*}} ins(%[[MM]], %[[LOAD]] :
  // CHECK: arith.subf {{.*}} {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK-NOT: scope.scope
  // CHECK: return
  func.func @_swa_bwd_dkdv_kernel_l0c(
    %arg0: memref<?xbf16> {tt.tensor_kind = 0 : i32},
    %arg1: memref<?xbf16> {tt.tensor_kind = 0 : i32},
    %arg2: i32 {tt.divisibility = 16 : i32}
  ) attributes {global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst_bf16 = arith.constant 0.000000e+00 : bf16
    %cst_scale = arith.constant 0.0883883461 : f32
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32

    %empty_2d_bf16 = tensor.empty() : tensor<64x64xbf16>

    // VECTOR block constants
    %cst_v = arith.constant {ssbuffer.block_id = 49 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %empty_v = tensor.empty() {ssbuffer.block_id = 49 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    %scale = linalg.fill {ssbuffer.block_id = 49 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_scale : f32) outs(%empty_v : tensor<64x64xf32>) -> tensor<64x64xf32>

    // CUBE block constants
    %cst_f32_cube = arith.constant {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %empty_cube_f32 = tensor.empty() {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
    %cube_init = linalg.fill {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_f32_cube : f32) outs(%empty_cube_f32 : tensor<64x64xf32>) -> tensor<64x64xf32>

    scf.for %iv = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
      // Shared index computations used by C1 and C4
      %iv_idx = arith.index_cast %iv {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %offset64 = arith.muli %iv_idx, %c64 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : index
      %stride_idx = arith.index_cast %arg2 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %cond_c2 = arith.cmpi slt, %iv_idx, %c64 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : index

      // ==========================================
      // C1: Block 6 (CUBE) — produces %tensor_c1 used by V1(32) and C4(12)
      // ==========================================
      %tensor_c1 = scope.scope : () -> tensor<64x64xbf16> {
        %alloc_c1 = memref.alloc() : memref<64x64xbf16>
        %cond_c1 = arith.cmpi slt, %iv_idx, %c64 : index
        scf.if %cond_c1 {
          linalg.fill ins(%cst_bf16 : bf16) outs(%alloc_c1 : memref<64x64xbf16>)
        } {hivm.unlikely_condition}
        %reint_c1 = memref.reinterpret_cast %arg0 to offset: [%offset64], sizes: [64, 64], strides: [%stride_idx, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
        %subv_c1_src = memref.subview %reint_c1[0, 0] [64, 64] [1, 1] : memref<64x64xbf16, strided<[?, 1], offset: ?>> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
        %subv_c1_dst = memref.subview %alloc_c1[0, 0] [64, 64] [1, 1] : memref<64x64xbf16> to memref<64x64xbf16, strided<[64, 1]>>
        memref.copy %subv_c1_src, %subv_c1_dst : memref<64x64xbf16, strided<[?, 1], offset: ?>> to memref<64x64xbf16, strided<[64, 1]>>
        %loaded_c1 = bufferization.to_tensor %alloc_c1 restrict writable : memref<64x64xbf16> to tensor<64x64xbf16>
        scope.return %loaded_c1 : tensor<64x64xbf16>
      } {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE", ssbuffer.load}
      %matmul_c1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%tensor_c1, %tensor_c1 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%cube_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // ==========================================
      // V1: Block 32 (VECTOR) — CUBE pred=6, CUBE succ=8, edge 32→33 via %v1_exp
      // ==========================================
      %v1_mul = arith.mulf %matmul_c1, %scale {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %v1_exp = math.exp %v1_mul {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %v1_trunc = arith.truncf %v1_exp {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xbf16>

      // ==========================================
      // C2: Block 8 (CUBE) — packed load feeds both its own matmul and C3
      // ==========================================
      %tensor_c2 = scope.scope : () -> tensor<64x64xbf16> {
        %alloc_c2 = memref.alloc() : memref<64x64xbf16>
        scf.if %cond_c2 {
          linalg.fill ins(%cst_bf16 : bf16) outs(%alloc_c2 : memref<64x64xbf16>)
        } {hivm.unlikely_condition}
        %c2_stride = arith.index_cast %arg2 : i32 to index
        %c2_mul = arith.muli %iv_idx, %c2_stride : index
        %c2_offset = arith.addi %offset64, %c2_mul : index
        %reint_c2 = memref.reinterpret_cast %arg1 to offset: [%c2_offset], sizes: [64, 64], strides: [%c2_stride, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
        %subv_c2_src = memref.subview %reint_c2[0, 0] [64, 64] [1, 1] : memref<64x64xbf16, strided<[?, 1], offset: ?>> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
        %subv_c2_dst = memref.subview %alloc_c2[0, 0] [64, 64] [1, 1] : memref<64x64xbf16> to memref<64x64xbf16, strided<[64, 1]>>
        memref.copy %subv_c2_src, %subv_c2_dst : memref<64x64xbf16, strided<[?, 1], offset: ?>> to memref<64x64xbf16, strided<[64, 1]>>
        %loaded_c2 = bufferization.to_tensor %alloc_c2 restrict writable : memref<64x64xbf16> to tensor<64x64xbf16>
        scope.return %loaded_c2 : tensor<64x64xbf16>
      } {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE", ssbuffer.load}
      %transposed_v1 = linalg.transpose ins(%v1_trunc : tensor<64x64xbf16>) outs(%empty_2d_bf16 : tensor<64x64xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"}
      %matmul_c2 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%transposed_v1, %tensor_c2 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%cube_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // ==========================================
      // C3: Block 10 (CUBE) — consumes both the L0C result and packed load.
      // The load clone is rolled back because the L0C dependency keeps a cycle.
      // ==========================================
      %matmul_c3 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%matmul_c2, %tensor_c2 : tensor<64x64xf32>, tensor<64x64xbf16>) outs(%cube_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // ==========================================
      // V2: Block 33 (VECTOR) — uses %v1_exp (edge 32→33), %matmul_c3 (edge 10→33)
      // ==========================================
      %v2_sub = arith.subf %matmul_c3, %v1_exp {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %v2_mul = arith.mulf %v1_exp, %v2_sub {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %v2_mul2 = arith.mulf %v2_mul, %scale {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %v2_trunc = arith.truncf %v2_mul2 {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xbf16>

      // ==========================================
      // C4: Block 12 (CUBE) — uses %v2_trunc (edge 33→12) and %tensor_c1 (edge 6→12)
      // ==========================================
      %transposed_v2 = linalg.transpose ins(%v2_trunc : tensor<64x64xbf16>) outs(%empty_2d_bf16 : tensor<64x64xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"}
      %matmul_c4 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%transposed_v2, %tensor_c1 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%cube_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // Store output to prevent DCE
      %reint_out = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [64, 64], strides: [%stride_idx, 1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
      %trunc_c4 = arith.truncf %matmul_c4 {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xbf16>
      bufferization.materialize_in_destination %trunc_c4 in writable %reint_out {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : (tensor<64x64xbf16>, memref<64x64xbf16, strided<[?, 1], offset: ?>>) -> ()
    }
    return
  }

  // Test case 3: C3(10) depends on BOTH C1(6)'s tensor.empty() (transpose
  // outs) and C2(8)'s to_tensor (transpose ins).
  // CHECK-LABEL: func.func @kernel_cross_cube_empty(
  // CHECK: %[[EMPTY:.*]] = tensor.empty() {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"}
  // CHECK: } {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE", ssbuffer.load}
  // CHECK: %[[CLONE_EMPTY_CASE:.*]] = scope.scope : () -> tensor<64x64xbf16> {
  // CHECK: memref.alloc() : memref<64x64xbf16>
  // CHECK: memref.copy
  // CHECK: bufferization.to_tensor
  // CHECK: scope.return
  // CHECK: } {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE", ssbuffer.load}
  // CHECK: linalg.transpose ins(%[[CLONE_EMPTY_CASE]] : {{.*}}) outs(%[[EMPTY]] : {{.*}})
  // CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 10 : i32
  // CHECK: arith.subf {{.*}} {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.subBlock = 33 : i32}
  func.func @kernel_cross_cube_empty(
    %arg0: memref<?xbf16> {tt.tensor_kind = 0 : i32},
    %arg1: memref<?xbf16> {tt.tensor_kind = 0 : i32},
    %arg2: i32 {tt.divisibility = 16 : i32}
  ) attributes {global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_f32 = arith.constant 0.000000e+00 : f32
    %cst_scale = arith.constant 0.0883883461 : f32

    %empty_v = tensor.empty() {ssbuffer.block_id = 49 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    %scale = linalg.fill {ssbuffer.block_id = 49 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_scale : f32) outs(%empty_v : tensor<64x64xf32>) -> tensor<64x64xf32>
    %empty_c = tensor.empty() {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
    %cube_init = linalg.fill {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_f32 : f32) outs(%empty_c : tensor<64x64xf32>) -> tensor<64x64xf32>

    scf.for %iv = %c0_i32 to %c64_i32 step %c1_i32 : i32 {
      // C1: block 6 — %empty_c1 reused by C3 (transpose outs)
      %empty_c1 = tensor.empty() {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xbf16>
      %matmul_c1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%empty_c1, %empty_c1 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%cube_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // V1: block 32
      %v1_mul = arith.mulf %matmul_c1, %scale {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %v1_exp = math.exp %v1_mul {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %v1_trunc = arith.truncf %v1_exp {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xbf16>

      // C2: block 8 — %tensor_c2 reused by C3 (transpose ins)
      %tensor_c2 = scope.scope : () -> tensor<64x64xbf16> {
        %alloc_c2 = memref.alloc() : memref<64x64xbf16>
        %src_c2 = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [64, 64], strides: [64, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
        memref.copy %src_c2, %alloc_c2 : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x64xbf16>
        %loaded_c2 = bufferization.to_tensor %alloc_c2 restrict writable : memref<64x64xbf16> to tensor<64x64xbf16>
        scope.return %loaded_c2 : tensor<64x64xbf16>
      } {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE", ssbuffer.load}
      %matmul_c2 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%v1_trunc, %tensor_c2 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%cube_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // C3: block 10 — depends on C1's %empty_c1 (outs) and C2's %tensor_c2 (ins)
      %transposed_c3 = linalg.transpose ins(%tensor_c2 : tensor<64x64xbf16>) outs(%empty_c1 : tensor<64x64xbf16>) permutation = [1, 0] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"}
      %matmul_c3 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%transposed_c3, %transposed_c3 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%cube_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // V2: block 33 — uses %matmul_c3 (from C3) and %v1_exp (from V1)
      %v2_sub = arith.subf %matmul_c3, %v1_exp {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %v2_trunc = arith.truncf %v2_sub {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xbf16>

      // C4: block 12 — uses %v2_trunc (from V2, now V1)
      %matmul_c4 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%v2_trunc, %v2_trunc : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%cube_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // Store output to prevent DCE
      %reint_out = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [64, 64], strides: [1, 1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
      %trunc_c4 = arith.truncf %matmul_c4 {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xbf16>
      bufferization.materialize_in_destination %trunc_c4 in writable %reint_out {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : (tensor<64x64xbf16>, memref<64x64xbf16, strided<[?, 1], offset: ?>>) -> ()
    }
    return
  }
}
