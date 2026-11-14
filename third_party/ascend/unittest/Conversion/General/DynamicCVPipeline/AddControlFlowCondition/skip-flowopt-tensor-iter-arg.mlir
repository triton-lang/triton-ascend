// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// Same-iteration tensor iter-arg pairs cannot add flowopt condition.

// CHECK-LABEL: func.func @skip_flowopt_tensor_iter_arg_pair
// CHECK: arith.cmpi eq
// CHECK: arith.subi
// CHECK: } {{.*}}ssbuffer.if = 0
// CHECK: arith.cmpi eq
// CHECK-NOT: arith.cmpi sge
// CHECK-NOT: arith.ori
// CHECK: arith.addi
// CHECK: } {{.*}}ssbuffer.if = 2
// CHECK: } {{.*}}ssbuffer.if = 1

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @skip_flowopt_tensor_iter_arg_pair(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant 0.000000e+00 : f32

    scope.scope : () -> () {
      %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"]} : memref<16x16xf32, #hivm.address_space<ub>>
      %mem = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %empty = tensor.empty() : tensor<16x16xf32>
      %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>

      scf.for %iv = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%t = %init) -> (tensor<16x16xf32>) : i32 {
        // Block 0: tensor consumer, cross producer of group 0.
        %used = arith.addf %t, %t {ssbuffer.block_id = 0 : i32} : tensor<16x16xf32>
        %p0 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<16x16xf32>
        hivm.hir.copy ins(%p0 : tensor<16x16xf32>) outs(%mem : memref<16x16xf32>) {ssbuffer.block_id = 0 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]}
        %p1 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<16x16xf32>
        hivm.hir.copy ins(%p1 : tensor<16x16xf32>) outs(%mem : memref<16x16xf32>) {ssbuffer.block_id = 0 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]}

        // Block 2: cross consumer of group 1, tensor producer yielded below.
        %empty2 = tensor.empty() {ssbuffer.block_id = 2 : i32} : tensor<16x16xf32>
        %new_t = linalg.fill {ssbuffer.block_id = 2 : i32} ins(%cst : f32) outs(%empty2 : tensor<16x16xf32>) -> tensor<16x16xf32>
        %c1 = tensor.empty() {ssbuffer.block_id = 2 : i32} : tensor<16x16xf32>
        hivm.hir.copy ins(%c1 : tensor<16x16xf32>) outs(%mem : memref<16x16xf32>) {ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [1 : i32, 0 : i32]}
        scf.yield %new_t : tensor<16x16xf32>
      } {ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %mem = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      scf.for %iv = %c0_i32 to %c16_i32 step %c1_i32 : i32 {
        // Block 1: middle hop. Consumes group 0, produces group 1.
        %mid_in = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<16x16xf32>
        hivm.hir.copy ins(%mid_in : tensor<16x16xf32>) outs(%mem : memref<16x16xf32>) {ssbuffer.block_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32]}
        %mid_out = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<16x16xf32>
        hivm.hir.copy ins(%mid_out : tensor<16x16xf32>) outs(%mem : memref<16x16xf32>) {ssbuffer.block_id = 1 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]}
        scf.yield
      } {ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
