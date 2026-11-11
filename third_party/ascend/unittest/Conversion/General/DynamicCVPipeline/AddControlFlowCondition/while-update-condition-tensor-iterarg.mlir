// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// UpdateConditionInfo on scf.while: tensor-typed iter_arg control vars.
// while carries (tensor, iv); block 5 consumes the tensor, block 6 produces a
// new tensor for yield. UpdateLoopOps appends i32=1 control args; conditions:
//   consumer: control_var == 1, then -1
//   producer: control_var == 0, then +1
// (same semantics as for-path iter_args_deps_add_conditions)

// CHECK-LABEL: func.func @while_update_condition_tensor_iterarg

// Tensor-iter control arg is seeded with 1 and appears on the while.
// CHECK: %[[TINIT:.*]] = arith.constant 1 : i32
// CHECK: scf.while
// CHECK-SAME: %[[TINIT]]

// The flag of the loop starts as the constant zero at the head of the loop
// body: it is the false value of the select that computes the flag.
// CHECK: ^bb0
// CHECK: %[[ZERO5:.*]] = arith.constant 0 : i32

// Consumer if=5: after cross sgt, eq 1 is and-ed into cond; then -1 inside if.
// The block 5 is on the second level of the vector core, it has no preload
// condition.
// Bind ONE only after sgt so it is the tensor eq constant, not the cross-core one.
// CHECK: arith.cmpi sgt
// CHECK: arith.andi
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: arith.cmpi eq, %{{.*}}, %[[ONE]] : i32
// The cross-core condition and the tensor condition are and-ed together, then
// the counter condition is added.
// CHECK: arith.andi
// CHECK: %[[COND5:.*]] = arith.andi
// The block is the first one of the loop: the flag it reads is the constant
// zero the loop starts with, its check is left out.
// CHECK: scf.if %[[COND5]]
// CHECK: arith.subi
// CHECK: } {{.*}}ssbuffer.if = 5
// The flag of the loop is not yielded by the block: it is computed with a
// select on the condition of the block, after the block.
// CHECK: %[[ONE5:.*]] = arith.constant 1 : i32
// CHECK: %[[FLAG5:.*]] = arith.select %[[COND5]], %[[ONE5]], %[[ZERO5]] : i32

// Producer if=6: after cross slt, eq 0 is and-ed into cond; then +1 inside if.
// The block 6 is on the first level of the vector core, the block 5 is on the
// second one: the preload condition c_block6 - c_block5 <= 2 (2 - 1 + (2 - 1))
// is and-ed in as well.
// CHECK: arith.cmpi slt
// CHECK: arith.andi
// The preload condition: c_block6 - c_block5 <= 2 (2 - 1 + (2 - 1)).
// CHECK: %[[DIFF_PRE6:.*]] = arith.subi
// CHECK: %[[SLE6:.*]] = arith.cmpi sle, %[[DIFF_PRE6]], %{{.*}} : i32
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: arith.cmpi eq, %{{.*}}, %[[ZERO]] : i32
// The cross-core condition, the preload condition and the tensor condition
// are and-ed together, then the counter condition is added.
// CHECK: %[[PRE_AND6:.*]] = arith.andi %{{.*}}, %[[SLE6]]
// CHECK: %[[TAND6:.*]] = arith.andi %[[PRE_AND6]], %{{.*}}
// CHECK: %[[COND6:.*]] = arith.andi %[[TAND6]], %{{.*}}
// The flag of the previous block gates this one.
// CHECK: %[[FLAGCMP6:.*]] = arith.cmpi eq, %[[FLAG5]], %{{.*}} : i32
// CHECK: %[[AND6F:.*]] = arith.andi %[[COND6]], %[[FLAGCMP6]]
// CHECK: scf.if %[[AND6F]]
// CHECK: arith.addi
// CHECK: } {{.*}}ssbuffer.if = 6

// CHECK: scf.yield

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @while_update_condition_tensor_iterarg(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %bound = arith.extsi %arg7 : i32 to i64

    scope.scope : () -> () {
      %empty0 = tensor.empty() : tensor<16x16xf32>
      %init_t = linalg.fill ins(%cst : f32) outs(%empty0 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %alloc = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_ub = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>

      %0:2 = scf.while (%arg14 = %init_t, %arg15 = %c0_i32) : (tensor<16x16xf32>, i32) -> (tensor<16x16xf32>, i32) {
        %1 = arith.extsi %arg15 {Undefined, ssbuffer.block_id = 4 : i32} : i32 to i64
        %2 = arith.cmpi slt, %1, %bound {Undefined, ssbuffer.block_id = 4 : i32} : i64
        scf.condition(%2) %arg14, %arg15 : tensor<16x16xf32>, i32
      } do {
      ^bb0(%arg14: tensor<16x16xf32>, %arg15: i32):
        // Consumer of tensor iter_arg (non-yield use) + cross-core input.
        hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %sum = arith.addf %arg14, %arg14 {ssbuffer.block_id = 5 : i32} : tensor<16x16xf32>
        %memspacecast = memref.memory_space_cast %alloc_ub {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>

        // Producer of tensor iter_arg (loop yield comes from this block) + cross-core output.
        // CreateIfOps else-yields %arg14 → analyzeTensorIterArgDependencies marks producer.
        hivm.hir.sync_block_set {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %empty1 = tensor.empty() {ssbuffer.block_id = 6 : i32} : tensor<16x16xf32>
        %new_t = linalg.fill {ssbuffer.block_id = 6 : i32} ins(%cst : f32) outs(%empty1 : tensor<16x16xf32>) -> tensor<16x16xf32>
        %cbuf_t = tensor.empty() {ssbuffer.block_id = 6 : i32} : tensor<8x8x16x16xf16>
        hivm.hir.copy ins(%cbuf_t : tensor<8x8x16x16xf16>) outs(%alloc : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]}
        %next6 = arith.addi %arg15, %c16_i32 {ssbuffer.block_id = 6 : i32} : i32
        scf.yield %new_t, %next6 : tensor<16x16xf32>, i32
      } attributes {Undefined, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %alloc_c = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_c {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_ub = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>

      %0 = scf.while (%arg14 = %c0_i32) : (i32) -> i32 {
        %1 = arith.extsi %arg14 {Undefined, ssbuffer.block_id = 4 : i32} : i32 to i64
        %2 = arith.cmpi slt, %1, %bound {Undefined, ssbuffer.block_id = 4 : i32} : i64
        scf.condition(%2) %arg14 : i32
      } do {
      ^bb0(%arg14: i32):
        hivm.hir.sync_block_wait {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %t0 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} ins(%t0 : tensor<128x128xf32>) outs(%alloc_ub : memref<128x128xf32, #hivm.address_space<ub>>)
        %next0 = arith.addi %arg14, %c16_i32 {ssbuffer.block_id = 0 : i32} : i32

        hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %conv = hivm.hir.convert_layout %alloc_c output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32]} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
        %next1 = arith.addi %arg14, %c16_i32 {ssbuffer.block_id = 1 : i32} : i32
        scf.yield %next1 : i32
      } attributes {Undefined, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
