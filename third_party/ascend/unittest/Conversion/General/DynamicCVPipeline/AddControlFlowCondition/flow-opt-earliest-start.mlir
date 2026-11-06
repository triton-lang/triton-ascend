// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// The block at depth 3 is a preload target. Its reference is the block two
// positions earlier in the chain. Both starts reach block 4 and the later
// write wins, so the reference is block 1.
// The preload condition is
//   (c_reference - c_target >= buffer_num + 1)
//   OR (total_rounds - c_target <= buffer_num)
// The flowOpt condition of the block is kept, the preload condition is added
// on top of it: the conditions only grow.
// CHECK-LABEL: func.func @flow_opt_earliest_start
// CHECK: scf.while (%{{[^ ]+}} = %{{[^ ,]+}}, %{{[^ ]+}} = %{{[^ ,]+}}, %[[CNT_B:[^ ,]+]] = %{{[^ ,]+}}, %{{[^ ]+}} = %{{[^ ,]+}}, %{{[^ ]+}} = %{{[^ ,]+}}, %[[CNT_TGT:[^ ,]+]] = %{{[^ ,]+}}, %{{[^ ]+}} = %{{[^ ,]+}})
// The flowOpt condition.
// CHECK: arith.cmpi sge, %[[CNT_B]], %{{[^,]+}} : i32
// CHECK: arith.muli %{{[^,]+}}, %c2_i32{{.*}} : i32
// CHECK: arith.addi %{{[^,]+}}, %{{[^,]+}} : i32
// CHECK: arith.cmpi sge, %[[CNT_B]], %{{[^,]+}} : i32
// CHECK: arith.ori
// The preload condition, added on top of the flowOpt condition.
// CHECK: %[[REF:.+]] = memref.load
// CHECK: %[[PD:.+]] = arith.subi %[[REF]], %{{[^,]+}}
// CHECK: arith.cmpi sge, %[[PD]], %c3_i32{{.*}} : i32
// CHECK: arith.subi %{{[^,]+}}, %{{[^,]+}} : i32
// CHECK: arith.subi %{{[^,]+}}, %c2_i32{{.*}} : i32
// CHECK: arith.cmpi sge, %{{[^,]+}}, %{{[^,]+}} : i32
// CHECK: arith.ori
// CHECK: arith.cmpi slt, %[[CNT_TGT]], %{{[^,]+}} : i32
// CHECK: arith.andi
// CHECK: arith.andi
// CHECK: arith.andi
// CHECK: scf.if

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @flow_opt_earliest_start(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32

    scope.scope : () -> () {
      %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      // Present so parsing loads the annotation dialect. Cross-core conditions
      // insert annotation.mark on ssbuf loads; without it the context has not
      // loaded the dialect and triton-opt aborts.
      annotation.mark %alloc {effects = ["write", "read"]} : memref<128xf32, #hivm.address_space<ub>>
      %mem = memref.memory_space_cast %alloc : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>

      scf.for %iv = %c0_i32 to %c16_i32 step %c1_i32 : i32 {
        // ifA: earliest start. Two cross producers (group 0) and three intra
        // producers (group 10).
        %t0 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t0 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 0 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]}
        hivm.hir.copy ins(%t0 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 0 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]}
        hivm.hir.copy ins(%t0 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 0 : i32, ssbuffer.intraDeps = [10 : i32, 1 : i32]}
        hivm.hir.copy ins(%t0 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 0 : i32, ssbuffer.intraDeps = [10 : i32, 1 : i32]}
        hivm.hir.copy ins(%t0 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 0 : i32, ssbuffer.intraDeps = [10 : i32, 1 : i32]}

        // ifB: later start. Cross producer of group 2.
        %t1 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t1 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 1 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32]}

        // ifMidA: consume group 0, produce group 1, consume intra group 10.
        %t2 = tensor.empty() {ssbuffer.block_id = 2 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t2 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32]}
        hivm.hir.copy ins(%t2 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]}
        hivm.hir.copy ins(%t2 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 2 : i32, ssbuffer.intraDeps = [10 : i32, 0 : i32]}

        // ifMidB: consume group 2, produce group 3.
        %t3 = tensor.empty() {ssbuffer.block_id = 3 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t3 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 3 : i32, ssbuffer.crossCoreDeps = [2 : i32, 0 : i32]}
        hivm.hir.copy ins(%t3 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 3 : i32, ssbuffer.crossCoreDeps = [3 : i32, 1 : i32]}

        // ifTarget: depth 3 from both starts. Consumes groups 1 and 3.
        %t4 = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t4 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 4 : i32, ssbuffer.crossCoreDeps = [1 : i32, 0 : i32]}
        hivm.hir.copy ins(%t4 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 4 : i32, ssbuffer.crossCoreDeps = [3 : i32, 0 : i32]}
        scf.yield
      } {ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %mem = memref.memory_space_cast %alloc : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
      scf.for %iv = %c0_i32 to %c16_i32 step %c1_i32 : i32 {
        %tp = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%tp : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 5 : i32, ssbuffer.intraDeps = [20 : i32, 1 : i32]}
        %tc = tensor.empty() {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%tc : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 6 : i32, ssbuffer.intraDeps = [20 : i32, 0 : i32]}
        scf.yield
      } {ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
