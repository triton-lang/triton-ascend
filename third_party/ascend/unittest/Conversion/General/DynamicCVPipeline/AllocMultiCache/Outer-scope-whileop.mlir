// RUN: triton-opt --add_multi_buffer_outer_scope %s | FileCheck %s --implicit-check-not=hivm.cross_buffer_select

// Keep transfer/receiver buffer selection in if/else. Reuse the original
// loop condition for one flag select shared by the matching wait and set.
// CHECK-LABEL: func.func @tc_while_ctov_sender
// CHECK: tightly_coupled_buffer
// CHECK: scf.while {{.*}} : (i32, i32) -> (i32, i32)
// CHECK: arith.remsi
// CHECK: %[[VCOND:.*]] = arith.cmpi eq
// CHECK: %[[VF:.*]] = arith.select %[[VCOND]], {{.*}} : i64
// CHECK-NEXT: {{.*}}sync_block_wait {{.*}} flag = %[[VF]]
// CHECK-NEXT: {{.*}}scf.if %[[VCOND]]
// CHECK: memref.memory_space_cast
// CHECK: } else {
// CHECK: memref.memory_space_cast
// CHECK: ssbuffer.cross_buffer
// CHECK-NEXT: {{.*}}sync_block_set {{.*}} flag = %[[VF]]
// CHECK: ssbuffer.iterCounter
// CHECK: ssbuffer.main_loop
// Initial/final notifications remain static for both buffers.
// CHECK: sync_block_set {{.*}} flag = 3
// CHECK: sync_block_set {{.*}} flag = 7
// CHECK: sync_block_wait {{.*}} flag = 3
// CHECK: sync_block_wait {{.*}} flag = 7
// CHECK: %[[CCOND:.*]] = arith.cmpi eq
// CHECK: %[[CF:.*]] = arith.select %[[CCOND]], {{.*}} : i64
// CHECK-NEXT: {{.*}}sync_block_wait {{.*}} flag = %[[CF]]
// CHECK-NEXT: {{.*}}scf.if %[[CCOND]]
// CHECK: hivm.hir.fixpipe
// CHECK: } else {
// CHECK: hivm.hir.fixpipe
// CHECK: ssbuffer.cross_buffer
// CHECK-NEXT: {{.*}}sync_block_set {{.*}} flag = %[[CF]]

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.inter_core_buf_count = 2 : i32} {
func.func @tc_while_ctov_sender() {
  %c0_i32 = arith.constant 0 : i32
  %c100_i32 = arith.constant 100 : i32
  %c1_i32 = arith.constant 1 : i32
  // --- VECTOR scope (receiver for C→V: memspace_cast reads from ub) ---
  scope.scope : () -> () {
    %buf_ub = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf_ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.while (%iter = %c0_i32) : (i32) -> i32 {
      %cond = arith.cmpi slt, %iter, %c100_i32 : i32
      scf.condition(%cond) %iter : i32
    } do {
    ^bb0(%iter: i32):
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
      %buf = memref.memory_space_cast %buf_ub {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>> to memref<128xf16>
      %t = bufferization.to_tensor %buf restrict writable : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
      %next_iter = arith.addi %iter, %c1_i32 : i32
      scf.yield %next_iter : i32
    } attributes {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
  hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
  // --- CUBE scope (sender for C→V: fixpipe writes to ub) ---
  scope.scope : () -> () {
    %buf_cc = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf32, #hivm.address_space<cc>>
    %buf_ub = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf_ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args() -> () : i32 {
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
      hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} ins(%buf_cc : memref<128xf32, #hivm.address_space<cc>>) outs(%buf_ub : memref<128xf16, #hivm.address_space<ub>>)
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}


// Both cores use while loops and preserve their buffer if/else regions.
// CHECK-LABEL: func.func @tc_while_both_sides
// CHECK: scf.while {{.*}} : (i32, i32) -> (i32, i32)
// CHECK: %[[VCOND:.*]] = arith.cmpi eq
// CHECK: %[[VF:.*]] = arith.select %[[VCOND]], {{.*}} : i64
// CHECK-NEXT: {{.*}}sync_block_wait {{.*}} flag = %[[VF]]
// CHECK-NEXT: {{.*}}scf.if %[[VCOND]]
// CHECK: memref.memory_space_cast
// CHECK: } else {
// CHECK: memref.memory_space_cast
// CHECK: ssbuffer.cross_buffer
// CHECK-NEXT: {{.*}}sync_block_set {{.*}} flag = %[[VF]]
// CHECK: scf.while {{.*}} : (i32, i32) -> (i32, i32)
// CHECK: %[[CCOND:.*]] = arith.cmpi eq
// CHECK: %[[CF:.*]] = arith.select %[[CCOND]], {{.*}} : i64
// CHECK-NEXT: {{.*}}sync_block_wait {{.*}} flag = %[[CF]]
// CHECK-NEXT: {{.*}}scf.if %[[CCOND]]
// CHECK: hivm.hir.fixpipe
// CHECK: } else {
// CHECK: hivm.hir.fixpipe
// CHECK: ssbuffer.cross_buffer
// CHECK-NEXT: {{.*}}sync_block_set {{.*}} flag = %[[CF]]

func.func @tc_while_both_sides() {
  %c0_i32 = arith.constant 0 : i32
  %c100_i32 = arith.constant 100 : i32
  %c1_i32 = arith.constant 1 : i32
  // --- VECTOR scope ---
  scope.scope : () -> () {
    %buf_ub = memref.alloc() {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf_ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.while (%iter = %c0_i32) : (i32) -> i32 {
      %cond = arith.cmpi slt, %iter, %c100_i32 : i32
      scf.condition(%cond) %iter : i32
    } do {
    ^bb0(%iter: i32):
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
      %buf = memref.memory_space_cast %buf_ub {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128xf16, #hivm.address_space<ub>> to memref<128xf16>
      %t = bufferization.to_tensor %buf restrict writable : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 6
      %next_iter = arith.addi %iter, %c1_i32 : i32
      scf.yield %next_iter : i32
    } attributes {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
  hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
  // --- CUBE scope ---
  scope.scope : () -> () {
    %buf_cc = memref.alloc() {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128xf32, #hivm.address_space<cc>>
    %buf_ub = memref.alloc() {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf_ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.while (%iter = %c0_i32) : (i32) -> i32 {
      %cond = arith.cmpi slt, %iter, %c100_i32 : i32
      scf.condition(%cond) %iter : i32
    } do {
    ^bb0(%iter: i32):
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
      hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>, ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2 : i32} ins(%buf_cc : memref<128xf32, #hivm.address_space<cc>>) outs(%buf_ub : memref<128xf16, #hivm.address_space<ub>>)
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
      %next_iter = arith.addi %iter, %c1_i32 : i32
      scf.yield %next_iter : i32
    } attributes {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}
}
