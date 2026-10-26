// RUN: triton-opt --split-input-file --add_multi_buffer_outer_scope %s | FileCheck %s

// Double-buffer polling must preserve the scf.for induction variable type.
// In particular, index has no integer bit width and cannot be passed to
// Type::getIntOrFloatBitWidth(). Both receiver and sender polling are checked.

// CHECK-LABEL: func.func @outer_poll_index
// CHECK: scf.for %[[IV_INDEX_V:[^ ]+]] = {{.*}} step %[[STEP_INDEX_V:[^ ]+]]
// CHECK-NEXT: %[[DIV_INDEX_V:[^ ]+]] = arith.divsi %[[IV_INDEX_V]], %[[STEP_INDEX_V]] {{.*}} : index
// CHECK-NEXT: %[[TWO_INDEX_V:[^ ]+]] = arith.constant {{.*}} 2 : index
// CHECK-NEXT: %[[REM_INDEX_V:[^ ]+]] = arith.remsi %[[DIV_INDEX_V]], %[[TWO_INDEX_V]] {{.*}} : index
// CHECK-NEXT: %[[ZERO_INDEX_V:[^ ]+]] = arith.constant {{.*}} 0 : index
// CHECK-NEXT: %[[COND_INDEX_V:[^ ]+]] = arith.cmpi eq, %[[REM_INDEX_V]], %[[ZERO_INDEX_V]] {{.*}} : index
// CHECK-NEXT: scf.if %[[COND_INDEX_V]]
// CHECK: ssbuffer.cross_buffer = 1 : i32
// CHECK: scf.for %[[IV_INDEX_C:[^ ]+]] = {{.*}} step %[[STEP_INDEX_C:[^ ]+]]
// CHECK-NEXT: %[[DIV_INDEX_C:[^ ]+]] = arith.divsi %[[IV_INDEX_C]], %[[STEP_INDEX_C]] {{.*}} : index
// CHECK-NEXT: %[[TWO_INDEX_C:[^ ]+]] = arith.constant {{.*}} 2 : index
// CHECK-NEXT: %[[REM_INDEX_C:[^ ]+]] = arith.remsi %[[DIV_INDEX_C]], %[[TWO_INDEX_C]] {{.*}} : index
// CHECK-NEXT: %[[ZERO_INDEX_C:[^ ]+]] = arith.constant {{.*}} 0 : index
// CHECK-NEXT: %[[COND_INDEX_C:[^ ]+]] = arith.cmpi eq, %[[REM_INDEX_C]], %[[ZERO_INDEX_C]] {{.*}} : index
// CHECK-NEXT: scf.if %[[COND_INDEX_C]]
// CHECK: ssbuffer.cross_buffer = 1 : i32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.inter_core_buf_count = 2 : i32} {
func.func @outer_poll_index() {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c1 = arith.constant 1 : index
  scope.scope : () -> () {
    %buf_ub = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf_ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.for %iter = %c0 to %c100 step %c1 : index {
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
      %buf = memref.memory_space_cast %buf_ub {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>> to memref<128xf16>
      %t = bufferization.to_tensor %buf restrict writable : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
  hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
  scope.scope : () -> () {
    %buf_cc = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cc>>
    %buf_ub = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf_ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.for %i = %c0 to %c100 step %c1 iter_args() -> () : index {
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
      hivm.hir.fixpipe {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} ins(%buf_cc : memref<128xf16, #hivm.address_space<cc>>) outs(%buf_ub : memref<128xf16, #hivm.address_space<ub>>)
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}


}

// -----

// CHECK-LABEL: func.func @outer_poll_i32
// CHECK: scf.for %[[IV_I32_V:[^ ]+]] = {{.*}} step %[[STEP_I32_V:[^ ]+]]
// CHECK-NEXT: %[[DIV_I32_V:[^ ]+]] = arith.divsi %[[IV_I32_V]], %[[STEP_I32_V]] {{.*}} : i32
// CHECK-NEXT: %[[TWO_I32_V:[^ ]+]] = arith.constant {{.*}} 2 : i32
// CHECK-NEXT: %[[REM_I32_V:[^ ]+]] = arith.remsi %[[DIV_I32_V]], %[[TWO_I32_V]] {{.*}} : i32
// CHECK-NEXT: %[[ZERO_I32_V:[^ ]+]] = arith.constant {{.*}} 0 : i32
// CHECK-NEXT: %[[COND_I32_V:[^ ]+]] = arith.cmpi eq, %[[REM_I32_V]], %[[ZERO_I32_V]] {{.*}} : i32
// CHECK-NEXT: scf.if %[[COND_I32_V]]
// CHECK: ssbuffer.cross_buffer = 1 : i32
// CHECK: scf.for %[[IV_I32_C:[^ ]+]] = {{.*}} step %[[STEP_I32_C:[^ ]+]]
// CHECK-NEXT: %[[DIV_I32_C:[^ ]+]] = arith.divsi %[[IV_I32_C]], %[[STEP_I32_C]] {{.*}} : i32
// CHECK-NEXT: %[[TWO_I32_C:[^ ]+]] = arith.constant {{.*}} 2 : i32
// CHECK-NEXT: %[[REM_I32_C:[^ ]+]] = arith.remsi %[[DIV_I32_C]], %[[TWO_I32_C]] {{.*}} : i32
// CHECK-NEXT: %[[ZERO_I32_C:[^ ]+]] = arith.constant {{.*}} 0 : i32
// CHECK-NEXT: %[[COND_I32_C:[^ ]+]] = arith.cmpi eq, %[[REM_I32_C]], %[[ZERO_I32_C]] {{.*}} : i32
// CHECK-NEXT: scf.if %[[COND_I32_C]]
// CHECK: ssbuffer.cross_buffer = 1 : i32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.inter_core_buf_count = 2 : i32} {
func.func @outer_poll_i32() {
  %c0 = arith.constant 0 : i32
  %c100 = arith.constant 100 : i32
  %c1 = arith.constant 1 : i32
  scope.scope : () -> () {
    %buf_ub = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf_ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.for %iter = %c0 to %c100 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
      %buf = memref.memory_space_cast %buf_ub {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>> to memref<128xf16>
      %t = bufferization.to_tensor %buf restrict writable : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
  hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
  scope.scope : () -> () {
    %buf_cc = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cc>>
    %buf_ub = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf_ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.for %i = %c0 to %c100 step %c1 iter_args() -> () : i32 {
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
      hivm.hir.fixpipe {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} ins(%buf_cc : memref<128xf16, #hivm.address_space<cc>>) outs(%buf_ub : memref<128xf16, #hivm.address_space<ub>>)
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}


}

// -----

// CHECK-LABEL: func.func @outer_poll_i64
// CHECK: scf.for %[[IV_I64_V:[^ ]+]] = {{.*}} step %[[STEP_I64_V:[^ ]+]]
// CHECK-NEXT: %[[DIV_I64_V:[^ ]+]] = arith.divsi %[[IV_I64_V]], %[[STEP_I64_V]] {{.*}} : i64
// CHECK-NEXT: %[[TWO_I64_V:[^ ]+]] = arith.constant {{.*}} 2 : i64
// CHECK-NEXT: %[[REM_I64_V:[^ ]+]] = arith.remsi %[[DIV_I64_V]], %[[TWO_I64_V]] {{.*}} : i64
// CHECK-NEXT: %[[ZERO_I64_V:[^ ]+]] = arith.constant {{.*}} 0 : i64
// CHECK-NEXT: %[[COND_I64_V:[^ ]+]] = arith.cmpi eq, %[[REM_I64_V]], %[[ZERO_I64_V]] {{.*}} : i64
// CHECK-NEXT: scf.if %[[COND_I64_V]]
// CHECK: ssbuffer.cross_buffer = 1 : i32
// CHECK: scf.for %[[IV_I64_C:[^ ]+]] = {{.*}} step %[[STEP_I64_C:[^ ]+]]
// CHECK-NEXT: %[[DIV_I64_C:[^ ]+]] = arith.divsi %[[IV_I64_C]], %[[STEP_I64_C]] {{.*}} : i64
// CHECK-NEXT: %[[TWO_I64_C:[^ ]+]] = arith.constant {{.*}} 2 : i64
// CHECK-NEXT: %[[REM_I64_C:[^ ]+]] = arith.remsi %[[DIV_I64_C]], %[[TWO_I64_C]] {{.*}} : i64
// CHECK-NEXT: %[[ZERO_I64_C:[^ ]+]] = arith.constant {{.*}} 0 : i64
// CHECK-NEXT: %[[COND_I64_C:[^ ]+]] = arith.cmpi eq, %[[REM_I64_C]], %[[ZERO_I64_C]] {{.*}} : i64
// CHECK-NEXT: scf.if %[[COND_I64_C]]
// CHECK: ssbuffer.cross_buffer = 1 : i32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.inter_core_buf_count = 2 : i32} {
func.func @outer_poll_i64() {
  %c0 = arith.constant 0 : i64
  %c100 = arith.constant 100 : i64
  %c1 = arith.constant 1 : i64
  scope.scope : () -> () {
    %buf_ub = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf_ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.for %iter = %c0 to %c100 step %c1 : i64 {
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
      %buf = memref.memory_space_cast %buf_ub {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>> to memref<128xf16>
      %t = bufferization.to_tensor %buf restrict writable : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
  hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
  scope.scope : () -> () {
    %buf_cc = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cc>>
    %buf_ub = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf_ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.for %i = %c0 to %c100 step %c1 iter_args() -> () : i64 {
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
      hivm.hir.fixpipe {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} ins(%buf_cc : memref<128xf16, #hivm.address_space<cc>>) outs(%buf_ub : memref<128xf16, #hivm.address_space<ub>>)
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}


}
