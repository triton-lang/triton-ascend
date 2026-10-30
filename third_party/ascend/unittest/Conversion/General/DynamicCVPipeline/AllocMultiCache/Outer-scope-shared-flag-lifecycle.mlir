// RUN: triton-opt --split-input-file --add_multi_buffer_outer_scope %s | FileCheck %s

// CHECK-LABEL: func.func @shared_v2c
// The shared seed is tagged tid1, but covers both tid0 and tid1.
// CHECK: sync_block_set {{.*}} flag = 0
// CHECK-NEXT: {{.*}}sync_block_set {{.*}} flag = 1
// CHECK: scf.for
// CHECK: %[[FLAG0:[^ ]+]] = arith.select %[[COND0:[^, ]+]], {{.*}} : i64
// CHECK-NEXT: {{.*}}sync_block_wait {{.*}} flag = %[[FLAG0]]
// CHECK-NEXT: {{.*}} = scf.if %[[COND0]] -> (tensor<128xf16>)
// CHECK: sync_block_set {{.*}} flag = %[[FLAG0]]
// CHECK: scf.for
// CHECK: %[[FLAG1:[^ ]+]] = arith.select %[[COND1:[^, ]+]], {{.*}} : i64
// CHECK-NEXT: {{.*}}sync_block_wait {{.*}} flag = %[[FLAG1]]
// CHECK-NEXT: {{.*}} = scf.if %[[COND1]] -> (tensor<128xf16>)
// CHECK: sync_block_set {{.*}} flag = %[[FLAG1]]
// CHECK: } {ssbuffer.main_loop = 0 : i32}
// CHECK: } {ssbuffer.main_loop = 1 : i32}
// The sender drain is duplicated once for the shared output flag.
// CHECK: sync_block_wait {{.*}} flag = 0
// CHECK-NEXT: {{.*}}sync_block_wait {{.*}} flag = 1
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.inter_core_buf_count = 2 : i32} {
func.func @shared_v2c() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  scope.scope : () -> () {
    %buf0 = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    annotation.mark %buf0 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    %buf1 = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    annotation.mark %buf1 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
      %cast = memref.memory_space_cast %buf0 {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<cbuf>> to memref<128xf16>
      %t = bufferization.to_tensor %cast restrict writable {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    } {ssbuffer.main_loop = 0 : i32}
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
      %cast = memref.memory_space_cast %buf1 {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>> to memref<128xf16>
      %t = bufferization.to_tensor %cast restrict writable {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    } {ssbuffer.main_loop = 1 : i32}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  scope.scope : () -> () {
    %buf0 = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    annotation.mark %buf0 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    %buf1 = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    annotation.mark %buf1 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    %empty = tensor.empty() : tensor<128xf16>
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      hivm.hir.copy ins(%empty : tensor<128xf16>) outs(%buf0 : memref<128xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}
      hivm.hir.sync_block_set {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    } {ssbuffer.main_loop = 0 : i32}
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      hivm.hir.copy ins(%empty : tensor<128xf16>) outs(%buf1 : memref<128xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}
      hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    } {ssbuffer.main_loop = 1 : i32}
    hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}
}

// -----

// CHECK-LABEL: func.func @shared_c2v
// The shared seed is tagged tid1, but covers both tid0 and tid1.
// CHECK: sync_block_set {{.*}} flag = 0
// CHECK-NEXT: {{.*}}sync_block_set {{.*}} flag = 1
// CHECK: scf.for
// CHECK: %[[FLAG0:[^ ]+]] = arith.select %[[COND0:[^, ]+]], {{.*}} : i64
// CHECK-NEXT: {{.*}}sync_block_wait {{.*}} flag = %[[FLAG0]]
// CHECK-NEXT: {{.*}} = scf.if %[[COND0]] -> (tensor<128xf16>)
// CHECK: sync_block_set {{.*}} flag = %[[FLAG0]]
// CHECK: scf.for
// CHECK: %[[FLAG1:[^ ]+]] = arith.select %[[COND1:[^, ]+]], {{.*}} : i64
// CHECK-NEXT: {{.*}}sync_block_wait {{.*}} flag = %[[FLAG1]]
// CHECK-NEXT: {{.*}} = scf.if %[[COND1]] -> (tensor<128xf16>)
// CHECK: sync_block_set {{.*}} flag = %[[FLAG1]]
// CHECK: } {ssbuffer.main_loop = 0 : i32}
// CHECK: } {ssbuffer.main_loop = 1 : i32}
// The sender drain is duplicated once for the shared output flag.
// CHECK: sync_block_wait {{.*}} flag = 0
// CHECK-NEXT: {{.*}}sync_block_wait {{.*}} flag = 1
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.inter_core_buf_count = 2 : i32} {
func.func @shared_c2v() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  scope.scope : () -> () {
    %buf0 = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf0 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<ub>>
    %buf1 = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf1 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      %cast = memref.memory_space_cast %buf0 {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<ub>> to memref<128xf16>
      %t = bufferization.to_tensor %cast restrict writable {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    } {ssbuffer.main_loop = 0 : i32}
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      %cast = memref.memory_space_cast %buf1 {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>> to memref<128xf16>
      %t = bufferization.to_tensor %cast restrict writable {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    } {ssbuffer.main_loop = 1 : i32}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  scope.scope : () -> () {
    %buf0 = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf0 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<ub>>
    %buf1 = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf1 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    %cc = memref.alloc() : memref<128xf16, #hivm.address_space<cc>>
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
      hivm.hir.fixpipe {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} ins(%cc : memref<128xf16, #hivm.address_space<cc>>) outs(%buf0 : memref<128xf16, #hivm.address_space<ub>>)
      hivm.hir.sync_block_set {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    } {ssbuffer.main_loop = 0 : i32}
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
      hivm.hir.fixpipe {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} ins(%cc : memref<128xf16, #hivm.address_space<cc>>) outs(%buf1 : memref<128xf16, #hivm.address_space<ub>>)
      hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    } {ssbuffer.main_loop = 1 : i32}
    hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}
}

// -----

// Missing or non-enclosing lifecycle must report SSBUFFER failure.
// CHECK: module attributes {{.*}}triton_ascend.dynamic_cv_pipeline.rc = 1 : i32
// CHECK-LABEL: func.func @no_shared_lifecycle_missing
// CHECK-NOT: ssbuffer.cross_buffer
// CHECK-NOT: flag = 1
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.inter_core_buf_count = 2 : i32} {
func.func @no_shared_lifecycle_missing() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  scope.scope : () -> () {
    %buf0 = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    annotation.mark %buf0 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    %buf1 = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    annotation.mark %buf1 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
      %cast = memref.memory_space_cast %buf0 {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<cbuf>> to memref<128xf16>
      %t = bufferization.to_tensor %cast restrict writable {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    } {ssbuffer.main_loop = 0 : i32}
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
      %cast = memref.memory_space_cast %buf1 {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>> to memref<128xf16>
      %t = bufferization.to_tensor %cast restrict writable {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    } {ssbuffer.main_loop = 1 : i32}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  scope.scope : () -> () {
    %buf0 = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    annotation.mark %buf0 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    %buf1 = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    annotation.mark %buf1 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    %empty = tensor.empty() : tensor<128xf16>
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      hivm.hir.copy ins(%empty : tensor<128xf16>) outs(%buf0 : memref<128xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}
      hivm.hir.sync_block_set {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    } {ssbuffer.main_loop = 0 : i32}
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      hivm.hir.copy ins(%empty : tensor<128xf16>) outs(%buf1 : memref<128xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}
      hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    } {ssbuffer.main_loop = 1 : i32}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}
}

// -----

// Missing or non-enclosing lifecycle must report SSBUFFER failure.
// CHECK: module attributes {{.*}}triton_ascend.dynamic_cv_pipeline.rc = 1 : i32
// CHECK-LABEL: func.func @no_shared_lifecycle_late
// CHECK-NOT: ssbuffer.cross_buffer
// CHECK-NOT: flag = 1
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.inter_core_buf_count = 2 : i32} {
func.func @no_shared_lifecycle_late() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  scope.scope : () -> () {
    %buf0 = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    annotation.mark %buf0 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    %buf1 = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    annotation.mark %buf1 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
      %cast = memref.memory_space_cast %buf0 {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<cbuf>> to memref<128xf16>
      %t = bufferization.to_tensor %cast restrict writable {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    } {ssbuffer.main_loop = 0 : i32}
    hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
      %cast = memref.memory_space_cast %buf1 {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>> to memref<128xf16>
      %t = bufferization.to_tensor %cast restrict writable {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    } {ssbuffer.main_loop = 1 : i32}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  scope.scope : () -> () {
    %buf0 = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    annotation.mark %buf0 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    %buf1 = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    annotation.mark %buf1 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    %empty = tensor.empty() : tensor<128xf16>
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      hivm.hir.copy ins(%empty : tensor<128xf16>) outs(%buf0 : memref<128xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}
      hivm.hir.sync_block_set {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    } {ssbuffer.main_loop = 0 : i32}
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      hivm.hir.copy ins(%empty : tensor<128xf16>) outs(%buf1 : memref<128xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}
      hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    } {ssbuffer.main_loop = 1 : i32}
    hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}
}
