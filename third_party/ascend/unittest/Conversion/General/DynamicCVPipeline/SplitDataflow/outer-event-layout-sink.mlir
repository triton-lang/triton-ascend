// RUN: triton-opt --split-input-file --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync %s | FileCheck %s

// The active partial/full branches get their own pure layout chain. The
// initialized no-event placeholder is converted too, so the unconditional
// V->C transfer and its credits remain outside the payload predicate.
// CHECK-NOT: triton_ascend.dynamic_cv_pipeline.rc
// CHECK-LABEL: func.func @nested_event
// CHECK: %[[OUTER:[a-zA-Z0-9_]+]]:2 = scf.if {{.*}} -> (tensor<2x2x16x16xbf16>, i32)
// CHECK: %[[INNER:[a-zA-Z0-9_]+]] = scf.if {{.*}} -> (tensor<2x2x16x16xbf16>)
// CHECK: arith.truncf
// CHECK: tensor.reshape
// CHECK: linalg.transpose
// CHECK: scf.yield {{.*}} : tensor<2x2x16x16xbf16>
// CHECK: } else {
// CHECK: arith.truncf
// CHECK: tensor.reshape
// CHECK: linalg.transpose
// CHECK: scf.yield {{.*}} : tensor<2x2x16x16xbf16>
// CHECK: scf.yield {{.*}} %[[INNER]], {{.*}} : tensor<2x2x16x16xbf16>, i32
// CHECK: } else {
// CHECK: linalg.fill
// CHECK: tensor.reshape
// CHECK: linalg.transpose
// CHECK: scf.yield {{.*}} : tensor<2x2x16x16xbf16>, i32
// CHECK: } {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR, VECTOR"}
// CHECK-NOT: tensor.reshape
// CHECK: hivm.hir.sync_block_wait
// CHECK: hivm.hir.copy ins(%[[OUTER]]#0 : tensor<2x2x16x16xbf16>) outs({{.*}}) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.crossCoreDeps = [{{.*}}1 : i32]
// CHECK: hivm.hir.sync_block_set
// CHECK: linalg.matmul
module attributes {ssbuffer.outer_main_loop, hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @nested_event(%n: index, %valid: i1, %partial: i1, %x: tensor<32x32xf32>, %b: tensor<32x32xbf16>, %acc: tensor<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tag = arith.constant 7 : i32
    %zero = arith.constant 0.0 : bf16
    scf.for %i = %c0 to %n step %c1 {
      %p:2 = scf.if %valid -> (tensor<32x32xbf16>, i32) {
        %branch = scf.if %partial -> tensor<32x32xbf16> {
          %e = math.exp %x {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
          %v = arith.truncf %e {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> to tensor<32x32xbf16>
          scf.yield %v : tensor<32x32xbf16>
        } else {
          %e = math.exp %x {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
          %v = arith.truncf %e {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> to tensor<32x32xbf16>
          scf.yield %v : tensor<32x32xbf16>
        } {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
        scf.yield %branch, %tag : tensor<32x32xbf16>, i32
      } else {
        %empty = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<32x32xbf16>
        %fill = linalg.fill {ssbuffer.block_id = 1 : i32} ins(%zero : bf16) outs(%empty : tensor<32x32xbf16>) -> tensor<32x32xbf16>
        scf.yield %fill, %tag : tensor<32x32xbf16>, i32
      } {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR, VECTOR"}
      %mm = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%p#0, %b : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
    } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
    return
  }
}

// -----

// A bare tensor.empty placeholder is not a legal initialized branch.
// CHECK-NOT: triton_ascend.dynamic_cv_pipeline.rc
// CHECK-LABEL: func.func @empty_placeholder
// CHECK: %[[P:[a-zA-Z0-9_]+]] = scf.if {{.*}} -> (tensor<32x32xbf16>)
// CHECK-NOT: tensor.reshape
// CHECK: } {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: tensor.reshape %[[P]]
// CHECK: linalg.transpose
// CHECK: hivm.hir.copy
module attributes {ssbuffer.outer_main_loop, hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @empty_placeholder(%n: index, %valid: i1, %x: tensor<32x32xf32>, %b: tensor<32x32xbf16>, %acc: tensor<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : bf16
    scf.for %i = %c0 to %n step %c1 {
      %p = scf.if %valid -> tensor<32x32xbf16> {
        %v = arith.truncf %x {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> to tensor<32x32xbf16>
        scf.yield %v : tensor<32x32xbf16>
      } else {
        %empty = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<32x32xbf16>
        scf.yield %empty : tensor<32x32xbf16>
      } {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
      %mm = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%p, %b : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
    } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
    return
  }
}

// -----

// An additional ND user must keep the original if result type.
// CHECK-NOT: triton_ascend.dynamic_cv_pipeline.rc
// CHECK-LABEL: func.func @nd_vector_user
// CHECK: %[[P:[a-zA-Z0-9_]+]] = scf.if {{.*}} -> (tensor<32x32xbf16>)
// CHECK-NOT: tensor.reshape
// CHECK: } {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: tensor.reshape %[[P]]
// CHECK: linalg.transpose
// CHECK: hivm.hir.copy
module attributes {ssbuffer.outer_main_loop, hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @nd_vector_user(%n: index, %valid: i1, %x: tensor<32x32xf32>, %b: tensor<32x32xbf16>, %acc: tensor<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : bf16
    scf.for %i = %c0 to %n step %c1 {
      %p = scf.if %valid -> tensor<32x32xbf16> {
        %v = arith.truncf %x {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> to tensor<32x32xbf16>
        scf.yield %v : tensor<32x32xbf16>
      } else {
        %empty = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<32x32xbf16>
        %fill = linalg.fill {ssbuffer.block_id = 1 : i32} ins(%zero : bf16) outs(%empty : tensor<32x32xbf16>) -> tensor<32x32xbf16>
        scf.yield %fill : tensor<32x32xbf16>
      } {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
      %nd = arith.addf %p, %p {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xbf16>
      %mm = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%p, %b : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
    } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
    return
  }
}

// -----

// Ordinary loops keep their existing layout and fusion behavior.
// CHECK-NOT: triton_ascend.dynamic_cv_pipeline.rc
// CHECK-LABEL: func.func @legacy_loop
// CHECK: %[[P:[a-zA-Z0-9_]+]] = scf.if {{.*}} -> (tensor<32x32xbf16>)
// CHECK-NOT: tensor.reshape
// CHECK: } {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: tensor.reshape %[[P]]
// CHECK: linalg.transpose
// CHECK: hivm.hir.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @legacy_loop(%n: index, %valid: i1, %x: tensor<32x32xf32>, %b: tensor<32x32xbf16>, %acc: tensor<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : bf16
    scf.for %i = %c0 to %n step %c1 {
      %p = scf.if %valid -> tensor<32x32xbf16> {
        %v = arith.truncf %x {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> to tensor<32x32xbf16>
        scf.yield %v : tensor<32x32xbf16>
      } else {
        %empty = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<32x32xbf16>
        %fill = linalg.fill {ssbuffer.block_id = 1 : i32} ins(%zero : bf16) outs(%empty : tensor<32x32xbf16>) -> tensor<32x32xbf16>
        scf.yield %fill : tensor<32x32xbf16>
      } {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
      %mm = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%p, %b : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
    }
    return
  }
}
