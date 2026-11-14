// RUN: triton-opt --triton-to-linalg --verify-each --split-input-file %s | FileCheck %s
// RUN: sed 's/Ascend950PR_9579/Ascend910B2/g' %s | triton-opt --triton-to-linalg --verify-each --split-input-file | FileCheck %s

// The two tensor pointers are ready before the unrelated scalar read. Their
// descriptors follow that source order, while the reads still execute in the
// original observer, lhs, rhs order. Distinct bases prevent CSE from hiding a
// reversed insertion order or merging the two views.
// CHECK-LABEL: func.func @early_tensor_views(
// CHECK-SAME: %[[LHS:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[RHS:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[OBSERVER:[^ ,]+]]: memref<?xf32>
// CHECK: %[[LEFT_VIEW:.*]] = memref.reinterpret_cast %[[LHS]] to offset: [{{[^]]+}}], sizes: [8], strides: [1]
// CHECK: %[[RIGHT_VIEW:.*]] = memref.reinterpret_cast %[[RHS]] to offset: [{{[^]]+}}], sizes: [8], strides: [1]
// CHECK-NOT: memref.copy
// CHECK: %[[OBSERVED:.*]] = memref.load %[[OBSERVER]]
// CHECK: memref.copy %[[LEFT_VIEW]],
// CHECK: memref.copy %[[RIGHT_VIEW]],
// CHECK: return %[[OBSERVED]] : f32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @early_tensor_views(%lhs: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %rhs: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %observer: !tt.ptr<f32>, %out_lhs: !tt.ptr<f32>, %out_rhs: !tt.ptr<f32>) -> f32 {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %lhs_base = tt.splat %lhs : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %left = tt.addptr %lhs_base, %range : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %rhs_base = tt.splat %rhs : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %right = tt.addptr %rhs_base, %range : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %observed = tt.load %observer : !tt.ptr<f32>
    %a = tt.load %left : tensor<8x!tt.ptr<f32>>
    %b = tt.load %right : tensor<8x!tt.ptr<f32>>
    %out_lhs_base = tt.splat %out_lhs : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %out_lhs_ptr = tt.addptr %out_lhs_base, %range : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %out_rhs_base = tt.splat %out_rhs : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %out_rhs_ptr = tt.addptr %out_rhs_base, %range : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %out_lhs_ptr, %a : tensor<8x!tt.ptr<f32>>
    tt.store %out_rhs_ptr, %b : tensor<8x!tt.ptr<f32>>
    tt.return %observed : f32
  }
}

// -----

// A scalar AddPtr also precedes the observer, but does not opt into tensor
// descriptor placement. Its unknown displacement requires a view.
// CHECK-LABEL: func.func @scalar_addptr_stays_at_use(
// CHECK-SAME: %[[SRC:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[OBSERVER:[^ ,]+]]: memref<?xf32>
// CHECK-NOT: memref.reinterpret_cast %[[SRC]]
// CHECK: %[[OBSERVED:.*]] = memref.load %[[OBSERVER]]
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[SRC]] to offset: [%{{.*}}], sizes: [1], strides: [1]
// CHECK: %[[VALUE:.*]] = memref.load %[[VIEW]]
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @scalar_addptr_stays_at_use(%src: !tt.ptr<f32>, %observer: !tt.ptr<f32>, %offset: i32) -> f32 {
    %ptr = tt.addptr %src, %offset : !tt.ptr<f32>, i32
    %observed = tt.load %observer : !tt.ptr<f32>
    %value = tt.load %ptr : !tt.ptr<f32>
    %result = arith.addf %observed, %value : f32
    tt.return %result : f32
  }
}

// -----

// Block-pointer producers retain use-site materialization. A dynamic advance
// prevents the second pointer from folding to its MakeTensorPtr input; neither
// descriptor may move across the observer merely because it is structured.
// CHECK-LABEL: func.func @block_pointers_stay_at_use(
// CHECK-SAME: %[[DIRECT:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[ADVANCED:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[OBSERVER:[^ ,]+]]: memref<?xf32>
// CHECK-NOT: memref.reinterpret_cast %[[DIRECT]]
// CHECK-NOT: memref.reinterpret_cast %[[ADVANCED]]
// CHECK: %[[OBSERVED:.*]] = memref.load %[[OBSERVER]]
// CHECK: %[[DIRECT_VIEW:.*]] = memref.reinterpret_cast %[[DIRECT]] to offset: [{{[^]]+}}], sizes: [8], strides: [1]
// CHECK: memref.copy %[[DIRECT_VIEW]],
// CHECK: %[[ADVANCED_VIEW:.*]] = memref.reinterpret_cast %[[ADVANCED]] to offset: [{{[^]]+}}], sizes: [8], strides: [1]
// CHECK: memref.copy %[[ADVANCED_VIEW]],
// CHECK: return %[[OBSERVED]] : f32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @block_pointers_stay_at_use(%direct: !tt.ptr<f32>, %advanced: !tt.ptr<f32>, %observer: !tt.ptr<f32>, %out_direct: !tt.ptr<f32>, %out_advanced: !tt.ptr<f32>, %row: i32, %step: i32) -> f32 {
    %parent = arith.constant 64 : i64
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i32
    %p = tt.make_tensor_ptr %direct, [%parent], [%one], [%row] {order = array<i32: 0>} : <tensor<8xf32>>
    %q = tt.make_tensor_ptr %advanced, [%parent], [%one], [%row] {order = array<i32: 0>} : <tensor<8xf32>>
    %next = tt.advance %q, [%step] : <tensor<8xf32>>
    %observed = tt.load %observer : !tt.ptr<f32>
    %a = tt.load %p : !tt.ptr<tensor<8xf32>>
    %b = tt.load %next : !tt.ptr<tensor<8xf32>>
    %out_p = tt.make_tensor_ptr %out_direct, [%parent], [%one], [%zero] {order = array<i32: 0>} : <tensor<8xf32>>
    %out_q = tt.make_tensor_ptr %out_advanced, [%parent], [%one], [%zero] {order = array<i32: 0>} : <tensor<8xf32>>
    tt.store %out_p, %a : !tt.ptr<tensor<8xf32>>
    tt.store %out_q, %b : !tt.ptr<tensor<8xf32>>
    tt.return %observed : f32
  }
}

// -----

// The producer is in the outer block. The branch-local descriptor must stay
// after the observer inside the branch, even though its address is available
// outside. This distinguishes same-block placement from general code motion.
// CHECK-LABEL: func.func @outer_addptr_stays_in_branch(
// CHECK-SAME: %[[SRC:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[OBSERVER:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[OUT:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[COND:[^ ,]+]]: i1
// CHECK-NOT: memref.reinterpret_cast %[[SRC]]
// CHECK: scf.if %[[COND]] {
// CHECK-NOT: memref.reinterpret_cast %[[SRC]]
// CHECK: memref.load %[[OBSERVER]]
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[SRC]] to offset: [{{[^]]+}}], sizes: [8], strides: [1]
// CHECK: memref.copy %[[VIEW]],
// CHECK: bufferization.materialize_in_destination
// CHECK: }
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @outer_addptr_stays_in_branch(%src: !tt.ptr<f32>, %observer: !tt.ptr<f32>, %out: !tt.ptr<f32>, %cond: i1) {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %base = tt.splat %src : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %range : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    scf.if %cond {
      %observed = tt.load %observer : !tt.ptr<f32>
      %value = tt.load %ptr : tensor<8x!tt.ptr<f32>>
      %observed_tile = tt.splat %observed : f32 -> tensor<8xf32>
      %result = arith.addf %value, %observed_tile : tensor<8xf32>
      %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
      %out_ptr = tt.addptr %out_base, %range : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
      tt.store %out_ptr, %result : tensor<8x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// AtomicRMW uses the same ordinary pointer materializer. Its tensor view may
// move to the producer, but the atomic effect must remain after the observer.
// Keep the old value unused so this case needs only the hardware atomic store.
// CHECK-LABEL: func.func @atomic_effect_keeps_its_order(
// CHECK-SAME: %[[COUNTER:[^ ,]+]]: memref<?xi32>
// CHECK-SAME: %[[OBSERVER:[^ ,]+]]: memref<?xi32>
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[COUNTER]] to offset: [{{[^]]+}}], sizes: [8], strides: [1]
// CHECK-NOT: hivm.hir.store
// CHECK: %[[OBSERVED:.*]] = memref.load %[[OBSERVER]]
// CHECK: hivm.hir.store {{.*}}outs(%[[VIEW]]
// CHECK-SAME: atomic = <add>
// CHECK: return %[[OBSERVED]] : i32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @atomic_effect_keeps_its_order(%counter: !tt.ptr<i32>, %observer: !tt.ptr<i32>) -> i32 {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %base = tt.splat %counter : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ptr = tt.addptr %base, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %increment = arith.constant dense<1> : tensor<8xi32>
    %observed = tt.load %observer : !tt.ptr<i32>
    %old = tt.atomic_rmw add, acq_rel, gpu, %ptr, %increment : (tensor<8x!tt.ptr<i32>>, tensor<8xi32>) -> tensor<8xi32>
    tt.return %observed : i32
  }
}

// -----

// The source AddPtr is early, but its negative offset is materialized as an
// index SSA value only at the tensor load. That actual operand does not
// dominate the producer anchor, so the view must remain after the observer.
// Cleanup can hoist the constant itself; the check intentionally observes the
// view versus the read instead of relying on the final constant position.
// CHECK-LABEL: func.func @negative_origin_waits_for_operand(
// CHECK-SAME: %[[SRC:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[OBSERVER:[^ ,]+]]: memref<?xf32>
// CHECK-NOT: memref.reinterpret_cast %[[SRC]]
// CHECK: %[[OBSERVED:.*]] = memref.load %[[OBSERVER]]
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[SRC]] to offset: [{{[^]]+}}], sizes: [8], strides: [1]
// CHECK: memref.copy %[[VIEW]],
// CHECK: return %[[OBSERVED]] : f32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @negative_origin_waits_for_operand(%src: !tt.ptr<f32>, %observer: !tt.ptr<f32>, %out: !tt.ptr<f32>) -> f32 {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %minus_two = arith.constant dense<-2> : tensor<8xi32>
    %offsets = arith.addi %range, %minus_two : tensor<8xi32>
    %base = tt.splat %src : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %offsets : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %observed = tt.load %observer : !tt.ptr<f32>
    %value = tt.load %ptr : tensor<8x!tt.ptr<f32>>
    %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_base, %range : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %out_ptr, %value : tensor<8x!tt.ptr<f32>>
    tt.return %observed : f32
  }
}
