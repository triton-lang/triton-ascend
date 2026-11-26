// RUN: triton-opt --normalize-outer-main-loop %s | FileCheck %s
// RUN: sed 's|// PRECISION: ||' %s | triton-opt --normalize-outer-main-loop | FileCheck %s --check-prefix=RETAIN
// RUN: sed 's|// ALIAS: ||' %s | triton-opt --normalize-outer-main-loop | FileCheck %s --check-prefix=RETAIN
// RUN: sed '/%[p]:3 = scf.for/,/%nf =/s/outs(%zero_init :/outs(%a :/' %s | triton-opt --normalize-outer-main-loop | FileCheck %s --check-prefix=RETAIN
// RUN: sed -e 's|// EARLY-MASK: ||' -e '/test.late_mask/s|^|//|' %s | triton-opt --normalize-outer-main-loop | FileCheck %s --check-prefix=MIXED

// Share QK/PV without changing arithmetic, repeated operand positions or zero
// initializers. Cache the partial count in the fifth cursor field.
// CHECK-LABEL: func.func @join_partial_full
// CHECK: arith.extui {{.*}} : i32 to i64
// CHECK: arith.extui {{.*}} : i32 to i64
// CHECK: scf.for {{.*}} : i64
// CHECK: arith.trunci {{.*}} : i64 to i32
// CHECK: memref.alloc
// CHECK: bufferization.to_tensor
// CHECK: scf.yield
// CHECK: linalg.matmul
// CHECK-NEXT: {{.*}}arith.mulf {{.*}} {test.qk_scale}
// CHECK: arith.sitofp
// CHECK: %[[PV_INIT:.*]] = linalg.fill {test.pv_init}
// CHECK-NEXT: {{.*}}linalg.matmul {{.*}}outs(%[[PV_INIT]] : tensor<4x4xf32>)
// CHECK-NEXT: {{.*}}arith.addf {{.*}} {test.pv_tail}
// CHECK-NOT: linalg.matmul
// CHECK-NOT: test.pv_tail
// CHECK: ssbuffer.explicit_main_loop
// CHECK-SAME: ssbuffer.outer_event_control_count = 5 : i32
// CHECK: ssbuffer.outer_event_fallback

// Different matmul properties, a memref alias crossing QK, or mismatched
// accumulator sources must retain the original nest without event state.
// RETAIN-LABEL: func.func @join_partial_full
// RETAIN: scf.for
// RETAIN: scf.for
// RETAIN: linalg.matmul
// RETAIN: linalg.matmul
// RETAIN: scf.for
// RETAIN: linalg.matmul
// RETAIN: linalg.matmul
// RETAIN-NOT: ssbuffer.
// RETAIN: return
func.func @join_partial_full(%begin: index, %end: index, %step: index,
                             %partial: memref<?xi32>, %full: memref<?xi32>,
                             %source: memref<4x4xf32>, %q: tensor<4x4xf32>,
                             %seed: tensor<4x4xf32>, %out: memref<4x4xf32>) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %zero_index = arith.constant 0 : index
  %zero_f32 = arith.constant 0.0 : f32
  scf.for %task = %begin to %end step %step {
    %np = memref.load %partial[%task] : memref<?xi32>
    %p:3 = scf.for %j = %zero to %np step %one
        iter_args(%m = %seed, %l = %seed, %a = %seed)
        -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) : i32 {
      %scratch = memref.alloc() : memref<4x4xf32>
      memref.copy %source, %scratch : memref<4x4xf32> to memref<4x4xf32>
      %v = bufferization.to_tensor %scratch restrict writable : memref<4x4xf32> to tensor<4x4xf32>
      // ALIAS: %view = memref.cast %scratch : memref<4x4xf32> to memref<?x4xf32>
      %iv = arith.addi %j, %one : i32
      %qk = linalg.matmul ins(%q, %q : tensor<4x4xf32>, tensor<4x4xf32>)
          outs(%seed : tensor<4x4xf32>) -> tensor<4x4xf32>
      // ALIAS: %x = memref.load %view[%zero_index, %zero_index] : memref<?x4xf32>
      %scaled_qk = arith.mulf %qk, %q {test.qk_scale} : tensor<4x4xf32>
      %scale = arith.sitofp %iv : i32 to f32
      %factor = linalg.fill ins(%scale : f32) outs(%seed : tensor<4x4xf32>) -> tensor<4x4xf32>
      %next_m = arith.addf %m, %factor : tensor<4x4xf32>
      %next_l = arith.mulf %l, %scaled_qk : tensor<4x4xf32>
      %scaled_a = arith.mulf %a, %factor : tensor<4x4xf32>
      %empty = tensor.empty() : tensor<4x4xf32>
      %zero_init = linalg.fill {test.pv_init} ins(%zero_f32 : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
      %pv = linalg.matmul ins(%scaled_qk, %v : tensor<4x4xf32>, tensor<4x4xf32>)
          outs(%zero_init : tensor<4x4xf32>) -> tensor<4x4xf32>
      %next_a = arith.addf %pv, %scaled_a {test.pv_tail} : tensor<4x4xf32>
      scf.yield %next_m, %next_l, %next_a : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
    }
    %nf = memref.load %full[%task] : memref<?xi32>
    %f:3 = scf.for %j = %zero to %nf step %one
        iter_args(%m = %p#0, %l = %p#1, %a = %p#2)
        -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) : i32 {
      %scratch = memref.alloc() : memref<4x4xf32>
      memref.copy %source, %scratch : memref<4x4xf32> to memref<4x4xf32>
      %v = bufferization.to_tensor %scratch restrict writable : memref<4x4xf32> to tensor<4x4xf32>
      %qk = linalg.matmul
          // PRECISION: {input_precision = "tf32"}
          ins(%q, %v : tensor<4x4xf32>, tensor<4x4xf32>)
          outs(%seed : tensor<4x4xf32>) -> tensor<4x4xf32>
      %scaled_qk = arith.mulf %qk, %q {test.qk_scale} : tensor<4x4xf32>
      %scale = arith.sitofp %j : i32 to f32
      %factor = linalg.fill ins(%scale : f32) outs(%seed : tensor<4x4xf32>) -> tensor<4x4xf32>
      %next_m = arith.mulf %m, %factor : tensor<4x4xf32>
      %next_l = arith.addf %l, %scaled_qk : tensor<4x4xf32>
      %scaled_a = arith.mulf %a, %factor : tensor<4x4xf32>
      %empty = tensor.empty() : tensor<4x4xf32>
      %zero_init = linalg.fill {test.pv_init} ins(%zero_f32 : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
      %pv = linalg.matmul ins(%scaled_qk, %v : tensor<4x4xf32>, tensor<4x4xf32>)
          outs(%zero_init : tensor<4x4xf32>) -> tensor<4x4xf32>
      %next_a = arith.addf %pv, %scaled_a {test.pv_tail} : tensor<4x4xf32>
      scf.yield %next_m, %next_l, %next_a : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
    }
    bufferization.materialize_in_destination %f#2 in writable %out : (tensor<4x4xf32>, memref<4x4xf32>) -> ()
  } {ssbuffer.outer_main_loop}
  return
}

// Select divergent scalar metadata before the partial-only mask, then share
// the private allocation/copy. A tensor prefix instead keeps the mixed branch.
// CHECK-LABEL: func.func @share_private_load
// CHECK: %[[KV:.*]] = scf.if {{.*}} -> (i32) {
// CHECK: memref.load
// CHECK-NOT: arith.mulf
// CHECK: } else {
// CHECK: memref.load
// CHECK: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: {{.*}}scf.if {{.*}} -> (tensor<4x4xf32>) {
// CHECK-NEXT: {{.*}}arith.mulf
// CHECK-NEXT: scf.yield
// CHECK-NEXT: } else {
// CHECK-NEXT: {{.*}}tensor.empty
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: {{.*}}arith.index_cast %[[KV]] : i32 to index
// CHECK-NEXT: {{.*}}memref.subview
// CHECK-NEXT: %[[PRIVATE:.*]] = memref.alloc
// CHECK-NEXT: memref.copy {{.*}}, %[[PRIVATE]]
// CHECK-NEXT: %[[LOADED:.*]] = bufferization.to_tensor %[[PRIVATE]]
// CHECK-NEXT: {{.*}}linalg.matmul ins({{.*}}, %[[LOADED]] :
// CHECK: linalg.matmul ins({{.*}}, %[[LOADED]] :
// CHECK-NOT: memref.alloc
// CHECK: ssbuffer.explicit_main_loop
// CHECK-SAME: ssbuffer.outer_event_control_count = 5 : i32
// MIXED-LABEL: func.func @share_private_load
// MIXED: scf.if {{.*}} -> (i32, tensor<4x4xf32>) {
// MIXED-NEXT: {{.*}}arith.mulf
// MIXED-NEXT: {{.*}}arith.index_cast
// MIXED-NEXT: {{.*}}memref.load
// MIXED: } else {
// MIXED: memref.load
// MIXED: scf.yield
// MIXED-NEXT: }
// MIXED-NEXT: {{.*}}arith.index_cast
// MIXED-NEXT: {{.*}}memref.subview
// MIXED-NEXT: {{.*}}memref.alloc
// MIXED-NEXT: memref.copy
// MIXED-NEXT: {{.*}}bufferization.to_tensor
// MIXED-NEXT: {{.*}}linalg.matmul
// MIXED: linalg.matmul
// MIXED: ssbuffer.explicit_main_loop
// MIXED-SAME: ssbuffer.outer_event_control_count = 5 : i32
func.func @share_private_load(%end: index, %np: i32, %nf: i32,
                              %partial_ids: memref<?xi32>, %full_ids: memref<?xi32>,
                              %source: memref<?x4xf32>, %q: tensor<4x4xf32>,
                              %out: memref<4x4xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %zero_i32 = arith.constant 0 : i32
  %one_i32 = arith.constant 1 : i32
  %zero_f32 = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x4xf32>
  %seed = linalg.fill ins(%zero_f32 : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  scf.for %task = %zero to %end step %one {
    %p = scf.for %j = %zero_i32 to %np step %one_i32 iter_args(%a = %seed) -> tensor<4x4xf32> : i32 {
      // EARLY-MASK: %mask = arith.mulf %q, %q : tensor<4x4xf32>
      %index = arith.index_cast %j : i32 to index
      %block = memref.load %partial_ids[%index] : memref<?xi32>
      %mask = arith.mulf %q, %q {test.late_mask} : tensor<4x4xf32>
      %offset = arith.index_cast %block : i32 to index
      %view = memref.subview %source[%offset, 0] [4, 4] [1, 1] : memref<?x4xf32> to memref<4x4xf32, strided<[4, 1], offset: ?>>
      %scratch = memref.alloc() : memref<4x4xf32>
      memref.copy %view, %scratch : memref<4x4xf32, strided<[4, 1], offset: ?>> to memref<4x4xf32>
      %v = bufferization.to_tensor %scratch restrict writable : memref<4x4xf32> to tensor<4x4xf32>
      %qk = linalg.matmul ins(%q, %v : tensor<4x4xf32>, tensor<4x4xf32>) outs(%seed : tensor<4x4xf32>) -> tensor<4x4xf32>
      %scaled = arith.mulf %qk, %q : tensor<4x4xf32>
      %prob = arith.mulf %scaled, %mask : tensor<4x4xf32>
      %pv_empty = tensor.empty() : tensor<4x4xf32>
      %pv_init = linalg.fill ins(%zero_f32 : f32) outs(%pv_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
      %pv = linalg.matmul ins(%prob, %v : tensor<4x4xf32>, tensor<4x4xf32>) outs(%pv_init : tensor<4x4xf32>) -> tensor<4x4xf32>
      %next = arith.addf %pv, %a : tensor<4x4xf32>
      scf.yield %next : tensor<4x4xf32>
    }
    %f = scf.for %j = %zero_i32 to %nf step %one_i32 iter_args(%a = %p) -> tensor<4x4xf32> : i32 {
      %index = arith.index_cast %j : i32 to index
      %block = memref.load %full_ids[%index] : memref<?xi32>
      %offset = arith.index_cast %block : i32 to index
      %view = memref.subview %source[%offset, 0] [4, 4] [1, 1] : memref<?x4xf32> to memref<4x4xf32, strided<[4, 1], offset: ?>>
      %scratch = memref.alloc() : memref<4x4xf32>
      memref.copy %view, %scratch : memref<4x4xf32, strided<[4, 1], offset: ?>> to memref<4x4xf32>
      %v = bufferization.to_tensor %scratch restrict writable : memref<4x4xf32> to tensor<4x4xf32>
      %qk = linalg.matmul ins(%q, %v : tensor<4x4xf32>, tensor<4x4xf32>) outs(%seed : tensor<4x4xf32>) -> tensor<4x4xf32>
      %scaled = arith.mulf %qk, %q : tensor<4x4xf32>
      %pv_empty = tensor.empty() : tensor<4x4xf32>
      %pv_init = linalg.fill ins(%zero_f32 : f32) outs(%pv_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
      %pv = linalg.matmul ins(%scaled, %v : tensor<4x4xf32>, tensor<4x4xf32>) outs(%pv_init : tensor<4x4xf32>) -> tensor<4x4xf32>
      %next = arith.addf %pv, %a : tensor<4x4xf32>
      scf.yield %next : tensor<4x4xf32>
    }
    bufferization.materialize_in_destination %f in writable %out : (tensor<4x4xf32>, memref<4x4xf32>) -> ()
  } {ssbuffer.outer_main_loop}
  return
}
