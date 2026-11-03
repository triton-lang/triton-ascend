// RUN: triton-opt %s --triton-to-linalg='named-ops=true' -verify-each | FileCheck %s
// RUN: triton-opt %s --triton-to-linalg -verify-each | FileCheck %s
// RUN: sed 's/Ascend910B2/Ascend950PR_9579/' %s | triton-opt --triton-to-linalg='named-ops=true compile-on-910-95=true' -verify-each | FileCheck %s

// An i1 splat used as both a mask and numerical data keeps the same active
// extent as a mask-only splat. Inactive loads must preserve `other`, and
// inactive stores must not access memory.
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  // CHECK-LABEL: func.func @load_mask_shared_with_data(
  // CHECK-SAME: %[[MASK:[^:]+]]: i1, %[[OTHER:[^:]+]]: i32
  // CHECK: %[[SIZE:.*]] = arith.index_castui %[[MASK]] : i1 to index
  // CHECK: scf.if
  // CHECK: linalg.fill ins(%[[OTHER]] : i32)
  // CHECK: %[[SRC_VIEW:.*]] = memref.subview %{{.*}}[0] [%[[SIZE]]] [1]
  // CHECK: %[[DST_VIEW:.*]] = memref.subview %{{.*}}[0] [%[[SIZE]]] [1]
  // CHECK: memref.copy %[[SRC_VIEW]], %[[DST_VIEW]]
  // CHECK: arith.extui
  // CHECK: return
  tt.func public @load_mask_shared_with_data(%src: !tt.ptr<i32>, %dst: !tt.ptr<i32>, %mask_dst: !tt.ptr<i32>, %mask: i1, %other: i32) {
    %mask_tensor = tt.splat %mask : i1 -> tensor<1xi1>
    %ptr = tt.splat %src : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %fallback = tt.splat %other : i32 -> tensor<1xi32>
    %value = tt.load %ptr, %mask_tensor, %fallback : tensor<1x!tt.ptr<i32>>
    %mask_data = arith.extui %mask_tensor : tensor<1xi1> to tensor<1xi32>
    %out = tt.splat %dst : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %mask_out = tt.splat %mask_dst : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    tt.store %out, %value : tensor<1x!tt.ptr<i32>>
    tt.store %mask_out, %mask_data : tensor<1x!tt.ptr<i32>>
    tt.return
  }

  // CHECK-LABEL: func.func @store_mask_shared_with_data(
  // CHECK-SAME: %[[MASK:[^:]+]]: i1
  // CHECK: %[[SIZE:.*]] = arith.index_castui %[[MASK]] : i1 to index
  // CHECK-DAG: %[[DST_VIEW:.*]] = memref.subview %{{.*}}[0] [%[[SIZE]]] [1]
  // CHECK-DAG: %[[VALUE:.*]] = tensor.extract_slice %{{.*}}[0] [%[[SIZE]]] [1]
  // CHECK: bufferization.materialize_in_destination %[[VALUE]] in writable %[[DST_VIEW]]
  tt.func public @store_mask_shared_with_data(%dst: !tt.ptr<i32>, %mask: i1) {
    %mask_tensor = tt.splat %mask : i1 -> tensor<1xi1>
    %ptr = tt.splat %dst : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %value = arith.extui %mask_tensor : tensor<1xi1> to tensor<1xi32>
    tt.store %ptr, %value, %mask_tensor : tensor<1x!tt.ptr<i32>>
    tt.return
  }

  // The same rule applies to blocks larger than one element; preserve the
  // original tensor shape when materializing the mask for its numerical use.
  // CHECK-LABEL: func.func @block_mask_shared_with_data(
  // CHECK-SAME: %[[MASK:[^:]+]]: i1
  // CHECK: %[[EIGHT:.*]] = arith.constant 8 : index
  // CHECK: %[[BIT:.*]] = arith.index_castui %[[MASK]] : i1 to index
  // CHECK: %[[SIZE:.*]] = arith.muli %[[BIT]], %[[EIGHT]] : index
  // CHECK: %[[SRC_VIEW:.*]] = memref.subview %{{.*}}[0] [%[[SIZE]]] [1]
  // CHECK: %[[DST_VIEW:.*]] = memref.subview %{{.*}}[0] [%[[SIZE]]] [1]
  // CHECK: memref.copy %[[SRC_VIEW]], %[[DST_VIEW]]
  // CHECK: return
  tt.func public @block_mask_shared_with_data(%src: !tt.ptr<i32>, %dst: !tt.ptr<i32>, %mask_dst: !tt.ptr<i32>, %mask: i1) {
    %zero = arith.constant dense<0> : tensor<8xi32>
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %base = tt.splat %src : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ptr = tt.addptr %base, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %mask_tensor = tt.splat %mask : i1 -> tensor<8xi1>
    %value = tt.load %ptr, %mask_tensor, %zero : tensor<8x!tt.ptr<i32>>
    %mask_data = arith.extui %mask_tensor : tensor<8xi1> to tensor<8xi32>
    %out_base = tt.splat %dst : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %out = tt.addptr %out_base, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %mask_out_base = tt.splat %mask_dst : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %mask_out = tt.addptr %mask_out_base, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %out, %value : tensor<8x!tt.ptr<i32>>
    tt.store %mask_out, %mask_data : tensor<8x!tt.ptr<i32>>
    tt.return
  }

  // A shared boolean splat must also remain a mask when intersected with a
  // per-element bound. Treating it as a numerical scalar makes parseAnd fail.
  // CHECK-LABEL: func.func @store_combined_mask_shared_with_data(
  // CHECK-SAME: %[[MASK:[^:]+]]: i1
  // CHECK: %[[EIGHT:.*]] = arith.constant 8 : index
  // CHECK: %[[BIT:.*]] = arith.index_castui %[[MASK]] : i1 to index
  // CHECK: %[[UNIFORM_SIZE:.*]] = arith.muli %[[BIT]], %[[EIGHT]] : index
  // CHECK: %[[INTERSECTION:.*]] = arith.minsi %[[UNIFORM_SIZE]], %{{.*}} : index
  // CHECK: %[[SIZE:.*]] = arith.maxsi %[[INTERSECTION]], %{{.*}} : index
  // CHECK-DAG: %[[DST_VIEW:.*]] = memref.subview %{{.*}}[0] [%[[SIZE]]] [1]
  // CHECK-DAG: %[[VALUE:.*]] = tensor.extract_slice %{{.*}}[0] [%[[SIZE]]] [1]
  // CHECK: bufferization.materialize_in_destination %[[VALUE]] in writable %[[DST_VIEW]]
  tt.func public @store_combined_mask_shared_with_data(%dst: !tt.ptr<i32>, %mask: i1, %size: i32) {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %limit = tt.splat %size : i32 -> tensor<8xi32>
    %bound = arith.cmpi slt, %range, %limit : tensor<8xi32>
    %uniform = tt.splat %mask : i1 -> tensor<8xi1>
    %combined = arith.andi %uniform, %bound : tensor<8xi1>
    %value = arith.extui %uniform : tensor<8xi1> to tensor<8xi32>
    %base = tt.splat %dst : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ptr = tt.addptr %base, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %ptr, %value, %combined : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}
