// RUN: triton-opt --plan-cube-block --plan-vector-block %s | FileCheck %s

// S208. Pure vector scf.if: the entire if operation should be identified as a single VECTOR op.
// The inputs and outputs outside the loop/if should be planned under consistent vector block IDs.
// CHECK-LABEL: func.func @test_pure_vector_if_op(
// CHECK: [[CST:%[a-zA-Z0-9_]+]] = arith.constant {ssbuffer.block_id = [[TC_VEC0:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: [[IF_RES:%[0-9]+]] = scf.if %arg2 -> (tensor<64x64xf32>) {
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[TC_VEC_IN:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: } else {
// CHECK: arith.mulf {{.*}} {ssbuffer.block_id = [[TC_VEC_IN2:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: } {ssbuffer.block_id = [[TC_VEC0]] : i32}
// CHECK: [[OUT:%[0-9]+]] = arith.subf [[IF_RES]], %arg0 {ssbuffer.block_id = [[TC_VEC0]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: return [[OUT]]
func.func @test_pure_vector_if_op(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: i1) -> tensor<64x64xf32> {
    %cst = arith.constant {ssbuffer.core_type = "VECTOR"} 1.000000e+00 : f32
    %0 = scf.if %arg2 -> (tensor<64x64xf32>) {
        %1 = arith.addf %arg0, %arg1 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        scf.yield %1 : tensor<64x64xf32>
    } else {
        %2 = arith.mulf %arg0, %arg1 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        scf.yield %2 : tensor<64x64xf32>
    }
    %3 = arith.subf %0, %arg0 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    return %3 : tensor<64x64xf32>
}

// S209. Pure cube scf.for: the entire loop operation is identified as CUBE_ONLY.
// CHECK-LABEL: func.func @test_pure_cube_for_op(
// CHECK: [[C0:%.+]] = arith.constant {ssbuffer.block_id = [[TC_CUBE_OUT:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} 0 : index
// CHECK: [[FOR_RES:%[0-9]+]] = scf.for {{.*}} step {{.*}} iter_args({{.*}}) -> (tensor<64x64xf32>) {
// CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = [[TC_CUBE_IN:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: } {ssbuffer.block_id = [[TC_CUBE_OUT]] : i32}
func.func @test_pure_cube_for_op(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %c0 = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1 = arith.constant {ssbuffer.core_type = "CUBE"} 1 : index
    %c4 = arith.constant {ssbuffer.core_type = "CUBE"} 4 : index
    %res = scf.for %iv = %c0 to %c4 step %c1 iter_args(%arg2 = %arg1) -> (tensor<64x64xf32>) {
        %trunc = arith.truncf %arg2 {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
        %out = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
        %mm = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%trunc, %arg0 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
        scf.yield %mm : tensor<64x64xf32>
    }
    return %res : tensor<64x64xf32>
}


// S210. SCF loop with nested non-CF region ops (e.g., linalg.generic).
// The preorder walk should skip the nested blocks of non-RegionBranchOp operations,
// thereby correctly identifying the loop as pure VECTOR despite containing scalar region ops.
// CHECK-LABEL: func.func @test_scf_with_nested_non_cf_regions(
// CHECK: [[C0:%.+]] = arith.constant {ssbuffer.block_id = [[TC_VEC_OUT:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} 0 : index
// CHECK: [[FOR_RES:%[0-9]+]] = scf.for {{.*}} step {{.*}} iter_args({{.*}}) -> (tensor<256xi32>) {
// CHECK: linalg.generic {indexing_maps = {{.*}}, iterator_types = ["parallel"]} {{.*}}ssbuffer.block_id = [[TC_VEC_IN:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"
// CHECK: } {ssbuffer.block_id = [[TC_VEC_OUT]] : i32}
#map_ut = affine_map<(d0) -> (d0)>
func.func @test_scf_with_nested_non_cf_regions(%arg0: tensor<256xi32>) -> tensor<256xi32> {
    %c0 = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %c1 = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : index
    %c4 = arith.constant {ssbuffer.core_type = "VECTOR"} 4 : index
    %res = scf.for %iv = %c0 to %c4 step %c1 iter_args(%arg1 = %arg0) -> (tensor<256xi32>) {
        %out = tensor.empty() {ssbuffer.core_type = "VECTOR"} : tensor<256xi32>
        %generic = linalg.generic {indexing_maps = [#map_ut], iterator_types = ["parallel"]} outs(%out : tensor<256xi32>) attrs = {ssbuffer.core_type = "VECTOR"} {
        ^bb0(%val: i32):
            %idx = linalg.index 0 : index
            %cast = arith.index_cast %idx : index to i32
            linalg.yield %cast : i32
        } -> tensor<256xi32>
        scf.yield %generic : tensor<256xi32>
    }
    return %res : tensor<256xi32>
}

// S211. Mixed scf.if containing both CUBE and VECTOR ops.
// The control-flow op should be identified as UNDETERMINED, preventing it from being treated
// as a single CUBE or VECTOR block (the scf.if should not obtain any ssbuffer.block_id attribute).
// CHECK-LABEL: func.func @test_mixed_cf_op_undetermined(
// CHECK-NOT: scf.if {{.*}} ssbuffer.block_id
// CHECK: scf.if %arg2 -> (tensor<64x64xf32>) {
// CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[TC_VEC_IN:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: } else {
// CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = [[TC_CUBE_IN:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
// CHECK: }
func.func @test_mixed_cf_op_undetermined(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf16>, %arg2: i1) -> tensor<64x64xf32> {
    %0 = scf.if %arg2 -> (tensor<64x64xf32>) {
        %1 = arith.addf %arg0, %arg0 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        scf.yield %1 : tensor<64x64xf32>
    } else {
        %out = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
        %2 = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%arg1, %arg1 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
        scf.yield %2 : tensor<64x64xf32>
    }
    return %0 : tensor<64x64xf32>
}


// S212. Alloc and its use spanning across a vector scf.for loop.
// The scf.for loop is identified as a pure VECTOR op and is fused with the alloc
// and its use, ensuring they all share the same block ID outside the loop.
// CHECK-LABEL: func.func @test_alloc_and_use_around_for_loop(
// CHECK: [[ALLOC:%[A-Za-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = [[TC_VEC_OUT:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: scf.for
// CHECK: } {ssbuffer.block_id = [[TC_VEC_OUT]] : i32}
// CHECK: [[TENSOR:%[0-9]+]] = bufferization.to_tensor [[ALLOC]] restrict writable {ssbuffer.block_id = [[TC_VEC_OUT]] : i32, ssbuffer.core_type = "VECTOR"}
func.func @test_alloc_and_use_around_for_loop(%arg0: memref<?xf16>, %arg1: i32) -> tensor<128x64xf16> {
    %c0 = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %c1 = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : index
    %c128 = arith.constant {ssbuffer.core_type = "VECTOR"} 128 : index
    %alloc = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<128x64xf16>
    scf.for %arg23 = %c0 to %c128 step %c1 {
        %subview = memref.subview %alloc[%arg23, 0] [1, 64] [1, 1] {ssbuffer.core_type = "VECTOR"} : memref<128x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
        %reinterpret = memref.reinterpret_cast %arg0 to offset: [%arg23], sizes: [1, 64], strides: [64, 1] {ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
        memref.copy %reinterpret, %subview {ssbuffer.core_type = "VECTOR"} : memref<1x64xf16, strided<[64, 1], offset: ?>> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    }
    %res = bufferization.to_tensor %alloc restrict writable {ssbuffer.core_type = "VECTOR"} : memref<128x64xf16>
    return %res : tensor<128x64xf16>
}

