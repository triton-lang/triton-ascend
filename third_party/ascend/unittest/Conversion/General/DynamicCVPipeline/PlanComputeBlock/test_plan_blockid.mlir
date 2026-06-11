// RUN: triton-opt --plan-cube-block --plan-vector-block %s | FileCheck %s

module{
    // S201. 设计用例需要同时存在cube->vector, cube->cube, vector->cube的依赖，并且cube类节点存在2个以上的dot节点，至少有一个dot包含向上依赖，至少有一个dot包含向下依赖， 需要检查每个block最多包含一个dot op
    
// CHECK-LABEL: func.func @test_core_type_deps(
// CHECK: [[ALLOC_VEC:%[A-Za-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = [[TC_VEC3:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : memref<64x64xf16>
// CHECK-NEXT: [[ALLOC_0_VEC:%[A-Za-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = [[TC_VECB:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : memref<64x64xf16>
// CHECK-NEXT: [[TO_TENSOR0:%[0-9]+]] = bufferization.to_tensor [[ALLOC_VEC]] restrict writable {ssbuffer.block_id = [[TC_VEC3]] : i32, ssbuffer.core_type = "VECTOR"} : memref<64x64xf16>
// CHECK-NEXT: [[TO_TENSOR1:%[0-9]+]] = bufferization.to_tensor [[ALLOC_0_VEC]] restrict writable {ssbuffer.block_id = [[TC_VECB]] : i32, ssbuffer.core_type = "VECTOR"} : memref<64x64xf16>
// CHECK-NEXT: [[SUBF_VEC:%[0-9]+]] = arith.subf %arg0, %arg1 {ssbuffer.block_id = [[TC_VEC3]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[EXP_VEC:%[0-9]+]] = math.exp [[SUBF_VEC]] {ssbuffer.block_id = [[TC_VEC3]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[TRUNC_CUBE0:%[0-9]+]] = arith.truncf [[EXP_VEC]] {ssbuffer.block_id = [[TC_CUBE0:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK-NEXT: [[EMPTY_VEC0:%[0-9]+]] = tensor.empty() {ssbuffer.block_id = [[TC_VEC3]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[MATMUL_CUBE0:%[0-9]+]] = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = [[TC_CUBE0]] : i32, ssbuffer.core_type = "CUBE"} ins([[TRUNC_CUBE0]], [[TO_TENSOR0]] : tensor<64x64xf16>, tensor<64x64xf16>) outs([[EMPTY_VEC0]] : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT: [[TRUNC_CUBE0_2:%[0-9]+]] = arith.truncf [[MATMUL_CUBE0]] {ssbuffer.block_id = [[TC_CUBE0]] : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK-NEXT: [[EMPTY_VEC1:%[0-9]+]] = tensor.empty() {ssbuffer.block_id = [[TC_VECB]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[MATMUL_CUBE1:%[0-9]+]] = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = [[TC_CUBE1:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} ins([[TRUNC_CUBE0_2]], [[TO_TENSOR1]] : tensor<64x64xf16>, tensor<64x64xf16>) outs([[EMPTY_VEC1]] : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT: [[EXTF_VEC4:%[0-9]+]] = arith.extf [[TRUNC_CUBE0]] {ssbuffer.block_id = [[TC_VEC4:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf32>
// CHECK-NEXT: [[ADDF_VEC4:%[0-9]+]] = arith.addf [[EXTF_VEC4]], %arg0 {ssbuffer.block_id = [[TC_VEC4]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[EMPTY_VEC4:%[0-9]+]] = tensor.empty() {ssbuffer.block_id = [[TC_VEC4]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[BROADCAST_VEC4:%[A-Za-z0-9_]+]] = linalg.broadcast ins(%arg2 : tensor<64xf32>) outs([[EMPTY_VEC4]] : tensor<64x64xf32>) dimensions = [1] {ssbuffer.block_id = [[TC_VEC4]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK-NEXT: [[SUBF_VEC4:%[0-9]+]] = arith.subf [[MATMUL_CUBE1]], [[BROADCAST_VEC4]] {ssbuffer.block_id = [[TC_VEC4]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[MULF_VEC4:%[0-9]+]] = arith.mulf [[ADDF_VEC4]], [[SUBF_VEC4]] {ssbuffer.block_id = [[TC_VEC4]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[TRUNC_CUBE2:%[0-9]+]] = arith.truncf [[MULF_VEC4]] {ssbuffer.block_id = [[TC_CUBE2:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK-NEXT: [[EMPTY_VEC3:%[0-9]+]] = tensor.empty() {ssbuffer.block_id = [[TC_VECD:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[MATMUL_CUBE2:%[0-9]+]] = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = [[TC_CUBE2]] : i32, ssbuffer.core_type = "CUBE"} ins([[TRUNC_CUBE2]], [[TRUNC_CUBE0_2]] : tensor<64x64xf16>, tensor<64x64xf16>) outs([[EMPTY_VEC3]] : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT: [[TRUNC_VEC5:%[0-9]+]] = arith.truncf [[MATMUL_CUBE2]] {ssbuffer.block_id = [[TC_VEC5:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK-NEXT: return [[TRUNC_VEC5]] : tensor<64x64xf16>

    func.func @test_core_type_deps(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>) -> tensor<64x64xf16> {
        %alloc_0 = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<64x64xf16>
        %alloc_1 = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<64x64xf16>
        %0 = bufferization.to_tensor %alloc_0 restrict writable {ssbuffer.core_type = "VECTOR"} : memref<64x64xf16>
        %1 = bufferization.to_tensor %alloc_1 restrict writable {ssbuffer.core_type = "VECTOR"} : memref<64x64xf16>
        %2 = arith.subf %arg0, %arg1 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %3 = math.exp %2 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %4 = arith.truncf %3 {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>   
        %out0 = tensor.empty() {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %5 = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%4, %0 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%out0 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %6 = arith.truncf %5 {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
        %out1 = tensor.empty() {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %7 = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%6, %1 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%out1 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %8 = arith.extf %4 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf32>
        %9 = arith.addf %8, %arg0 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %broadcasted = tensor.empty() {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %10 = linalg.broadcast ins(%arg2 : tensor<64xf32>) outs(%broadcasted : tensor<64x64xf32>) dimensions = [1] {ssbuffer.core_type = "VECTOR"}
        %11 = arith.subf %7, %10 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %12 = arith.mulf %9, %11 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %13 = arith.truncf %12 {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
        %out2 = tensor.empty() {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %14 = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%13, %6 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%out2 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %15 = arith.truncf %14 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
        return %15 : tensor<64x64xf16>
    }

    // S105. cube block应该缩点， 具体例子：IR序列：abcdef, acde是CUBE， bf是VECTOR， a的block_id为0，c,d的block_id为1，e的block_id为3，他们的依赖关系是a->b->d, c->d, c->e->f

// CHECK-LABEL: func.func @test_shrink_cube_block(
// CHECK-NEXT: [[ALLOC_VEC:%[A-Za-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = [[TC_VEC4:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : memref<64x128xf16>
// CHECK-NEXT: [[TO_TENSOR_VEC4:%[0-9]+]] = bufferization.to_tensor [[ALLOC_VEC]] restrict writable {ssbuffer.block_id = [[TC_VEC4]] : i32, ssbuffer.core_type = "VECTOR"} : memref<64x128xf16>
// CHECK-NEXT: [[SUBF_CUBE2:%[0-9]+]] = arith.subf %arg0, %arg5 {ssbuffer.block_id = [[TC_CUBE2:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
// CHECK-NEXT: [[EXP_VEC6:%[0-9]+]] = math.exp [[SUBF_CUBE2]] {ssbuffer.block_id = [[TC_VEC6:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[TRUNC_CUBE3:%[0-9]+]] = arith.truncf [[EXP_VEC6]] {ssbuffer.block_id = [[TC_CUBE3:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK-NEXT: [[EMPTY_VEC8:%[0-9]+]] = tensor.empty() {ssbuffer.block_id = [[TC_VEC8:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16>
// CHECK-NEXT: [[TRANSPOSE_VEC8:%[A-Za-z0-9_]+]] = linalg.transpose ins([[TRUNC_CUBE3]] : tensor<64x64xf16>) outs([[EMPTY_VEC8]] : tensor<64x64xf16>) permutation = [1, 0] {ssbuffer.block_id = [[TC_VEC8]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK-NEXT: [[MATMUL_CUBE0:%[0-9]+]] = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = [[TC_CUBE0:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} ins([[TRANSPOSE_VEC8]], [[TO_TENSOR_VEC4]] : tensor<64x64xf16>, tensor<64x128xf16>) outs(%arg2 : tensor<64x128xf32>) -> tensor<64x128xf32>
// CHECK-NEXT: [[EMPTY_VEC9_0:%[0-9]+]] = tensor.empty() {ssbuffer.block_id = [[TC_VEC9:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[EMPTY_VEC4:%[0-9]+]] = tensor.empty() {ssbuffer.block_id = [[TC_VEC4]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[MATMUL_CUBE1:%[0-9]+]] = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = [[TC_CUBE1:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} ins([[TO_TENSOR_VEC4]], %arg4 : tensor<64x128xf16>, tensor<128x64xf16>) outs([[EMPTY_VEC4]] : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT: [[BROADCAST_VEC9:%[A-Za-z0-9_]+]] = linalg.broadcast ins(%arg6 : tensor<64xf32>) outs([[EMPTY_VEC9_0]] : tensor<64x64xf32>) dimensions = [1] {ssbuffer.block_id = [[TC_VEC9]] : i32, ssbuffer.core_type = "VECTOR"}
// CHECK-NEXT: [[SUBF_VEC9:%[0-9]+]] = arith.subf [[MATMUL_CUBE1]], [[BROADCAST_VEC9]] {ssbuffer.block_id = [[TC_VEC9]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[EXTF_VEC9:%[0-9]+]] = arith.extf [[TRUNC_CUBE3]] {ssbuffer.block_id = [[TC_VEC9]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf32>
// CHECK-NEXT: [[MULF_VEC9:%[0-9]+]] = arith.mulf [[EXTF_VEC9]], [[SUBF_VEC9]] {ssbuffer.block_id = [[TC_VEC9]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[TRUNC_VEC9:%[0-9]+]] = arith.truncf [[MULF_VEC9]] {ssbuffer.block_id = [[TC_VEC9]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK-NEXT: return [[TRUNC_VEC9]] : tensor<64x64xf16>

    func.func @test_shrink_cube_block(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x128xf32>, %arg3: tensor<128x64xf16>, %arg4: tensor<128x64xf16>, %arg5: tensor<64x64xf32>, %arg6: tensor<64xf32>) -> tensor<64x64xf16> {
        %alloc_8 = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<64x128xf16>
        %32 = bufferization.to_tensor %alloc_8 restrict writable {ssbuffer.core_type = "VECTOR"} : memref<64x128xf16>
        %37 = arith.subf %arg0, %arg5 {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
        %38 = math.exp %37 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %39 = arith.truncf %38 {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
        %40 = tensor.empty() {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16>
        %transposed_13 = linalg.transpose ins(%39 : tensor<64x64xf16>) outs(%40 : tensor<64x64xf16>) permutation = [1, 0] {ssbuffer.core_type = "VECTOR"}
        %41 = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%transposed_13, %32 : tensor<64x64xf16>, tensor<64x128xf16>) outs(%arg2 : tensor<64x128xf32>) -> tensor<64x128xf32>
        %2 = tensor.empty() {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %3 = tensor.empty() {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %42 = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%32, %arg4 : tensor<64x128xf16>, tensor<128x64xf16>) outs(%3 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %broadcasted_14 = linalg.broadcast ins(%arg6 : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1] {ssbuffer.core_type = "VECTOR"}
        %43 = arith.subf %42, %broadcasted_14 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %44 = arith.extf %39 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf32>
        %45 = arith.mulf %44, %43 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %46 = arith.truncf %45 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
        return %46 : tensor<64x64xf16>
    }

    // dot 锚点驱动的分块算法-willCreateCycle：当前block融合后，可能跟其他节点形成环，设计用例需要存在可能成环的场景 A(cube)->B(vector)->C(dot)->D(cube), A-C. 这种情况下AC的blockid应该不同

// CHECK-LABEL: func.func @test_fusion_cycle_dot(
// CHECK: [[ALLOC_VEC:%[A-Za-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = [[TC_VEC4:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : memref<64x64xf16>
// CHECK-NEXT: [[TO_TENSOR_VEC:%[0-9]+]] = bufferization.to_tensor [[ALLOC_VEC]] restrict writable {ssbuffer.block_id = [[TC_VEC4]] : i32, ssbuffer.core_type = "VECTOR"} : memref<64x64xf16>
// CHECK-NEXT: [[TRUNC_CUBE1:%[0-9]+]] = arith.truncf %arg0 {ssbuffer.block_id = [[TC_CUBE1:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK-NEXT: [[EXTF_VEC3:%[0-9]+]] = arith.extf [[TRUNC_CUBE1]] {ssbuffer.block_id = [[TC_VEC3:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf32>
// CHECK-NEXT: [[ADDF_VEC3:%[0-9]+]] = arith.addf [[EXTF_VEC3]], %arg1 {ssbuffer.block_id = [[TC_VEC3]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[TRUNC_VEC3:%[0-9]+]] = arith.truncf [[ADDF_VEC3]] {ssbuffer.block_id = [[TC_VEC3]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK-NEXT: [[EMPTY_VEC2:%[0-9]+]] = tensor.empty() {ssbuffer.block_id = [[TC_VEC2:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[MATMUL_CUBE0:%[0-9]+]] = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = [[TC_CUBE0:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} ins([[TRUNC_VEC3]], [[TRUNC_CUBE1]] : tensor<64x64xf16>, tensor<64x64xf16>) outs([[EMPTY_VEC2]] : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT: [[ADDF_CUBE0:%[0-9]+]] = arith.addf [[MATMUL_CUBE0]], %arg0 {ssbuffer.block_id = [[TC_CUBE0]] : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
// CHECK-NEXT: return [[ADDF_CUBE0]] : tensor<64x64xf32>

    func.func @test_fusion_cycle_dot(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
        %alloc_0 = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<64x64xf16>
        %0 = bufferization.to_tensor %alloc_0 restrict writable {ssbuffer.core_type = "VECTOR"} : memref<64x64xf16>
        %a = arith.truncf %arg0 {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
        %b0 = arith.extf %a {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf32>
        %b1 = arith.addf %b0, %arg1 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %b2 = arith.truncf %b1 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
        %out_c = tensor.empty() {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %c = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%b2, %a : tensor<64x64xf16>, tensor<64x64xf16>) outs(%out_c : tensor<64x64xf32>) -> tensor<64x64xf32>
        %d = arith.addf %c, %arg0 {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
        return %d : tensor<64x64xf32>
    }

    // S207. ir中存在一个for循环包括了2个迭代变量arg1, arg2, 循环内ir序列为abcdefg, be为dot节点, 其余均为CUBE节点, c是对arg1加一个常数, f是对arg2加一个常数, g是yeild c,f, 除此之外的依赖关系为 arg1->a->b, arg2->d->e.

// CHECK-LABEL: func.func @test_for_loop_deps(
// CHECK: [[C0:%[A-Za-z0-9]+]] = arith.constant {ssbuffer.block_id = [[TC_CUBE2:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} 0 : index
// CHECK-NEXT: [[C1:%[A-Za-z0-9]+]] = arith.constant {ssbuffer.block_id = [[TC_CUBE2]] : i32, ssbuffer.core_type = "CUBE"} 1 : index
// CHECK-NEXT: [[C4:%[A-Za-z0-9]+]] = arith.constant {ssbuffer.block_id = [[TC_CUBE2]] : i32, ssbuffer.core_type = "CUBE"} 4 : index
// CHECK-NEXT: [[CST:%[A-Za-z0-9]+]] = arith.constant {ssbuffer.block_id = [[TC_CUBE2]] : i32, ssbuffer.core_type = "CUBE"} dense<1.000000e+00> : tensor<64x64xf32>
// CHECK: [[FOR_RES:%[0-9]+]]:2 = scf.for [[IV:%[A-Za-z0-9]+]] = [[C0]] to [[C4]] step [[C1]] iter_args([[ARG4:%[A-Za-z0-9]+]] = %arg1, [[ARG5:%[A-Za-z0-9]+]] = %arg2)
// CHECK: [[TRUNC0:%[A-Za-z0-9]+]] = arith.truncf [[ARG4]] {ssbuffer.block_id = [[TC_CUBE0:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK-NEXT: [[EMPTY0:%[A-Za-z0-9]+]] = tensor.empty() {ssbuffer.block_id = [[TC_CUBE0]] : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
// CHECK-NEXT: [[MM0:%[0-9]+]] = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = [[TC_CUBE0]] : i32, ssbuffer.core_type = "CUBE"} ins([[TRUNC0]], %arg0 : tensor<64x64xf16>, tensor<64x64xf16>) outs([[EMPTY0]] : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT: [[ADDF0:%[0-9]+]] = arith.addf [[ARG4]], [[CST]] {ssbuffer.block_id = [[TC_CUBE0]] : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
// CHECK: [[TRUNC1:%[0-9]+]] = arith.truncf [[ARG5]] {ssbuffer.block_id = [[TC_CUBE1:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK-NEXT: [[EMPTY1:%[0-9]+]] = tensor.empty() {ssbuffer.block_id = [[TC_CUBE1]] : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
// CHECK-NEXT: [[MM1:%[0-9]+]] = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = [[TC_CUBE1]] : i32, ssbuffer.core_type = "CUBE"} ins([[TRUNC1]], %arg0 : tensor<64x64xf16>, tensor<64x64xf16>) outs([[EMPTY1]] : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT: [[ADDF1:%[0-9]+]] = arith.addf [[ARG5]], [[CST]] {ssbuffer.block_id = [[TC_CUBE1]] : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
// CHECK-NEXT: scf.yield [[ADDF0]], [[ADDF1]] : tensor<64x64xf32>, tensor<64x64xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return [[FOR_RES]]#0, [[FOR_RES]]#1 : tensor<64x64xf32>, tensor<64x64xf32>

    func.func @test_for_loop_deps(%arg0: tensor<64x64xf16>, %arg1_init: tensor<64x64xf32>, %arg2_init: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
        %c0 = arith.constant {ssbuffer.core_type = "CUBE"}  0 : index
        %c1 = arith.constant {ssbuffer.core_type = "CUBE"}  1 : index
        %c4 = arith.constant {ssbuffer.core_type = "CUBE"}  4 : index
        %cst = arith.constant {ssbuffer.core_type = "CUBE"}  dense<1.0> : tensor<64x64xf32>
        %result:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%arg1 = %arg1_init, %arg2 = %arg2_init) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
            %a = arith.truncf %arg1 {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
            %out_b = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
            %b = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%a, %arg0 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%out_b : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c = arith.addf %arg1, %cst {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
            %d = arith.truncf %arg2 {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
            %out_e = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
            %e = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%d, %arg0 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%out_e : tensor<64x64xf32>) -> tensor<64x64xf32>
            %f = arith.addf %arg2, %cst {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
            scf.yield %c, %f : tensor<64x64xf32>, tensor<64x64xf32>
        }
        return %result#0, %result#1 : tensor<64x64xf32>, tensor<64x64xf32>
    }

    // a. not all error只cut部分, 但是没有新的candidates

// CHECK-LABEL: func.func @test_cut_err_op(
// CHECK: [[VEC_ADD1_0:%[0-9]+]] = arith.addf %arg0, %arg1 {ssbuffer.block_id = [[TC_VEC1:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK-NEXT: [[VEC_MUL1:%[0-9]+]] = arith.mulf [[VEC_ADD1_0]], %arg0 {ssbuffer.block_id = [[TC_VEC1]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK-NEXT: [[VEC_ADD1_1:%[0-9]+]] = arith.addf [[VEC_MUL1]], [[VEC_MUL1]] {ssbuffer.block_id = [[TC_VEC1]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK-NEXT: [[CUBE_TRUNC0:%[0-9]+]] = arith.truncf [[VEC_ADD1_1]] {ssbuffer.block_id = [[TC_CUBE0:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK: [[VEC_ADD2_0:%[0-9]+]] = arith.addf %arg0, %arg0 {ssbuffer.block_id = [[TC_VEC2:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK-NEXT: [[VEC_TRUNC2:%[0-9]+]] = arith.truncf [[VEC_ADD2_0]] {ssbuffer.block_id = [[TC_VEC2]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK-NEXT: [[VEC_ADD2_1:%[0-9]+]] = arith.addf [[CUBE_TRUNC0]], [[VEC_TRUNC2]] {ssbuffer.block_id = [[TC_VEC2]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16>
// CHECK-NEXT: return [[VEC_ADD2_1]] : tensor<128x128xf16>

    func.func @test_cut_err_op(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf16> {
        %a = arith.addf %arg0, %arg1 {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %b = arith.mulf %a, %arg0 {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %c = arith.addf %b, %b {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %d = arith.truncf %c {ssbuffer.core_type = "CUBE"} : tensor<128x128xf32> to tensor<128x128xf16>
        %e = arith.addf %arg0, %arg0 {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %e_cast = arith.truncf %e {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32> to tensor<128x128xf16>
        %f = arith.addf %d, %e_cast {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16>
        return %f : tensor<128x128xf16>
    }

    // d. not all error && has new cadidates: (a的特殊情况)

// CHECK-LABEL: func.func @test_cut_err_op_new_cand(
// CHECK: [[VEC_ADD1:%[0-9]+]] = arith.addf %arg0, %arg1 {ssbuffer.block_id = [[TC_VEC1:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK-NEXT: [[VEC_MUL1:%[0-9]+]] = arith.mulf %arg0, %arg1 {ssbuffer.block_id = [[TC_VEC1]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK-NEXT: [[VEC_ADD1_2:%[0-9]+]] = arith.addf [[VEC_ADD1]], [[VEC_MUL1]] {ssbuffer.block_id = [[TC_VEC1]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK-NEXT: [[CUBE_TRUNC0:%[0-9]+]] = arith.truncf [[VEC_ADD1_2]] {ssbuffer.block_id = [[TC_CUBE0:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK-NEXT: [[VEC_ADD2:%[0-9]+]] = arith.addf [[CUBE_TRUNC0]], [[CUBE_TRUNC0]] {ssbuffer.block_id = [[TC_VEC2:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16>
// CHECK-NEXT: [[VEC_MUL2:%[0-9]+]] = arith.mulf [[VEC_ADD2]], [[VEC_ADD2]] {ssbuffer.block_id = [[TC_VEC2]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16>
// CHECK-NEXT: return [[VEC_MUL2]] : tensor<128x128xf16>
    
    func.func @test_cut_err_op_new_cand(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf16> {
        %a = arith.addf %arg0, %arg1 {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %b = arith.mulf %arg0, %arg1 {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %c = arith.addf %a, %b {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %d = arith.truncf %c {ssbuffer.core_type = "CUBE"} : tensor<128x128xf32> to tensor<128x128xf16>
        %e = arith.addf %d, %d {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16>
        %f = arith.mulf %e, %e {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16>
        return %f : tensor<128x128xf16>
    }

    // S104. 带有region的op处理：需要考虑入度的计算；

// CHECK-LABEL: func.func @test_if_region(
// CHECK: [[VEC_ADD2_0:%[0-9]+]] = arith.addf %arg0, %arg1 {ssbuffer.block_id = [[TC_VEC2:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK-NEXT: [[VEC_ADD2_1:%[0-9]+]] = arith.addf %arg1, %arg2 {ssbuffer.block_id = [[TC_VEC2]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK: [[TRUE:%[A-Za-z0-9]+]] = arith.constant true
// CHECK: scf.if [[TRUE]] -> (tensor<128x128xf32>)
// CHECK-NEXT: [[VEC_MUL0:%[0-9]+]] = arith.mulf [[VEC_ADD2_1]], [[VEC_ADD2_0]] {ssbuffer.block_id = [[TC_VEC0:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK-NEXT: [[VEC_MUL1:%[0-9]+]] = arith.mulf [[VEC_ADD2_1]], [[VEC_ADD2_0]] {ssbuffer.block_id = [[TC_VEC1:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK-NEXT: [[VEC_ADD0:%[0-9]+]] = arith.addf [[VEC_MUL0]], %arg2 {ssbuffer.block_id = [[TC_VEC0]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK-NEXT: scf.yield [[VEC_ADD0]] : tensor<128x128xf32>
// CHECK-NEXT: } else
// CHECK-NEXT: scf.yield [[VEC_ADD2_1]] : tensor<128x128xf32>
// CHECK-NEXT: }
// CHECK-NEXT: [[VEC_ADD3:%[0-9]+]] = arith.addf [[VEC_ADD2_1]], %2 {ssbuffer.block_id = [[TC_VEC3:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK-NEXT: return [[VEC_ADD3]] : tensor<128x128xf32>

    func.func @test_if_region(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
        %6 = arith.addf %arg0, %arg1 {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %1 = arith.addf %arg1, %arg2 {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %cond = arith.constant true
        %4 = scf.if %cond -> (tensor<128x128xf32>) {
            %arg1_inner = arith.mulf %1, %6 {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
            %arg6_inner = arith.mulf %1, %6 {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
            %7 = arith.addf %arg1_inner, %arg2 {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
            scf.yield %7 : tensor<128x128xf32>
        } else {
            scf.yield %1 : tensor<128x128xf32>
        }
        %2 = arith.addf %1, %4 {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        return %2 : tensor<128x128xf32>
    }

    // 这个例子有三条语句A（vector）B（vector）C(cube), 这三条语句互不相干

// CHECK-LABEL: func.func @test_independent_nodes(
// CHECK: [[VEC_ADD1:%[0-9]+]] = arith.addf %arg0, %arg1 {ssbuffer.block_id = [[TC_VEC1:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[VEC_MUL1:%[0-9]+]] = arith.mulf %arg1, %arg2 {ssbuffer.block_id = [[TC_VEC1]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
// CHECK-NEXT: [[CUBE_TRUNC0:%[0-9]+]] = arith.truncf %arg2 {ssbuffer.block_id = [[TC_CUBE0:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK-NEXT: return [[VEC_ADD1]], [[VEC_MUL1]], [[CUBE_TRUNC0]] : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf16>

    func.func @test_independent_nodes(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf16>) {
        %a = arith.addf %arg0, %arg1 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %b = arith.mulf %arg1, %arg2 {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
        %c = arith.truncf %arg2 {ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xf16>
        return %a, %b, %c : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf16>
    }


    // findCandidates/updateCandidates：需要包含所有的vector节点均被其他类型节点阻塞

// CHECK-LABEL: func.func @test_core_type_dependencies(
// CHECK: [[VEC_ADD0:%[0-9]+]] = arith.addf %arg0, %arg1 {ssbuffer.block_id = [[TC_VEC2:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK-NEXT: [[VEC_MUL0:%[0-9]+]] = arith.mulf [[VEC_ADD0]], %arg0 {ssbuffer.block_id = [[TC_VEC2]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK-NEXT: [[CUBE_TRUNC0:%[0-9]+]] = arith.truncf [[VEC_MUL0]] {ssbuffer.block_id = [[TC_CUBE1:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK: [[CUBE_ADD0:%[0-9]+]] = arith.addf %arg0, %arg1 {ssbuffer.block_id = [[TC_CUBE0:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
// CHECK-NEXT: [[CUBE_MUL0:%[0-9]+]] = arith.mulf [[CUBE_ADD0]], %arg0 {ssbuffer.block_id = [[TC_CUBE0]] : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
// CHECK-NEXT: [[VEC_TRUNC0:%[0-9]+]] = arith.truncf [[CUBE_MUL0]] {ssbuffer.block_id = [[TC_VEC4:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK: [[VEC_ADD1:%[0-9]+]] = arith.addf %arg0, %arg1 {ssbuffer.block_id = [[TC_VEC3:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK-NEXT: [[VEC_MUL1:%[0-9]+]] = arith.mulf [[VEC_ADD1]], [[CUBE_MUL0]] {ssbuffer.block_id = [[TC_VEC3]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK-NEXT: [[CUBE_TRUNC1:%[0-9]+]] = arith.truncf [[VEC_MUL1]] {ssbuffer.block_id = [[TC_CUBE2:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK-NEXT: return [[CUBE_TRUNC1]] : tensor<128x128xf16>

    func.func @test_core_type_dependencies(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf16> {
        %0 = arith.addf %arg0, %arg1 {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %1 = arith.mulf %0, %arg0 {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %2 = arith.truncf %1 {ssbuffer.core_type = "CUBE"} : tensor<128x128xf32> to tensor<128x128xf16>
        %3 = arith.addf %arg0, %arg1 {ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
        %4 = arith.mulf %3, %arg0 {ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
        %5 = arith.truncf %4 {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32> to tensor<128x128xf16>
        %6 = arith.addf %arg0, %arg1 {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %7 = arith.mulf %6, %4 {ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %8 = arith.truncf %7 {ssbuffer.core_type = "CUBE"} : tensor<128x128xf32> to tensor<128x128xf16>
        return %8 : tensor<128x128xf16>
    }

}
