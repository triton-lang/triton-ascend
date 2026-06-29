module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @addmm_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: f32, %arg7: f32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f16
    %c32 = arith.constant {ssbuffer.core_type = "VECTOR"} 32 : index
    %c32_0 = arith.constant {ssbuffer.core_type = "CUBE"} 32 : index
    %c1 = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : index
    %c1_1 = arith.constant {ssbuffer.core_type = "CUBE"} 1 : index
    %c0 = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %c0_2 = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c31_i32 = arith.constant {ssbuffer.core_type = "VECTOR"} 31 : i32
    %c1_i32 = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c0_i32 = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : i32
    %c32_i32 = arith.constant {ssbuffer.core_type = "VECTOR"} 32 : i32
    %c32_i32_3 = arith.constant {ssbuffer.core_type = "CUBE"} 32 : i32
    %c32_i64 = arith.constant {ssbuffer.core_type = "VECTOR"} 32 : i64
    %c32_i64_4 = arith.constant {ssbuffer.core_type = "CUBE"} 32 : i64
    %cst_5 = arith.constant {ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %cst_6 = arith.constant {ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %0 = tensor.empty() {ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %1 = linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%cst_5 : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %2 = arith.extsi %arg14 {ssbuffer.core_type = "VECTOR"} : i32 to i64
    %3 = arith.extsi %arg14 {ssbuffer.core_type = "CUBE"} : i32 to i64
    %4 = arith.extsi %arg15 {ssbuffer.core_type = "VECTOR"} : i32 to i64
    %5 = arith.muli %2, %c32_i64 {ssbuffer.core_type = "VECTOR"} : i64
    %6 = arith.muli %3, %c32_i64_4 {ssbuffer.core_type = "CUBE"} : i64
    %7 = arith.muli %4, %c32_i64 {ssbuffer.core_type = "VECTOR"} : i64
    %8 = arith.addi %arg8, %c31_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %9 = arith.divsi %8, %c32_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %10 = arith.index_cast %5 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %11 = arith.index_cast %6 {ssbuffer.core_type = "CUBE"} : i64 to index
    %12 = arith.index_cast %arg9 {ssbuffer.core_type = "VECTOR"} : i32 to index
    %13 = arith.index_cast %arg9 {ssbuffer.core_type = "CUBE"} : i32 to index
    %14 = arith.muli %10, %12 {ssbuffer.core_type = "VECTOR"} : index
    %15 = arith.muli %11, %13 {ssbuffer.core_type = "CUBE"} : index
    %16 = arith.index_cast %7 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %17 = arith.index_cast %arg10 {ssbuffer.core_type = "VECTOR"} : i32 to index
    %18 = arith.muli %16, %17 {ssbuffer.core_type = "VECTOR"} : index
    %19 = tensor.empty() {ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %20 = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %21 = linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%cst_5 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %22 = linalg.fill {ssbuffer.core_type = "CUBE"} ins(%cst_6 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %23:3 = scf.for %arg17 = %c0_i32 to %9 step %c1_i32 iter_args(%arg18 = %22, %arg19 = %15, %arg20 = %c0) -> (tensor<32x32xf32>, index, index)  : i32 {
      %reinterpret_cast_10 = memref.reinterpret_cast %arg2 to offset: [%arg19], sizes: [32, 32], strides: [%13, %c1_1] {ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<32x32xf16, strided<[?, ?], offset: ?>>
      %49 = arith.addi %arg20, %18 {ssbuffer.core_type = "VECTOR"} : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%49], sizes: [32, 32], strides: [%c1, %17] {ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<32x32xf16, strided<[?, ?], offset: ?>>
      %50 = arith.muli %arg17, %c32_i32 {ssbuffer.core_type = "VECTOR"} : i32
      %51 = arith.muli %arg17, %c32_i32_3 {ssbuffer.core_type = "CUBE"} : i32
      %52 = arith.subi %arg8, %50 {ssbuffer.core_type = "VECTOR"} : i32
      %53 = arith.subi %arg8, %51 {ssbuffer.core_type = "CUBE"} : i32
      %alloc_12 = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<32x32xf16>
      %54 = arith.index_cast %52 {ssbuffer.core_type = "VECTOR"} : i32 to index
      %55 = arith.index_cast %53 {ssbuffer.core_type = "CUBE"} : i32 to index
      %56 = arith.maxsi %54, %c0 {ssbuffer.core_type = "VECTOR"} : index
      %57 = arith.maxsi %55, %c0_2 {ssbuffer.core_type = "CUBE"} : index
      %58 = arith.minsi %56, %c32 {ssbuffer.core_type = "VECTOR"} : index
      %59 = arith.minsi %57, %c32_0 {ssbuffer.core_type = "CUBE"} : index
      %60 = arith.cmpi slt, %58, %c32 {ssbuffer.core_type = "VECTOR"} : index
      %61 = arith.cmpi slt, %59, %c32_0 {ssbuffer.core_type = "CUBE"} : index
      scf.if %61 {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%alloc_12 : memref<32x32xf16>)
      } {hivm.unlikely_condition}
      %subview_13 = memref.subview %reinterpret_cast_10[0, 0] [32, %59] [1, 1] {ssbuffer.core_type = "CUBE"} : memref<32x32xf16, strided<[?, ?], offset: ?>> to memref<32x?xf16, strided<[?, ?], offset: ?>>
      %subview_14 = memref.subview %alloc_12[0, 0] [32, %59] [1, 1] {ssbuffer.core_type = "CUBE"} : memref<32x32xf16> to memref<32x?xf16, strided<[32, 1]>>
      memref.copy %subview_13, %subview_14 {ssbuffer.core_type = "CUBE"} : memref<32x?xf16, strided<[?, ?], offset: ?>> to memref<32x?xf16, strided<[32, 1]>>
      %62 = bufferization.to_tensor %alloc_12 restrict writable {ssbuffer.core_type = "CUBE"} : memref<32x32xf16>
      %alloc_15 = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<32x32xf16>
      scf.if %60 {
        linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%cst : f16) outs(%alloc_15 : memref<32x32xf16>)
      } {hivm.unlikely_condition}
      %subview_16 = memref.subview %reinterpret_cast_11[0, 0] [%58, 32] [1, 1] {ssbuffer.core_type = "VECTOR"} : memref<32x32xf16, strided<[?, ?], offset: ?>> to memref<?x32xf16, strided<[?, ?], offset: ?>>
      %subview_17 = memref.subview %alloc_15[0, 0] [%58, 32] [1, 1] {ssbuffer.core_type = "VECTOR"} : memref<32x32xf16> to memref<?x32xf16, strided<[32, 1]>>
      memref.copy %subview_16, %subview_17 {ssbuffer.core_type = "VECTOR"} : memref<?x32xf16, strided<[?, ?], offset: ?>> to memref<?x32xf16, strided<[32, 1]>>
      annotation.mark %alloc_15 {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "VECTOR"} : memref<32x32xf16>
      %63 = bufferization.to_tensor %alloc_15 restrict writable {ssbuffer.core_type = "VECTOR"} : memref<32x32xf16>
      annotation.mark %63 {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf16>
      %64 = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%62, %63 : tensor<32x32xf16>, tensor<32x32xf16>) outs(%arg18 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %65 = arith.addi %arg19, %c32_0 {ssbuffer.core_type = "CUBE"} : index
      %66 = arith.addi %arg20, %c32 {ssbuffer.core_type = "VECTOR"} : index
      scf.yield {ssbuffer.core_type = "CUBE, CUBE, VECTOR"} %64, %65, %66 : tensor<32x32xf32>, index, index
    } {hivm.matmul_limited_in_cube, ssbuffer.core_type = "CUBE, CUBE, VECTOR"}
    %24 = arith.cmpi sgt, %9, %c0_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %25 = scf.if %24 -> (tensor<32x32xf32>) {
      scf.yield {ssbuffer.core_type = "CUBE"} %23#0 : tensor<32x32xf32>
    } else {
      %49 = linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%cst_5 : f32) outs(%23#0 : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield {ssbuffer.core_type = "CUBE"} %49 : tensor<32x32xf32>
    } {ssbuffer.core_type = "CUBE"}
    %26 = arith.addf %25, %1 {ssbuffer.add_from_matmul, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %27 = arith.addi %10, %16 {ssbuffer.core_type = "VECTOR"} : index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%27], sizes: [32, 32], strides: [1, 1] {ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<32x32xf16, strided<[1, 1], offset: ?>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg4 to offset: [%27], sizes: [32, 32], strides: [1, 1] {ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<32x32xf16, strided<[1, 1], offset: ?>>
    %alloc = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<32x32xf16>
    %28 = arith.addi %10, %c32 {ssbuffer.core_type = "VECTOR"} : index
    %29 = arith.maxsi %10, %c1 {ssbuffer.core_type = "VECTOR"} : index
    %30 = arith.minsi %28, %29 {ssbuffer.core_type = "VECTOR"} : index
    %31 = arith.subi %30, %10 {ssbuffer.core_type = "VECTOR"} : index
    %32 = arith.addi %16, %c32 {ssbuffer.core_type = "VECTOR"} : index
    %33 = arith.maxsi %16, %c1 {ssbuffer.core_type = "VECTOR"} : index
    %34 = arith.minsi %32, %33 {ssbuffer.core_type = "VECTOR"} : index
    %35 = arith.subi %34, %16 {ssbuffer.core_type = "VECTOR"} : index
    %36 = arith.minsi %31, %c32 {ssbuffer.core_type = "VECTOR"} : index
    %37 = arith.minsi %35, %c32 {ssbuffer.core_type = "VECTOR"} : index
    %38 = arith.cmpi slt, %36, %c32 {ssbuffer.core_type = "VECTOR"} : index
    %39 = arith.cmpi slt, %37, %c32 {ssbuffer.core_type = "VECTOR"} : index
    %40 = arith.ori %38, %39 {ssbuffer.core_type = "VECTOR"} : i1
    scf.if %40 {
      linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%cst : f16) outs(%alloc : memref<32x32xf16>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast_7[0, 0] [%36, %37] [1, 1] {ssbuffer.core_type = "VECTOR"} : memref<32x32xf16, strided<[1, 1], offset: ?>> to memref<?x?xf16, strided<[1, 1], offset: ?>>
    %subview_8 = memref.subview %alloc[0, 0] [%36, %37] [1, 1] {ssbuffer.core_type = "VECTOR"} : memref<32x32xf16> to memref<?x?xf16, strided<[32, 1]>>
    memref.copy %subview, %subview_8 {ssbuffer.core_type = "VECTOR"} : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1]>>
    %41 = bufferization.to_tensor %alloc restrict writable {ssbuffer.core_type = "VECTOR"} : memref<32x32xf16>
    %42 = arith.extf %41 {ssbuffer.core_type = "VECTOR"} : tensor<32x32xf16> to tensor<32x32xf32>
    %43 = linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%arg6 : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %44 = arith.mulf %26, %43 {ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %45 = linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%arg7 : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %46 = arith.mulf %42, %45 {ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %47 = arith.addf %44, %46 {ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %48 = arith.truncf %47 {ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> to tensor<32x32xf16>
    %extracted_slice = tensor.extract_slice %48[0, 0] [%36, %37] [1, 1] {ssbuffer.core_type = "VECTOR"} : tensor<32x32xf16> to tensor<?x?xf16>
    %subview_9 = memref.subview %reinterpret_cast[0, 0] [%36, %37] [1, 1] {ssbuffer.core_type = "VECTOR"} : memref<32x32xf16, strided<[1, 1], offset: ?>> to memref<?x?xf16, strided<[1, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_9 {ssbuffer.core_type = "VECTOR"} : (tensor<?x?xf16>, memref<?x?xf16, strided<[1, 1], offset: ?>>) -> ()
    return
  }
}

