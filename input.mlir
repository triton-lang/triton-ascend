module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @kernel_da_bwd_q_u(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg10: i32, %arg11: f32, %arg12: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg13: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : bf16
    %c32 = arith.constant {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} 32 : index
    %c640 = arith.constant {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} 640 : index
    %c0 = arith.constant {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %cst_0 = arith.constant {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} 1.000000e+00 : f32
    %cst_1 = arith.constant {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} 1.000000e+06 : f32
    %c0_i8 = arith.constant {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} 0 : i8
    %cst_2 = arith.constant {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} -1.000000e+06 : f32
    %c31_i32 = arith.constant {MixUse, ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} 31 : i32
    %c1_i32 = arith.constant {Undefined, ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
    %c0_i32 = arith.constant {MixUse, ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %c32_i32 = arith.constant {MixUse, ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} 32 : i32
    %c5_i32 = arith.constant {MixUse, ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} 5 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} 128 : i32
    %cst_3 = arith.constant {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %0 = tensor.empty() {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
    %1 = linalg.fill {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_3 : f32) outs(%0 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %2 = tensor.empty() {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %3 = linalg.fill {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_2 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %4 = tensor.empty() {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xi8>
    %5 = linalg.fill {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} ins(%c0_i8 : i8) outs(%4 : tensor<32x32xi8>) -> tensor<32x32xi8>
    %6 = linalg.fill {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_1 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %7 = linalg.fill {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_0 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %8 = linalg.fill {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_3 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %9 = tensor.empty() {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32xi32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%9 : tensor<32xi32>) attrs =  {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR", tt.from_make_range, tt.make_range_offset = 0 : index, tt.make_range_size = 32 : index} {
    ^bb0(%out: i32):
      %19 = linalg.index 0 : index
      %20 = arith.index_cast %19 : index to i32
      linalg.yield %20 : i32
    } -> tensor<32xi32>
    %expanded = tensor.expand_shape %10 [[0, 1]] output_shape [32, 1] {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32xi32> into tensor<32x1xi32>
    %expanded_4 = tensor.expand_shape %10 [[0, 1]] output_shape [1, 32] {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32xi32> into tensor<1x32xi32>
    %reinterpret_cast = memref.reinterpret_cast %arg13 to offset: [0], sizes: [32, 32], strides: [64, 1] {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<32x32xi8, strided<[64, 1]>>
    %alloc = memref.alloc() {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} : memref<32x32xi8>
    memref.copy %reinterpret_cast, %alloc {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<32x32xi8, strided<[64, 1]>> to memref<32x32xi8>
    %11 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<32x32xi8>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg12 to offset: [0], sizes: [32, 32], strides: [64, 1] {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<32x32xi8, strided<[64, 1]>>
    %alloc_6 = memref.alloc() {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} : memref<32x32xi8>
    memref.copy %reinterpret_cast_5, %alloc_6 {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<32x32xi8, strided<[64, 1]>> to memref<32x32xi8>
    %12 = bufferization.to_tensor %alloc_6 restrict writable {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<32x32xi8>
    %13 = arith.remsi %arg19, %arg16 {Undefined, ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %14 = linalg.fill {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} ins(%arg11 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %15 = arith.cmpi ne, %11, %5 {DataUse, ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xi8>
    %16 = linalg.fill {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} ins(%arg11 : f32) outs(%0 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %17 = arith.cmpi ne, %12, %5 {DataUse, ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xi8>
    %c128 = arith.constant {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"} 128 : index
    %c32_7 = arith.constant {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"} 32 : index
    %c640_8 = arith.constant {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"} 640 : index
    %c0_9 = arith.constant {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"} 0 : index
    %c31_i32_10 = arith.constant {MixUse, ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"} 31 : i32
    %c0_i32_11 = arith.constant {MixUse, ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"} 0 : i32
    %c32_i32_12 = arith.constant {MixUse, ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"} 32 : i32
    %c5_i32_13 = arith.constant {MixUse, ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"} 5 : i32
    %c128_i32_14 = arith.constant {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"} 128 : i32
    %18:2 = scf.for %arg22 = %c0_i32 to %arg10 step %c1_i32 iter_args(%arg23 = %c0_i32, %arg24 = %c0_i32) -> (i32, i32)  : i32 {
      %19 = arith.index_cast %arg22 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %reinterpret_cast_15 = memref.reinterpret_cast %arg9 to offset: [%19], sizes: [1], strides: [1] {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %20 = memref.load %reinterpret_cast_15[%c0_9] {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : memref<1xi32, strided<[1], offset: ?>>
      %21 = arith.subi %20, %arg23 {MixUse, ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : i32
      %22 = arith.addi %21, %c31_i32_10 {MixUse, ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : i32
      %23 = arith.divsi %22, %c32_i32_12 {MixUse, ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : i32
      %24 = arith.addi %arg24, %23 {MixUse, ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : i32
      %25 = arith.addi %arg14, %arg23 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : i32
      %26 = arith.addi %arg14, %20 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : i32
      %27 = arith.index_cast %arg22 {ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
      %reinterpret_cast_16 = memref.reinterpret_cast %arg9 to offset: [%27], sizes: [1], strides: [1] {ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %28 = memref.load %reinterpret_cast_16[%c0] {ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : memref<1xi32, strided<[1], offset: ?>>
      %29 = arith.subi %28, %arg23 {MixUse, ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %30 = arith.addi %29, %c31_i32 {MixUse, ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %31 = arith.divsi %30, %c32_i32 {MixUse, ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %32 = arith.addi %arg24, %31 {MixUse, ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %33 = arith.muli %arg24, %c5_i32 {Undefined, ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %34 = arith.remsi %33, %arg16 {Undefined, ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %35 = arith.subi %13, %34 {Undefined, ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %36 = arith.addi %35, %arg16 {Undefined, ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %37 = arith.remsi %36, %arg16 {Undefined, ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %38 = arith.addi %33, %37 {Undefined, ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %39 = arith.muli %32, %c5_i32 {Undefined, ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %40 = tensor.empty() {ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x1xi32>
      %41 = linalg.fill {ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} ins(%28 : i32) outs(%40 : tensor<32x1xi32>) -> tensor<32x1xi32>
      %42 = tensor.empty() {ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x32xi32>
      %43 = linalg.fill {ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} ins(%28 : i32) outs(%42 : tensor<1x32xi32>) -> tensor<1x32xi32>
      scf.for %arg25 = %38 to %39 step %arg16  : i32 {
        %44 = arith.divsi %arg25, %c5_i32_13 {MixUse, ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : i32
        %45 = arith.subi %44, %arg24 {MixUse, ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : i32
        %46 = arith.muli %45, %c32_i32_12 {MixUse, ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : i32
        %47 = arith.addi %arg23, %46 {MixUse, ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : i32
        %48 = arith.index_cast %47 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %alloc_17 = memref.alloc() {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
        %49 = arith.addi %48, %c32_7 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : index
        %50 = arith.index_cast %20 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %51 = arith.maxsi %48, %50 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : index
        %52 = arith.minsi %49, %51 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : index
        %53 = arith.subi %52, %48 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : index
        %54 = arith.cmpi slt, %53, %c32_7 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : index
        %alloc_18 = memref.alloc() {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
        scf.if %54 {
          linalg.fill {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%alloc_18 : memref<32x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 19 : i32}
        scf.if %54 {
          linalg.fill {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%alloc_17 : memref<32x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 19 : i32}
        %55 = arith.remsi %arg25, %c5_i32_13 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : i32
        %56 = arith.muli %55, %c128_i32_14 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : i32
        %57 = arith.index_cast %56 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %58 = arith.muli %48, %c640_8 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : index
        %59 = arith.addi %57, %58 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%59], sizes: [32, 128], strides: [640, 1] {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<32x128xbf16, strided<[640, 1], offset: ?>>
        %subview = memref.subview %reinterpret_cast_19[0, 0] [%53, 128] [1, 1] {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16, strided<[640, 1], offset: ?>> to memref<?x128xbf16, strided<[640, 1], offset: ?>>
        %subview_20 = memref.subview %alloc_17[0, 0] [%53, 128] [1, 1] {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
        memref.copy %subview, %subview_20 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : memref<?x128xbf16, strided<[640, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
        %60 = bufferization.to_tensor %alloc_17 restrict writable {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
        %61 = arith.divsi %55, %c5_i32_13 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : i32
        %62 = arith.muli %61, %c128_i32_14 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : i32
        %63 = arith.index_cast %62 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %64 = arith.muli %48, %c128 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : index
        %65 = arith.addi %63, %64 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_21 = memref.reinterpret_cast %arg3 to offset: [%65], sizes: [32, 128], strides: [128, 1] {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<32x128xbf16, strided<[128, 1], offset: ?>>
        %subview_22 = memref.subview %reinterpret_cast_21[0, 0] [%53, 128] [1, 1] {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
        %subview_23 = memref.subview %alloc_18[0, 0] [%53, 128] [1, 1] {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
        memref.copy %subview_22, %subview_23 {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : memref<?x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
        %66 = bufferization.to_tensor %alloc_18 restrict writable {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
        %67 = tensor.empty() {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x32xbf16>
        %transposed = linalg.transpose ins(%66 : tensor<32x128xbf16>) outs(%67 : tensor<128x32xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"}
        %68 = tensor.empty() {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
        %cst_24 = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
        %69 = linalg.fill {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_24 : f32) outs(%68 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %70 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} ins(%60, %transposed : tensor<32x128xbf16>, tensor<128x32xbf16>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %alloc_25 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
        %alloc_26 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
        scf.if %54 {
          linalg.fill {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%alloc_26 : memref<32x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 20 : i32}
        scf.if %54 {
          linalg.fill {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%alloc_25 : memref<32x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 20 : i32}
        %reinterpret_cast_27 = memref.reinterpret_cast %arg5 to offset: [%59], sizes: [32, 128], strides: [640, 1] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<32x128xbf16, strided<[640, 1], offset: ?>>
        %subview_28 = memref.subview %reinterpret_cast_27[0, 0] [%53, 128] [1, 1] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16, strided<[640, 1], offset: ?>> to memref<?x128xbf16, strided<[640, 1], offset: ?>>
        %subview_29 = memref.subview %alloc_25[0, 0] [%53, 128] [1, 1] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
        memref.copy %subview_28, %subview_29 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} : memref<?x128xbf16, strided<[640, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
        %71 = bufferization.to_tensor %alloc_25 restrict writable {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
        %reinterpret_cast_30 = memref.reinterpret_cast %arg4 to offset: [%65], sizes: [32, 128], strides: [128, 1] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<32x128xbf16, strided<[128, 1], offset: ?>>
        %subview_31 = memref.subview %reinterpret_cast_30[0, 0] [%53, 128] [1, 1] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
        %subview_32 = memref.subview %alloc_26[0, 0] [%53, 128] [1, 1] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
        memref.copy %subview_31, %subview_32 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} : memref<?x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
        %72 = bufferization.to_tensor %alloc_26 restrict writable {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
        %transposed_33 = linalg.transpose ins(%72 : tensor<32x128xbf16>) outs(%67 : tensor<128x32xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"}
        %73 = tensor.empty() {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
        %cst_34 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
        %74 = linalg.fill {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_34 : f32) outs(%73 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %75 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} ins(%71, %transposed_33 : tensor<32x128xbf16>, tensor<128x32xbf16>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %76 = arith.addi %25, %46 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "CUBE"} : i32
        %77 = arith.index_cast %76 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %78 = arith.addi %77, %c32_7 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "CUBE"} : index
        %79 = arith.index_cast %26 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %80 = arith.maxsi %77, %79 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "CUBE"} : index
        %81 = arith.minsi %78, %80 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "CUBE"} : index
        %82 = arith.subi %81, %77 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "CUBE"} : index
        %83 = arith.cmpi slt, %82, %c32_7 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "CUBE"} : index
        %alloc_35 = memref.alloc() {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
        scf.if %83 {
          linalg.fill {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%alloc_35 : memref<32x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 22 : i32}
        %84 = arith.muli %77, %c128 {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} : index
        %85 = arith.addi %63, %84 {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_36 = memref.reinterpret_cast %arg3 to offset: [%85], sizes: [32, 128], strides: [128, 1] {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<32x128xbf16, strided<[128, 1], offset: ?>>
        %subview_37 = memref.subview %reinterpret_cast_36[0, 0] [%82, 128] [1, 1] {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
        %subview_38 = memref.subview %alloc_35[0, 0] [%82, 128] [1, 1] {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
        memref.copy %subview_37, %subview_38 {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} : memref<?x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
        %86 = bufferization.to_tensor %alloc_35 restrict writable {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
        %transposed_39 = linalg.transpose ins(%86 : tensor<32x128xbf16>) outs(%67 : tensor<128x32xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"}
        %87 = tensor.empty() {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
        %cst_40 = arith.constant {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
        %88 = linalg.fill {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_40 : f32) outs(%87 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %89 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} ins(%60, %transposed_39 : tensor<32x128xbf16>, tensor<128x32xbf16>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %alloc_41 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
        scf.if %83 {
          linalg.fill {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%alloc_41 : memref<32x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 23 : i32}
        %reinterpret_cast_42 = memref.reinterpret_cast %arg4 to offset: [%85], sizes: [32, 128], strides: [128, 1] {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<32x128xbf16, strided<[128, 1], offset: ?>>
        %subview_43 = memref.subview %reinterpret_cast_42[0, 0] [%82, 128] [1, 1] {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
        %subview_44 = memref.subview %alloc_41[0, 0] [%82, 128] [1, 1] {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
        memref.copy %subview_43, %subview_44 {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"} : memref<?x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
        %90 = bufferization.to_tensor %alloc_41 restrict writable {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
        %transposed_45 = linalg.transpose ins(%90 : tensor<32x128xbf16>) outs(%67 : tensor<128x32xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"}
        %91 = tensor.empty() {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
        %cst_46 = arith.constant {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
        %92 = linalg.fill {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_46 : f32) outs(%91 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %93 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"} ins(%71, %transposed_45 : tensor<32x128xbf16>, tensor<128x32xbf16>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %94 = arith.divsi %arg25, %c5_i32 {MixUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %95 = arith.subi %94, %arg24 {MixUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %96 = arith.muli %95, %c32_i32 {MixUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %97 = arith.addi %arg23, %96 {MixUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %98 = arith.index_cast %97 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %99 = arith.addi %98, %c32 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : index
        %100 = arith.index_cast %28 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %101 = arith.maxsi %98, %100 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : index
        %102 = arith.minsi %99, %101 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : index
        %103 = arith.subi %102, %98 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : index
        %104 = arith.cmpi slt, %103, %c32 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : index
        %alloc_47 = memref.alloc() {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : memref<32xf32>
        %alloc_48 = memref.alloc() {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : memref<32xf32>
        scf.if %104 {
          linalg.fill {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_3 : f32) outs(%alloc_48 : memref<32xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 35 : i32}
        scf.if %104 {
          linalg.fill {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_3 : f32) outs(%alloc_47 : memref<32xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 35 : i32}
        %105 = arith.remsi %arg25, %c5_i32 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %106 = arith.muli %105, %arg15 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %107 = arith.index_cast %106 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %108 = arith.addi %107, %98 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : index
        %reinterpret_cast_49 = memref.reinterpret_cast %arg6 to offset: [%108], sizes: [32], strides: [1] {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
        %reinterpret_cast_50 = memref.reinterpret_cast %arg7 to offset: [%108], sizes: [32], strides: [1] {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
        %subview_51 = memref.subview %reinterpret_cast_50[0] [%103] [1] {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %subview_52 = memref.subview %alloc_47[0] [%103] [1] {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : memref<32xf32> to memref<?xf32, strided<[1]>>
        memref.copy %subview_51, %subview_52 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
        %109 = bufferization.to_tensor %alloc_47 restrict writable {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : memref<32xf32>
        %subview_53 = memref.subview %reinterpret_cast_49[0] [%103] [1] {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %subview_54 = memref.subview %alloc_48[0] [%103] [1] {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : memref<32xf32> to memref<?xf32, strided<[1]>>
        memref.copy %subview_53, %subview_54 {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
        %110 = bufferization.to_tensor %alloc_48 restrict writable {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : memref<32xf32>
        %111 = linalg.fill {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} ins(%97 : i32) outs(%40 : tensor<32x1xi32>) -> tensor<32x1xi32>
        %112 = arith.addi %111, %expanded {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x1xi32>
        %113 = arith.cmpi slt, %112, %41 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x1xi32>
        %114 = linalg.fill {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} ins(%97 : i32) outs(%42 : tensor<1x32xi32>) -> tensor<1x32xi32>
        %115 = arith.addi %114, %expanded_4 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x32xi32>
        %116 = arith.cmpi slt, %115, %43 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x32xi32>
        %117 = tensor.empty() {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xi1>
        %collapsed = tensor.collapse_shape %113 [[0, 1]] {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x1xi1> into tensor<32xi1>
        %broadcasted = linalg.broadcast ins(%collapsed : tensor<32xi1>) outs(%117 : tensor<32x32xi1>) dimensions = [1]  {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"}
        %collapsed_55 = tensor.collapse_shape %116 [[0, 1]] {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x32xi1> into tensor<32xi1>
        %broadcasted_56 = linalg.broadcast ins(%collapsed_55 : tensor<32xi1>) outs(%117 : tensor<32x32xi1>) dimensions = [0]  {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"}
        %118 = arith.andi %broadcasted, %broadcasted_56 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xi1>
        %119 = arith.mulf %70, %14 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %120 = arith.uitofp %118 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xi1> to tensor<32x32xf32>
        %121 = arith.subf %120, %7 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %122 = arith.mulf %121, %6 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %123 = arith.addf %119, %122 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %124 = arith.select %15, %123, %3 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xi1>, tensor<32x32xf32>
        %broadcasted_57 = linalg.broadcast ins(%109 : tensor<32xf32>) outs(%2 : tensor<32x32xf32>) dimensions = [1]  {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"}
        %125 = arith.subf %124, %broadcasted_57 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %126 = math.exp %125 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %broadcasted_58 = linalg.broadcast ins(%110 : tensor<32xf32>) outs(%2 : tensor<32x32xf32>) dimensions = [1]  {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"}
        %127 = arith.subf %75, %broadcasted_58 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %128 = arith.mulf %126, %127 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %129 = arith.truncf %128 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> to tensor<32x32xbf16>
        %130 = arith.mulf %89, %14 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %131 = arith.addf %130, %122 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %132 = arith.select %17, %131, %3 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xi1>, tensor<32x32xf32>
        %133 = arith.subf %132, %broadcasted_57 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %134 = math.exp %133 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %135 = arith.subf %93, %broadcasted_58 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %136 = arith.mulf %134, %135 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %137 = arith.truncf %136 {DataUse, ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> to tensor<32x32xbf16>
        %broadcasted_59 = linalg.broadcast ins(%109 : tensor<32xf32>) outs(%0 : tensor<32x128xf32>) dimensions = [1]  {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"}
        %broadcasted_60 = linalg.broadcast ins(%110 : tensor<32xf32>) outs(%0 : tensor<32x128xf32>) dimensions = [1]  {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"}
        %138 = tensor.empty() {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x128xf32>
        %cst_61 = arith.constant {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
        %139 = linalg.fill {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_61 : f32) outs(%138 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %140 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "CUBE"} ins(%137, %86 : tensor<32x32xbf16>, tensor<32x128xbf16>) outs(%1 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %141 = tensor.empty() {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x128xf32>
        %cst_62 = arith.constant {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
        %142 = linalg.fill {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_62 : f32) outs(%141 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %143 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "CUBE"} ins(%129, %66 : tensor<32x32xbf16>, tensor<32x128xbf16>) outs(%1 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %144 = arith.mulf %143, %16 {DataUse, ssbuffer.block_id = 36 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
        %145 = arith.addf %144, %1 {DataUse, ssbuffer.block_id = 36 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
        %146 = arith.mulf %140, %16 {DataUse, ssbuffer.block_id = 36 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
        %147 = arith.addf %145, %146 {DataUse, ssbuffer.block_id = 36 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
        %148 = arith.divsi %96, %c128_i32 {Undefined, ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %149 = arith.muli %148, %c128_i32 {Undefined, ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %150 = arith.divsi %149, %c32_i32 {Undefined, ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %151 = scf.for %arg26 = %150 to %95 step %c1_i32 iter_args(%arg27 = %147) -> (tensor<32x128xf32>)  : i32 {
          %158 = arith.muli %arg26, %c32_i32_12 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : i32
          %159 = arith.addi %25, %158 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : i32
          %160 = arith.index_cast %159 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %alloc_65 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
          %161 = arith.addi %160, %c32_7 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : index
          %162 = arith.maxsi %160, %79 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : index
          %163 = arith.minsi %161, %162 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : index
          %164 = arith.subi %163, %160 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : index
          %165 = arith.cmpi slt, %164, %c32_7 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : index
          scf.if %165 {
            linalg.fill {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%alloc_65 : memref<32x128xbf16>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 9 : i32}
          %166 = arith.muli %160, %c128 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : index
          %167 = arith.addi %63, %166 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : index
          %reinterpret_cast_66 = memref.reinterpret_cast %arg3 to offset: [%167], sizes: [32, 128], strides: [128, 1] {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<32x128xbf16, strided<[128, 1], offset: ?>>
          %subview_67 = memref.subview %reinterpret_cast_66[0, 0] [%164, 128] [1, 1] {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
          %subview_68 = memref.subview %alloc_65[0, 0] [%164, 128] [1, 1] {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
          memref.copy %subview_67, %subview_68 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : memref<?x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
          %168 = bufferization.to_tensor %alloc_65 restrict writable {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
          %transposed_69 = linalg.transpose ins(%168 : tensor<32x128xbf16>) outs(%67 : tensor<128x32xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"}
          %169 = tensor.empty() {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
          %cst_70 = arith.constant {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
          %170 = linalg.fill {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_70 : f32) outs(%169 : tensor<32x32xf32>) -> tensor<32x32xf32>
          %171 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} ins(%60, %transposed_69 : tensor<32x128xbf16>, tensor<128x32xbf16>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
          %alloc_71 = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
          scf.if %165 {
            linalg.fill {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%alloc_71 : memref<32x128xbf16>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 10 : i32}
          %reinterpret_cast_72 = memref.reinterpret_cast %arg4 to offset: [%167], sizes: [32, 128], strides: [128, 1] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<32x128xbf16, strided<[128, 1], offset: ?>>
          %subview_73 = memref.subview %reinterpret_cast_72[0, 0] [%164, 128] [1, 1] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
          %subview_74 = memref.subview %alloc_71[0, 0] [%164, 128] [1, 1] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
          memref.copy %subview_73, %subview_74 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} : memref<?x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
          %172 = bufferization.to_tensor %alloc_71 restrict writable {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} : memref<32x128xbf16>
          %transposed_75 = linalg.transpose ins(%172 : tensor<32x128xbf16>) outs(%67 : tensor<128x32xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"}
          %173 = tensor.empty() {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
          %cst_76 = arith.constant {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
          %174 = linalg.fill {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_76 : f32) outs(%173 : tensor<32x32xf32>) -> tensor<32x32xf32>
          %175 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} ins(%71, %transposed_75 : tensor<32x128xbf16>, tensor<128x32xbf16>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
          %176 = arith.mulf %171, %14 {DataUse, ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
          %177 = arith.subf %176, %broadcasted_57 {DataUse, ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
          %178 = math.exp %177 {DataUse, ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
          %179 = arith.subf %175, %broadcasted_58 {DataUse, ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
          %180 = arith.mulf %178, %179 {DataUse, ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
          %181 = arith.truncf %180 {DataUse, ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> to tensor<32x32xbf16>
          %182 = tensor.empty() {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x128xf32>
          %cst_77 = arith.constant {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
          %183 = linalg.fill {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_77 : f32) outs(%182 : tensor<32x128xf32>) -> tensor<32x128xf32>
          %184 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} ins(%181, %168 : tensor<32x32xbf16>, tensor<32x128xbf16>) outs(%1 : tensor<32x128xf32>) -> tensor<32x128xf32>
          %185 = arith.mulf %184, %16 {DataUse, ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
          %186 = arith.addf %arg27, %185 {DataUse, ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
          scf.yield {ssbuffer.core_type = "VECTOR"} %186 : tensor<32x128xf32>
        } {DataUse, ssbuffer.block_id = 40 : i32, ssbuffer.core_type = "VECTOR"}
        %152 = scf.for %arg26 = %c0_i32 to %148 step %c1_i32 iter_args(%arg27 = %151) -> (tensor<32x128xf32>)  : i32 {
          %158 = arith.muli %arg26, %c128_i32_14 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32
          %159 = arith.addi %25, %158 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32
          %160 = arith.index_cast %159 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %alloc_65 = memref.alloc() {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
          %161 = arith.addi %160, %c128 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %162 = arith.maxsi %160, %79 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %163 = arith.minsi %161, %162 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %164 = arith.subi %163, %160 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %165 = arith.cmpi slt, %164, %c128 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          scf.if %165 {
            linalg.fill {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%alloc_65 : memref<128x128xbf16>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 15 : i32}
          %166 = arith.muli %160, %c128 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %167 = arith.addi %63, %166 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %reinterpret_cast_66 = memref.reinterpret_cast %arg3 to offset: [%167], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<128x128xbf16, strided<[128, 1], offset: ?>>
          %subview_67 = memref.subview %reinterpret_cast_66[0, 0] [%164, 128] [1, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
          %subview_68 = memref.subview %alloc_65[0, 0] [%164, 128] [1, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
          memref.copy %subview_67, %subview_68 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : memref<?x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
          %168 = bufferization.to_tensor %alloc_65 restrict writable {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
          %169 = tensor.empty() {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xbf16>
          %transposed_69 = linalg.transpose ins(%168 : tensor<128x128xbf16>) outs(%169 : tensor<128x128xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"}
          %170 = tensor.empty() {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x128xf32>
          %cst_70 = arith.constant {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
          %171 = linalg.fill {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_70 : f32) outs(%170 : tensor<32x128xf32>) -> tensor<32x128xf32>
          %172 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} ins(%60, %transposed_69 : tensor<32x128xbf16>, tensor<128x128xbf16>) outs(%1 : tensor<32x128xf32>) -> tensor<32x128xf32>
          %alloc_71 = memref.alloc() {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
          scf.if %165 {
            linalg.fill {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%alloc_71 : memref<128x128xbf16>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 16 : i32}
          %reinterpret_cast_72 = memref.reinterpret_cast %arg4 to offset: [%167], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<128x128xbf16, strided<[128, 1], offset: ?>>
          %subview_73 = memref.subview %reinterpret_cast_72[0, 0] [%164, 128] [1, 1] {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
          %subview_74 = memref.subview %alloc_71[0, 0] [%164, 128] [1, 1] {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
          memref.copy %subview_73, %subview_74 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : memref<?x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
          %173 = bufferization.to_tensor %alloc_71 restrict writable {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
          %transposed_75 = linalg.transpose ins(%173 : tensor<128x128xbf16>) outs(%169 : tensor<128x128xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"}
          %174 = tensor.empty() {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x128xf32>
          %cst_76 = arith.constant {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
          %175 = linalg.fill {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_76 : f32) outs(%174 : tensor<32x128xf32>) -> tensor<32x128xf32>
          %176 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} ins(%71, %transposed_75 : tensor<32x128xbf16>, tensor<128x128xbf16>) outs(%1 : tensor<32x128xf32>) -> tensor<32x128xf32>
          %177 = arith.mulf %172, %16 {DataUse, ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
          %178 = arith.subf %177, %broadcasted_59 {DataUse, ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
          %179 = math.exp %178 {DataUse, ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
          %180 = arith.subf %176, %broadcasted_60 {DataUse, ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
          %181 = arith.mulf %179, %180 {DataUse, ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
          %182 = arith.truncf %181 {DataUse, ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32> to tensor<32x128xbf16>
          %183 = tensor.empty() {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x128xf32>
          %cst_77 = arith.constant {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
          %184 = linalg.fill {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_77 : f32) outs(%183 : tensor<32x128xf32>) -> tensor<32x128xf32>
          %185 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} ins(%182, %168 : tensor<32x128xbf16>, tensor<128x128xbf16>) outs(%1 : tensor<32x128xf32>) -> tensor<32x128xf32>
          %186 = arith.mulf %185, %16 {DataUse, ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
          %187 = arith.addf %arg27, %186 {DataUse, ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
          scf.yield {ssbuffer.core_type = "VECTOR"} %187 : tensor<32x128xf32>
        } {DataUse, ssbuffer.block_id = 41 : i32, ssbuffer.core_type = "VECTOR"}
        %153 = arith.muli %105, %c128_i32 {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %154 = arith.index_cast %153 {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %155 = arith.muli %98, %c640 {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} : index
        %156 = arith.addi %154, %155 {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} : index
        %reinterpret_cast_63 = memref.reinterpret_cast %arg8 to offset: [%156], sizes: [32, 128], strides: [640, 1] {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<32x128xbf16, strided<[640, 1], offset: ?>>
        %157 = arith.truncf %152 {DataUse, ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32> to tensor<32x128xbf16>
        %extracted_slice = tensor.extract_slice %157[0, 0] [%103, 128] [1, 1] {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xbf16> to tensor<?x128xbf16>
        %subview_64 = memref.subview %reinterpret_cast_63[0, 0] [%103, 128] [1, 1] {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} : memref<32x128xbf16, strided<[640, 1], offset: ?>> to memref<?x128xbf16, strided<[640, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_64 {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[640, 1], offset: ?>>) -> ()
      } {Undefined, ssbuffer.block_id = 42 : i32}
      scf.yield {ssbuffer.core_type = "CUBE, CUBE"} %20, %24 : i32, i32
    } {Undefined, ssbuffer.block_id = 43 : i32, ssbuffer.core_type = "CUBE, CUBE"}
    return
  }
}