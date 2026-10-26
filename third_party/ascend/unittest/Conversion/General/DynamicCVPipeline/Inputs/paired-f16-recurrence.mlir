#map = affine_map<(d0) -> (d0)>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 4 : i32, ssbuffer.load_store_buf_count = 1 : i32} {
  func.func @_attn_fwd_paired_f16(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf8E4M3FN> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf8E4M3FN> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf8E4M3FN> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg9: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c32768 = arith.constant 32768 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0xFC00 : f16
    %c512_i32 = arith.constant 512 : i32
    %c-8192_i32 = arith.constant -8192 : i32
    %c1536_i32 = arith.constant 1536 : i32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c24_i32 = arith.constant 24 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1536_i64 = arith.constant 1536 : i64
    %c128_i32 = arith.constant 128 : i32
    %c3_i64 = arith.constant 3 : i64
    %c12_i64 = arith.constant 12 : i64
    %c196608_i64 = arith.constant 196608 : i64
    %c12_i32 = arith.constant 12 : i32
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant 0.0883883461 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = tensor.empty() : tensor<256xi32>
    %3 = linalg.fill ins(%c32_i32 : i32) outs(%2 : tensor<256xi32>) -> tensor<256xi32>
    %4 = linalg.fill ins(%c16_i32 : i32) outs(%2 : tensor<256xi32>) -> tensor<256xi32>
    %5 = linalg.fill ins(%c2_i32 : i32) outs(%2 : tensor<256xi32>) -> tensor<256xi32>
    %6 = tensor.empty() : tensor<32x8x16x16xf32>
    %7 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<32x8x16x16xf32>) -> tensor<32x8x16x16xf32>
    %8 = tensor.empty() : tensor<128xf16>
    %9 = linalg.fill ins(%cst_0 : f16) outs(%8 : tensor<128xf16>) -> tensor<128xf16>
    %10 = tensor.empty() : tensor<128xf32>
    %11 = linalg.fill ins(%cst : f32) outs(%10 : tensor<128xf32>) -> tensor<128xf32>
    %12 = tensor.empty() : tensor<128x128xf32>
    %13 = linalg.fill ins(%cst_1 : f32) outs(%12 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %14 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : tensor<256xi32>) attrs =  {tt.from_make_range, tt.make_range_offset = 0 : index, tt.make_range_size = 256 : index} {
    ^bb0(%out: i32):
      %25 = linalg.index 0 : index
      %26 = arith.index_cast %25 : index to i32
      linalg.yield %26 : i32
    } -> tensor<256xi32>
    %15 = arith.divsi %14, %3 : tensor<256xi32>
    %16 = arith.muli %15, %3 : tensor<256xi32>
    %17 = arith.remsi %14, %4 : tensor<256xi32>
    %18 = arith.muli %17, %5 : tensor<256xi32>
    %19 = arith.addi %16, %18 : tensor<256xi32>
    %20 = arith.remsi %14, %3 : tensor<256xi32>
    %21 = arith.divsi %20, %4 : tensor<256xi32>
    %22 = arith.addi %19, %21 : tensor<256xi32>
    %23 = arith.trunci %22 : tensor<256xi32> to tensor<256xi8>
    %24 = tensor.empty() : tensor<16x8x16x32xf8E4M3FN>
    %c48_i32 = arith.constant 48 : i32
    scf.for %arg16 = %arg13 to %c48_i32 step %c24_i32  : i32 {
      %25 = arith.divsi %arg16, %c12_i32 : i32
      %26 = arith.remsi %arg16, %c12_i32 : i32
      %27 = arith.extsi %25 : i32 to i64
      %28 = arith.muli %27, %c196608_i64 : i64
      %29 = arith.muli %27, %c12_i64 : i64
      %30 = arith.muli %27, %c3_i64 : i64
      %31 = arith.index_cast %28 : i64 to index
      %32 = arith.muli %26, %c128_i32 : i32
      %33 = arith.maxsi %32, %c0_i32 : i32
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.muli %34, %c128 : index
      %36 = arith.addi %35, %31 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [128, 128], strides: [128, 1] : memref<?xf8E4M3FN> to memref<128x128xf8E4M3FN, strided<[128, 1], offset: ?>>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg8 to offset: [%36], sizes: [128, 128], strides: [128, 1] : memref<?xf32> to memref<128x128xf32, strided<[128, 1], offset: ?>>
      %alloc = memref.alloc() : memref<128x128xf8E4M3FN>
      memref.copy %reinterpret_cast, %alloc : memref<128x128xf8E4M3FN, strided<[128, 1], offset: ?>> to memref<128x128xf8E4M3FN>
      %37 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf8E4M3FN> to tensor<128x128xf8E4M3FN>
      %38 = arith.index_cast %29 : i64 to index
      %39 = arith.index_cast %26 : i32 to index
      %40 = arith.addi %38, %39 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%40], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %41 = memref.load %reinterpret_cast_4[%c0] : memref<1xf32, strided<[1], offset: ?>>
      %inserted = tensor.insert %41 into %0[%c0] : tensor<1xf32>
      %42 = arith.mulf %inserted, %1 : tensor<1xf32>
      %extracted = tensor.extract %42[%c0] : tensor<1xf32>
      %43:3 = scf.for %arg17 = %c0_i32 to %c1536_i32 step %c512_i32 iter_args(%arg18 = %9, %arg19 = %11, %arg20 = %13) -> (tensor<128xf16>, tensor<128xf32>, tensor<128x128xf32>)  : i32 {
        %52 = arith.maxsi %arg17, %c0_i32 : i32
        %53 = arith.index_cast %52 : i32 to index
        %54 = arith.muli %53, %c128 : index
        %55 = arith.addi %54, %31 : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%55], sizes: [256, 128], strides: [128, 1] : memref<?xf8E4M3FN> to memref<256x128xf8E4M3FN, strided<[128, 1], offset: ?>>
        %56 = arith.addi %55, %c32768 : index
        %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%56], sizes: [256, 128], strides: [128, 1] : memref<?xf8E4M3FN> to memref<256x128xf8E4M3FN, strided<[128, 1], offset: ?>>
        %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [%55], sizes: [512, 128], strides: [128, 1] : memref<?xf8E4M3FN> to memref<512x128xf8E4M3FN, strided<[128, 1], offset: ?>>
        %57 = arith.divsi %arg17, %c512_i32 : i32
        %58 = arith.index_cast %30 : i64 to index
        %59 = arith.index_cast %57 : i32 to index
        %60 = arith.addi %58, %59 : index
        %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%60], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
        %61 = memref.load %reinterpret_cast_10[%c0] : memref<1xf32, strided<[1], offset: ?>>
        %inserted_11 = tensor.insert %extracted into %0[%c0] : tensor<1xf32>
        %inserted_12 = tensor.insert %61 into %0[%c0] : tensor<1xf32>
        %62 = arith.mulf %inserted_11, %inserted_12 : tensor<1xf32>
        %extracted_13 = tensor.extract %62[%c0] : tensor<1xf32>
        %63 = arith.bitcast %extracted_13 : f32 to i32
        %64 = arith.andi %63, %c-8192_i32 : i32
        %65 = arith.bitcast %64 : i32 to f32
        %alloc_14 = memref.alloc() : memref<256x128xf8E4M3FN>
        memref.copy %reinterpret_cast_6, %alloc_14 : memref<256x128xf8E4M3FN, strided<[128, 1], offset: ?>> to memref<256x128xf8E4M3FN>
        %66 = bufferization.to_tensor %alloc_14 restrict writable : memref<256x128xf8E4M3FN> to tensor<256x128xf8E4M3FN>
        %67 = tensor.empty() : tensor<128x256xf8E4M3FN>
        %transposed = linalg.transpose ins(%66 : tensor<256x128xf8E4M3FN>) outs(%67 : tensor<128x256xf8E4M3FN>) permutation = [1, 0]
        %68 = tensor.empty() : tensor<128x256xf32>
        %69 = linalg.fill ins(%cst_1 : f32) outs(%68 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %70 = linalg.matmul {input_precision = "ieee"} ins(%37, %transposed : tensor<128x128xf8E4M3FN>, tensor<128x256xf8E4M3FN>) outs(%69 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %71 = hivm.hir.convert_layout %70 output_shape [16, 8, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<128x256xf32>) -> tensor<16x8x16x16xf32>
        %inserted_slice = tensor.insert_slice %71 into %7[0, 0, 0, 0] [16, 8, 16, 16] [1, 1, 1, 1] : tensor<16x8x16x16xf32> into tensor<32x8x16x16xf32>
        %alloc_15 = memref.alloc() : memref<256x128xf8E4M3FN>
        memref.copy %reinterpret_cast_7, %alloc_15 : memref<256x128xf8E4M3FN, strided<[128, 1], offset: ?>> to memref<256x128xf8E4M3FN>
        %72 = bufferization.to_tensor %alloc_15 restrict writable : memref<256x128xf8E4M3FN> to tensor<256x128xf8E4M3FN>
        %transposed_16 = linalg.transpose ins(%72 : tensor<256x128xf8E4M3FN>) outs(%67 : tensor<128x256xf8E4M3FN>) permutation = [1, 0]
        %73 = linalg.matmul {input_precision = "ieee"} ins(%37, %transposed_16 : tensor<128x128xf8E4M3FN>, tensor<128x256xf8E4M3FN>) outs(%69 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %74 = hivm.hir.convert_layout %73 output_shape [16, 8, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<128x256xf32>) -> tensor<16x8x16x16xf32>
        %inserted_slice_17 = tensor.insert_slice %74 into %inserted_slice[16, 0, 0, 0] [16, 8, 16, 16] [1, 1, 1, 1] : tensor<16x8x16x16xf32> into tensor<32x8x16x16xf32>
        %75 = linalg.fill ins(%65 : f32) outs(%6 : tensor<32x8x16x16xf32>) -> tensor<32x8x16x16xf32>
        %76 = arith.mulf %inserted_slice_17, %75 : tensor<32x8x16x16xf32>
        %77 = arith.truncf %76 : tensor<32x8x16x16xf32> to tensor<32x8x16x16xf16>
        %78:3 = hivm.hir.custom {symbol = "__builtin_online_softmax_nz", arg_attrs = [{}, {}, {}, {}], gm_addr_args_indices = array<i32>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%77, %arg18, %23 : tensor<32x8x16x16xf16>, tensor<128xf16>, tensor<256xi8>) outs(%24, %8, %10 : tensor<16x8x16x32xf8E4M3FN>, tensor<128xf16>, tensor<128xf32>) -> (tensor<16x8x16x32xf8E4M3FN>, tensor<128xf16>, tensor<128xf32>)
        %79 = arith.extf %arg18 : tensor<128xf16> to tensor<128xf32>
        %80 = arith.extf %78#1 : tensor<128xf16> to tensor<128xf32>
        %81 = arith.subf %79, %80 : tensor<128xf32>
        %82 = math.exp %81 : tensor<128xf32>
        %83 = arith.mulf %arg19, %82 : tensor<128xf32>
        %84 = arith.addf %83, %78#2 : tensor<128xf32>
        %alloc_19 = memref.alloc() : memref<512x128xf8E4M3FN>
        memref.copy %reinterpret_cast_8, %alloc_19 : memref<512x128xf8E4M3FN, strided<[128, 1], offset: ?>> to memref<512x128xf8E4M3FN>
        %85 = bufferization.to_tensor %alloc_19 restrict writable : memref<512x128xf8E4M3FN> to tensor<512x128xf8E4M3FN>
        %86 = hivm.hir.convert_layout %78#0 output_shape [128, 512] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>} : (tensor<16x8x16x32xf8E4M3FN>) -> tensor<128x512xf8E4M3FN>
        %87 = linalg.matmul {input_precision = "ieee"} ins(%86, %85 : tensor<128x512xf8E4M3FN>, tensor<512x128xf8E4M3FN>) outs(%13 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %reinterpret_cast_21 = memref.reinterpret_cast %arg7 to offset: [%60], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
        %92 = memref.load %reinterpret_cast_21[%c0] : memref<1xf32, strided<[1], offset: ?>>
        %93 = linalg.fill ins(%92 : f32) outs(%12 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %94 = arith.mulf %87, %93 : tensor<128x128xf32>
        %broadcasted_22 = linalg.broadcast ins(%82 : tensor<128xf32>) outs(%12 : tensor<128x128xf32>) dimensions = [1]
        %95 = arith.mulf %arg20, %broadcasted_22 : tensor<128x128xf32>
        %96 = arith.addf %95, %94 : tensor<128x128xf32>
        scf.yield %78#1, %84, %96 : tensor<128xf16>, tensor<128xf32>, tensor<128x128xf32>
      }
      %44 = arith.muli %27, %c1536_i64 : i64
      %45 = arith.index_cast %44 : i64 to index
      %46 = arith.index_cast %32 : i32 to index
      %47 = arith.addi %45, %46 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg9 to offset: [%47], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
      %48 = arith.extf %43#0 : tensor<128xf16> to tensor<128xf32>
      %49 = math.log %43#1 : tensor<128xf32>
      %50 = arith.addf %48, %49 : tensor<128xf32>
      bufferization.materialize_in_destination %50 in writable %reinterpret_cast_5 : (tensor<128xf32>, memref<128xf32, strided<[1], offset: ?>>) -> ()
      %broadcasted = linalg.broadcast ins(%43#1 : tensor<128xf32>) outs(%12 : tensor<128x128xf32>) dimensions = [1]
      %51 = arith.divf %43#2, %broadcasted : tensor<128x128xf32>
      bufferization.materialize_in_destination %51 in writable %reinterpret_cast_3 : (tensor<128x128xf32>, memref<128x128xf32, strided<[128, 1], offset: ?>>) -> ()
    }
    return
  }
}
