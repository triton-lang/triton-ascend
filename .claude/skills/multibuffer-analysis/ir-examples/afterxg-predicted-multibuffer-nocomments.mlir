module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @chunk_dplr_fwd_kernel_o(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c512 = arith.constant 512 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c16384_i32 = arith.constant 16384 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x16xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64x16xf32>) -> tensor<64x16xf32>
    %2 = tensor.empty() : tensor<64x64xf16>
    %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<64x64xf16>) -> tensor<64x64xf16>
    %4 = arith.divsi %arg15, %c8_i32 : i32
    %5 = arith.remsi %arg15, %c8_i32 : i32
    %6 = arith.addi %arg9, %c63_i32 : i32
    %7 = arith.divsi %6, %c64_i32 : i32
    %8 = arith.muli %4, %7 : i32
    %9 = arith.addi %8, %arg14 : i32
    %10 = arith.muli %4, %arg9 : i32
    %11 = arith.muli %10, %c8_i32 : i32
    %12 = arith.addi %11, %5 : i32
    %13 = arith.muli %12, %c128_i32 : i32
    %14 = arith.index_cast %13 : i32 to index
    %15 = arith.muli %arg14, %c64_i32 : i32
    %16 = arith.muli %9, %c8_i32 : i32
    %17 = arith.addi %16, %5 : i32
    %18 = arith.muli %17, %c16384_i32 : i32
    %19 = arith.index_cast %18 : i32 to index
    %20 = arith.muli %arg13, %c16_i32 : i32

    %buf19_0 = memref.alloc() : memref<64x32xf16>
    %buf19_1 = memref.alloc() : memref<64x32xf16>
    %buf22_0 = memref.alloc() : memref<32x16xf16>
    %buf22_1 = memref.alloc() : memref<32x16xf16>
    %c4_index = arith.constant 4 : index
    %c2_index = arith.constant 2 : index
    %c0_index = arith.constant 0 : index
    %c1_index = arith.constant 1 : index
    %true = arith.constant true
    %false = arith.constant false

    %21 = scf.for %arg16 = %c0_i32 to %c4_i32 step %c1_i32
        iter_args(

          %arg17 = %1,
          %flag19_0 = %false,
          %flag19_1 = %false,
          %prodCnt19 = %c0_index,
          %consCnt19 = %c0_index,
          %flag22_0 = %false,
          %flag22_1 = %false,
          %prodCnt22 = %c0_index,
          %consCnt22 = %c0_index
        ) -> (tensor<64x16xf32>, i1, i1, index, index, i1, i1, index, index) : i32 {

      %flag19_0_empty = arith.cmpi eq, %flag19_0, %false : i1
      %prod19_lt_tc = arith.cmpi ult, %prodCnt19, %c4_index : index
      %cond_prod_19_0 = arith.andi %flag19_0_empty, %prod19_lt_tc : i1
      %if_prod_19_0:2 = scf.if %cond_prod_19_0 -> (i1, index) {

        %iv_proj_19 = arith.index_cast %prodCnt19 : index to i32
        %84_p = arith.muli %iv_proj_19, %c32_i32 : i32
        %87_p = arith.maxsi %84_p, %c0_i32 : i32
        %88_p = arith.index_cast %87_p : i32 to index
        %89_p = arith.muli %86, %c1024 : index
        %90_p = arith.addi %89_p, %14 : index
        %92_p = arith.addi %90_p, %88_p : index
        %rc17_p = memref.reinterpret_cast %arg2 to offset: [%92_p], sizes: [64, 32], strides: [1024, 1] : memref<?xf16> to memref<64x32xf16, strided<[1024, 1], offset: ?>>
        %98_p = arith.subi %92_p, %14 : index
        %99_p = arith.divsi %98_p, %c1024 : index
        %100_p = arith.subi %91, %99_p : index
        %101_p = arith.maxsi %100_p, %c0 : index
        %102_p = arith.minsi %101_p, %c64 : index
        %103_p = arith.remsi %98_p, %c1024 : index
        %104_p = arith.subi %c128, %103_p : index
        %105_p = arith.maxsi %104_p, %c0 : index
        %106_p = arith.minsi %105_p, %c32 : index
        %107_p = arith.subi %c0_i32, %15 : i32
        %108_p = arith.maxsi %107_p, %c0_i32 : i32
        %109_p = arith.index_cast %108_p : i32 to index
        %110_p = arith.minsi %109_p, %102_p : index
        %111_p = arith.subi %102_p, %110_p : index
        %116_p = arith.subi %106_p, %115 : index
        %117_p = arith.cmpi slt, %111_p, %c64 : index
        %118_p = arith.cmpi slt, %116_p, %c32 : index
        %119_p = arith.ori %117_p, %118_p : i1
        scf.if %119_p {
          linalg.fill ins(%cst : f16) outs(%buf19_0 : memref<64x32xf16>)
        } {hivm.unlikely_condition}
        %subview_20_p = memref.subview %rc17_p[0, 0] [%111_p, %116_p] [1, 1] : memref<64x32xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        %subview_21_p = memref.subview %buf19_0[%110_p, %115] [%111_p, %116_p] [1, 1] : memref<64x32xf16> to memref<?x?xf16, strided<[32, 1], offset: ?>>
        memref.copy %subview_20_p, %subview_21_p : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1], offset: ?>>
        %prodCnt19_plus_1 = arith.addi %prodCnt19, %c1_index : index
        scf.yield %true, %prodCnt19_plus_1 : i1, index
      } else {
        scf.yield %flag19_0, %prodCnt19 : i1, index
      }
      %flag19_0_next = %if_prod_19_0#0
      %prodCnt19_next_slot0 = %if_prod_19_0#1
      %flag19_1_empty = arith.cmpi eq, %flag19_1, %false : i1
      %prod19_lt_tc_1 = arith.cmpi ult, %prodCnt19_next_slot0, %c4_index : index
      %cond_prod_19_1 = arith.andi %flag19_1_empty, %prod19_lt_tc_1 : i1
      %if_prod_19_1:2 = scf.if %cond_prod_19_1 -> (i1, index) {
        %iv_proj_19 = arith.index_cast %prodCnt19_next_slot0 : index to i32

        %84_p = arith.muli %iv_proj_19, %c32_i32 : i32
        %87_p = arith.maxsi %84_p, %c0_i32 : i32
        %88_p = arith.index_cast %87_p : i32 to index
        %89_p = arith.muli %86, %c1024 : index
        %90_p = arith.addi %89_p, %14 : index
        %92_p = arith.addi %90_p, %88_p : index
        %rc17_p = memref.reinterpret_cast %arg2 to offset: [%92_p], sizes: [64, 32], strides: [1024, 1] : memref<?xf16> to memref<64x32xf16, strided<[1024, 1], offset: ?>>

        %prodCnt19_plus_1 = arith.addi %prodCnt19_next_slot0, %c1_index : index
        scf.yield %true, %prodCnt19_plus_1 : i1, index
      } else {
        scf.yield %flag19_1, %prodCnt19_next_slot0 : i1, index
      }
      %flag19_1_next = %if_prod_19_1#0
      %prodCnt19_final = %if_prod_19_1#1
      %flag22_0_empty = arith.cmpi eq, %flag22_0, %false : i1
      %prod22_lt_tc = arith.cmpi ult, %prodCnt22, %c4_index : index
      %cond_prod_22_0 = arith.andi %flag22_0_empty, %prod22_lt_tc : i1
      %if_prod_22_0:2 = scf.if %cond_prod_22_0 -> (i1, index) {
        %iv_proj_22 = arith.index_cast %prodCnt22 : index to i32

        %prodCnt22_plus_1 = arith.addi %prodCnt22, %c1_index : index
        scf.yield %true, %prodCnt22_plus_1 : i1, index
      } else {
        scf.yield %flag22_0, %prodCnt22 : i1, index
      }
      %flag22_0_next = %if_prod_22_0#0
      %prodCnt22_next_slot0 = %if_prod_22_0#1
      %flag22_1_empty = arith.cmpi eq, %flag22_1, %false : i1
      %prod22_lt_tc_1 = arith.cmpi ult, %prodCnt22_next_slot0, %c4_index : index
      %cond_prod_22_1 = arith.andi %flag22_1_empty, %prod22_lt_tc_1 : i1
      %if_prod_22_1:2 = scf.if %cond_prod_22_1 -> (i1, index) {
        %iv_proj_22 = arith.index_cast %prodCnt22_next_slot0 : index to i32

        %prodCnt22_plus_1 = arith.addi %prodCnt22_next_slot0, %c1_index : index
        scf.yield %true, %prodCnt22_plus_1 : i1, index
      } else {
        scf.yield %flag22_1, %prodCnt22_next_slot0 : i1, index
      }
      %flag22_1_next = %if_prod_22_1#0
      %prodCnt22_final = %if_prod_22_1#1

      %target19 = arith.remsi %consCnt19, %c2_index : index
      %to_tensor_19_0 = bufferization.to_tensor %buf19_0 restrict writable : memref<64x32xf16>
      %to_tensor_19_1 = bufferization.to_tensor %buf19_1 restrict writable : memref<64x32xf16>
      %target19_eq_0 = arith.cmpi eq, %target19, %c0_index : index
      %selected_19 = arith.select %target19_eq_0, %to_tensor_19_0, %to_tensor_19_1 : tensor<64x32xf16>
      %target22 = arith.remsi %consCnt22, %c2_index : index
      %to_tensor_22_0 = bufferization.to_tensor %buf22_0 restrict writable : memref<32x16xf16>
      %to_tensor_22_1 = bufferization.to_tensor %buf22_1 restrict writable : memref<32x16xf16>
      %target22_eq_0 = arith.cmpi eq, %target22, %c0_index : index
      %selected_22 = arith.select %target22_eq_0, %to_tensor_22_0, %to_tensor_22_1 : tensor<32x16xf16>
      %84 = arith.muli %arg16, %c32_i32 : i32
      %85 = arith.maxsi %15, %c0_i32 : i32
      %86 = arith.index_cast %85 : i32 to index
      %87 = arith.maxsi %84, %c0_i32 : i32
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.muli %86, %c1024 : index
      %90 = arith.addi %89, %14 : index
      %91 = arith.index_cast %arg9 : i32 to index
      %92 = arith.addi %90, %88 : index
      %reinterpret_cast_17 = memref.reinterpret_cast %arg2 to offset: [%92], sizes: [64, 32], strides: [1024, 1] : memref<?xf16> to memref<64x32xf16, strided<[1024, 1], offset: ?>>
      %93 = arith.maxsi %20, %c0_i32 : i32
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.muli %88, %c128 : index
      %96 = arith.addi %95, %19 : index
      %97 = arith.addi %96, %94 : index
      %reinterpret_cast_18 = memref.reinterpret_cast %arg7 to offset: [%97], sizes: [32, 16], strides: [128, 1] : memref<?xf16> to memref<32x16xf16, strided<[128, 1], offset: ?>>
      %141 = linalg.matmul {input_precision = "ieee"} ins(%selected_19, %selected_22 : tensor<64x32xf16>, tensor<32x16xf16>) outs(%arg17 : tensor<64x16xf32>) -> tensor<64x16xf32>

      %if_rel_19_0 = scf.if %target19_eq_0 -> (i1) {
        scf.yield %false : i1
      } else {
        scf.yield %flag19_0_next : i1
      }
      %flag19_0_final = %if_rel_19_0
      %target19_eq_1 = arith.cmpi eq, %target19, %c1_index : index
      %if_rel_19_1 = scf.if %target19_eq_1 -> (i1) {
        scf.yield %false : i1
      } else {
        scf.yield %flag19_1_next : i1
      }
      %flag19_1_final = %if_rel_19_1
      %if_rel_22_0 = scf.if %target22_eq_0 -> (i1) {
        scf.yield %false : i1
      } else {
        scf.yield %flag22_0_next : i1
      }
      %flag22_0_final = %if_rel_22_0
      %target22_eq_1 = arith.cmpi eq, %target22, %c1_index : index
      %if_rel_22_1 = scf.if %target22_eq_1 -> (i1) {
        scf.yield %false : i1
      } else {
        scf.yield %flag22_1_next : i1
      }
      %flag22_1_final = %if_rel_22_1
      %consCnt19_next = arith.addi %consCnt19, %c1_index : index
      %consCnt22_next = arith.addi %consCnt22, %c1_index : index
      scf.yield %141,
                %flag19_0_final, %flag19_1_final, %prodCnt19_final, %consCnt19_next,
                %flag22_0_final, %flag22_1_final, %prodCnt22_final, %consCnt22_next
                : tensor<64x16xf32>, i1, i1, index, index, i1, i1, index, index
    }
    %22 = arith.muli %12, %c64_i32 : i32
    %23 = arith.index_cast %22 : i32 to index
    %24 = arith.maxsi %15, %c0_i32 : i32
    %25 = arith.index_cast %24 : i32 to index
    %26 = arith.muli %25, %c512 : index
    %27 = arith.addi %26, %23 : index
    %28 = arith.index_cast %arg9 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<?xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<?xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
    %29 = arith.maxsi %20, %c0_i32 : i32
    %30 = arith.index_cast %29 : i32 to index
    %31 = arith.muli %25, %c1024 : index
    %32 = arith.addi %31, %14 : index
    %33 = arith.addi %32, %30 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%33], sizes: [64, 16], strides: [1024, 1] : memref<?xf16> to memref<64x16xf16, strided<[1024, 1], offset: ?>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%33], sizes: [64, 16], strides: [1024, 1] : memref<?xf16> to memref<64x16xf16, strided<[1024, 1], offset: ?>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg8 to offset: [%33], sizes: [64, 16], strides: [1024, 1] : memref<?xf16> to memref<64x16xf16, strided<[1024, 1], offset: ?>>
    %34 = tensor.empty() : tensor<64xi32>
    %35 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%34 : tensor<64xi32>) attrs =  {tt.from_make_range, tt.make_range_offset = 0 : index, tt.make_range_size = 64 : index} {
    ^bb0(%out: i32):
      %84 = linalg.index 0 : index
      %85 = arith.index_cast %84 : index to i32
      linalg.yield %85 : i32
    } -> tensor<64xi32>
    %36 = tensor.empty() : tensor<64x64xi32>
    %broadcasted = linalg.broadcast ins(%35 : tensor<64xi32>) outs(%36 : tensor<64x64xi32>) dimensions = [1]
    %broadcasted_5 = linalg.broadcast ins(%35 : tensor<64xi32>) outs(%36 : tensor<64x64xi32>) dimensions = [0]
    %37 = arith.cmpi sge, %broadcasted, %broadcasted_5 : tensor<64x64xi32>
    %alloc = memref.alloc() : memref<64x64xf16>
    %38 = arith.divsi %26, %c512 : index
    %39 = arith.subi %28, %38 : index
    %40 = arith.maxsi %39, %c0 : index
    %41 = arith.minsi %40, %c64 : index
    %42 = arith.remsi %26, %c512 : index
    %43 = arith.subi %c64, %42 : index
    %44 = arith.maxsi %43, %c0 : index
    %45 = arith.minsi %44, %c64 : index
    %46 = arith.subi %c0_i32, %15 : i32
    %47 = arith.maxsi %46, %c0_i32 : i32
    %48 = arith.index_cast %47 : i32 to index
    %49 = arith.minsi %48, %41 : index
    %50 = arith.subi %41, %49 : index
    %51 = arith.minsi %45, %c0 : index
    %52 = arith.subi %45, %51 : index
    %53 = arith.cmpi slt, %50, %c64 : index
    %54 = arith.cmpi slt, %52, %c64 : index
    %55 = arith.ori %53, %54 : i1
    scf.if %55 {
      linalg.fill ins(%cst : f16) outs(%alloc : memref<64x64xf16>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0, 0] [%50, %52] [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16, strided<[512, 1], offset: ?>>
    %subview_6 = memref.subview %alloc[%49, %51] [%50, %52] [1, 1] : memref<64x64xf16> to memref<?x?xf16, strided<[64, 1], offset: ?>>
    memref.copy %subview, %subview_6 : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16, strided<[64, 1], offset: ?>>
    %56 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
    %alloc_7 = memref.alloc() : memref<64x64xf16>
    scf.if %55 {
      linalg.fill ins(%cst : f16) outs(%alloc_7 : memref<64x64xf16>)
    } {hivm.unlikely_condition}
    %subview_8 = memref.subview %reinterpret_cast_1[0, 0] [%50, %52] [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16, strided<[512, 1], offset: ?>>
    %subview_9 = memref.subview %alloc_7[%49, %51] [%50, %52] [1, 1] : memref<64x64xf16> to memref<?x?xf16, strided<[64, 1], offset: ?>>
    memref.copy %subview_8, %subview_9 : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16, strided<[64, 1], offset: ?>>
    %57 = bufferization.to_tensor %alloc_7 restrict writable : memref<64x64xf16>
    %58 = arith.select %37, %56, %3 : tensor<64x64xi1>, tensor<64x64xf16>
    %59 = arith.select %37, %57, %3 : tensor<64x64xi1>, tensor<64x64xf16>
    %alloc_10 = memref.alloc() : memref<64x16xf16>
    %60 = arith.subi %33, %14 : index
    %61 = arith.divsi %60, %c1024 : index
    %62 = arith.subi %28, %61 : index
    %63 = arith.maxsi %62, %c0 : index
    %64 = arith.minsi %63, %c64 : index
    %65 = arith.remsi %60, %c1024 : index
    %66 = arith.subi %c128, %65 : index
    %67 = arith.maxsi %66, %c0 : index
    %68 = arith.minsi %67, %c16 : index
    %69 = arith.minsi %48, %64 : index
    %70 = arith.subi %64, %69 : index
    %71 = arith.subi %c0_i32, %20 : i32
    %72 = arith.maxsi %71, %c0_i32 : i32
    %73 = arith.index_cast %72 : i32 to index
    %74 = arith.minsi %73, %68 : index
    %75 = arith.subi %68, %74 : index
    %76 = arith.cmpi slt, %70, %c64 : index
    %77 = arith.cmpi slt, %75, %c16 : index
    %78 = arith.ori %76, %77 : i1
    scf.if %78 {
      linalg.fill ins(%cst : f16) outs(%alloc_10 : memref<64x16xf16>)
    } {hivm.unlikely_condition}
    %subview_11 = memref.subview %reinterpret_cast_2[0, 0] [%70, %75] [1, 1] : memref<64x16xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
    %subview_12 = memref.subview %alloc_10[%69, %74] [%70, %75] [1, 1] : memref<64x16xf16> to memref<?x?xf16, strided<[16, 1], offset: ?>>
    memref.copy %subview_11, %subview_12 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[16, 1], offset: ?>>
    %79 = bufferization.to_tensor %alloc_10 restrict writable : memref<64x16xf16>
    %alloc_13 = memref.alloc() : memref<64x16xf16>
    scf.if %78 {
      linalg.fill ins(%cst : f16) outs(%alloc_13 : memref<64x16xf16>)
    } {hivm.unlikely_condition}
    %subview_14 = memref.subview %reinterpret_cast_3[0, 0] [%70, %75] [1, 1] : memref<64x16xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
    %subview_15 = memref.subview %alloc_13[%69, %74] [%70, %75] [1, 1] : memref<64x16xf16> to memref<?x?xf16, strided<[16, 1], offset: ?>>
    memref.copy %subview_14, %subview_15 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[16, 1], offset: ?>>
    %80 = bufferization.to_tensor %alloc_13 restrict writable : memref<64x16xf16>
    %81 = linalg.matmul {input_precision = "ieee"} ins(%58, %79 : tensor<64x64xf16>, tensor<64x16xf16>) outs(%21 : tensor<64x16xf32>) -> tensor<64x16xf32>
    %82 = linalg.matmul {input_precision = "ieee"} ins(%59, %80 : tensor<64x64xf16>, tensor<64x16xf16>) outs(%81 : tensor<64x16xf32>) -> tensor<64x16xf32>
    %83 = arith.truncf %82 : tensor<64x16xf32> to tensor<64x16xf16>
    %extracted_slice = tensor.extract_slice %83[%69, %74] [%70, %75] [1, 1] : tensor<64x16xf16> to tensor<?x?xf16>
    %subview_16 = memref.subview %reinterpret_cast_4[0, 0] [%70, %75] [1, 1] : memref<64x16xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_16 : (tensor<?x?xf16>, memref<?x?xf16, strided<[1024, 1], offset: ?>>) -> ()
    return
  }
}
