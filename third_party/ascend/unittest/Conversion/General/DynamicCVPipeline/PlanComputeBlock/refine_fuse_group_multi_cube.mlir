// RUN: triton-opt %s --allow-unregistered-dialect --plan-compute-block | FileCheck %s

// Regression test for refineFuseGroup pruning in PlanVectorBlockPass.
//
// One greedily-built VECTOR fuse group feeds two different CUBE matmuls:
//   - the first cube (block_id = 4) consumes %139 / %140;
//   - the second cube (block_id = 5) consumes %131 (an arith.truncf 64x32xf32->bf16).
// The pruning must keep only the first cube's dependency cone in one VECTOR
// block and evict the second cube's cone (the truncf chain) into a *separate*
// VECTOR block. Before the fix every op landed in the same VECTOR block.
//
// Check that the first cube's operands come from one VECTOR block while the
// truncf feeding the second cube lands in a different VECTOR block.

// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"{{.*}} : tensor<128x64xf32>
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"{{.*}} ins({{.*}} : tensor<128x64xf32>, tensor<64x32xf32>)
// CHECK: arith.truncf {{.*}}ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"{{.*}} : tensor<64x32xf32> to tensor<64x32xbf16>
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @chunk_gated_delta_rule_bwd_kernel_dhu_k128_blockdim128(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: f32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %c4096 = arith.constant 4096 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c524288_i64 = arith.constant 524288 : i64
    %c-1_i32 = arith.constant {MixUse} -1 : i32
    %c16384_i64 = arith.constant 16384 : i64
    %c63_i32 = arith.constant {MixUse} 63 : i32
    %c64 = arith.constant {DataUse} 64 : index
    %c0 = arith.constant {DataUse} 0 : index
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant {MixUse} 64 : i32
    %c1_i32 = arith.constant {MixUse} 1 : i32
    %c128_i64 = arith.constant 128 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x32xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %2 = tensor.empty() : tensor<64xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
    %4 = tensor.empty() : tensor<128x32xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<128x32xf32>) -> tensor<128x32xf32>
    %6 = arith.divsi %arg16, %c32_i32 : i32
    %7 = arith.remsi %arg16, %c32_i32 : i32
    %8 = arith.muli %6, %arg11 : i32
    %9 = arith.addi %arg11, %c63_i32 {MixUse} : i32
    %10 = arith.divsi %9, %c64_i32 {MixUse} : i32
    %11 = arith.muli %6, %10 : i32
    %12 = arith.muli %8, %c32_i32 : i32
    %13 = arith.addi %12, %7 : i32
    %14 = arith.extsi %13 : i32 to i64
    %15 = arith.muli %14, %c128_i64 : i64
    %16 = arith.index_cast %15 : i64 to index
    %17 = arith.muli %11, %c32_i32 : i32
    %18 = arith.addi %17, %7 : i32
    %19 = arith.extsi %18 : i32 to i64
    %20 = arith.muli %19, %c16384_i64 : i64
    %21 = arith.subi %10, %c1_i32 {MixUse} : i32
    %22 = arith.muli %arg15, %c32_i32 : i32
    %23 = arith.index_cast %12 : i32 to index
    %24 = arith.index_cast %7 : i32 to index
    %25 = arith.addi %23, %24 : index
    %26 = linalg.fill ins(%arg10 : f32) outs(%4 : tensor<128x32xf32>) -> tensor<128x32xf32>
    %27 = scf.for %arg18 = %c-1_i32 to %21 step %c1_i32 iter_args(%arg19 = %5) -> (tensor<128x32xf32>)  : i32 {
      %28 = arith.subi %21, %arg18 {MixUse} : i32
      %29 = arith.addi %28, %c-1_i32 {MixUse} : i32
      %30 = arith.extsi %29 : i32 to i64
      %31 = arith.muli %30, %c524288_i64 : i64
      %32 = arith.addi %20, %31 : i64
      %33 = arith.index_cast %32 : i64 to index
      %34 = arith.maxsi %22, %c0_i32 : i32
      %35 = arith.index_cast %34 : i32 to index
      %36 = arith.addi %33, %35 : index
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%36], sizes: [128, 32], strides: [128, 1] : memref<?xbf16> to memref<128x32xbf16, strided<[128, 1], offset: ?>>
      %37 = arith.truncf %arg19 {DataUse} : tensor<128x32xf32> to tensor<128x32xbf16>
      %38 = arith.divsi %35, %c128 : index
      %39 = arith.subi %c128, %38 : index
      %40 = arith.maxsi %39, %c0 : index
      %41 = arith.minsi %40, %c128 : index
      %42 = arith.remsi %35, %c128 : index
      %43 = arith.subi %c128, %42 : index
      %44 = arith.maxsi %43, %c0 : index
      %45 = arith.minsi %44, %c32 : index
      %46 = arith.minsi %41, %c0 : index
      %47 = arith.subi %41, %46 : index
      %48 = arith.subi %c0_i32, %22 : i32
      %49 = arith.maxsi %48, %c0_i32 : i32
      %50 = arith.index_cast %49 : i32 to index
      %51 = arith.minsi %50, %45 : index
      %52 = arith.subi %45, %51 : index
      %extracted_slice = tensor.extract_slice %37[%46, %51] [%47, %52] [1, 1] : tensor<128x32xbf16> to tensor<?x?xbf16>
      %subview = memref.subview %reinterpret_cast[0, 0] [%47, %52] [1, 1] : memref<128x32xbf16, strided<[128, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[128, 1], offset: ?>>) -> ()
      %53 = arith.muli %28, %c64_i32 : i32
      %54 = arith.minsi %53, %arg11 : i32
      %55 = arith.subi %54, %c1_i32 : i32
      %56 = arith.addi %8, %55 : i32
      %57 = arith.muli %56, %c32_i32 : i32
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.addi %58, %24 : index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%59], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %60 = memref.load %reinterpret_cast_1[%c0] : memref<1xf32, strided<[1], offset: ?>>
      %61 = arith.muli %29, %c64_i32 {MixUse} : i32
      %62 = arith.maxsi %61, %c0_i32 : i32
      %63 = arith.index_cast %62 : i32 to index
      %64 = arith.muli %63, %c32 : index
      %65 = arith.addi %64, %25 : index
      %66 = arith.index_cast %arg11 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%65], sizes: [64], strides: [32] : memref<?xf32> to memref<64xf32, strided<[32], offset: ?>>
      %alloc = memref.alloc() : memref<64xf32>
      %67 = arith.divsi %64, %c32 : index
      %68 = arith.subi %66, %67 : index
      %69 = arith.maxsi %68, %c0 : index
      %70 = arith.minsi %69, %c64 : index
      %71 = arith.subi %c0_i32, %61 : i32
      %72 = arith.maxsi %71, %c0_i32 : i32
      %73 = arith.index_cast %72 : i32 to index
      %74 = arith.minsi %73, %70 : index
      %75 = arith.subi %70, %74 : index
      %76 = arith.cmpi slt, %75, %c64 : index
      scf.if %76 {
        linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<64xf32>)
      } {hivm.unlikely_condition}
      %subview_3 = memref.subview %reinterpret_cast_2[0] [%75] [1] : memref<64xf32, strided<[32], offset: ?>> to memref<?xf32, strided<[32], offset: ?>>
      %subview_4 = memref.subview %alloc[%74] [%75] [1] : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>>
      memref.copy %subview_3, %subview_4 : memref<?xf32, strided<[32], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %77 = bufferization.to_tensor %alloc restrict writable : memref<64xf32>
      %78 = tensor.empty() : tensor<1xf32>
      %inserted = tensor.insert %60 into %78[%c0] : tensor<1xf32>
      %79 = math.exp %inserted {DataUse} : tensor<1xf32>
      %extracted = tensor.extract %79[%c0] {DataUse} : tensor<1xf32>
      %80 = math.exp %77 {DataUse} : tensor<64xf32>
      %81 = arith.muli %63, %c4096 : index
      %82 = arith.addi %81, %16 : index
      %83 = arith.addi %82, %35 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg8 to offset: [%83], sizes: [64, 32], strides: [4096, 1] : memref<?xbf16> to memref<64x32xbf16, strided<[4096, 1], offset: ?>>
      %reinterpret_cast_6 = memref.reinterpret_cast %arg9 to offset: [%83], sizes: [64, 32], strides: [4096, 1] : memref<?xbf16> to memref<64x32xbf16, strided<[4096, 1], offset: ?>>
      %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [%83], sizes: [64, 32], strides: [4096, 1] : memref<?xbf16> to memref<64x32xbf16, strided<[4096, 1], offset: ?>>
      %alloc_8 = memref.alloc() : memref<64x32xbf16>
      %84 = arith.subi %83, %16 : index
      %85 = arith.divsi %84, %c4096 : index
      %86 = arith.subi %66, %85 : index
      %87 = arith.maxsi %86, %c0 : index
      %88 = arith.minsi %87, %c64 : index
      %89 = arith.remsi %84, %c4096 : index
      %90 = arith.subi %c128, %89 : index
      %91 = arith.maxsi %90, %c0 : index
      %92 = arith.minsi %91, %c32 : index
      %93 = arith.minsi %73, %88 : index
      %94 = arith.subi %88, %93 : index
      %95 = arith.minsi %50, %92 : index
      %96 = arith.subi %92, %95 : index
      %97 = arith.cmpi slt, %94, %c64 : index
      %98 = arith.cmpi slt, %96, %c32 : index
      %99 = arith.ori %97, %98 : i1
      scf.if %99 {
        linalg.fill ins(%cst : bf16) outs(%alloc_8 : memref<64x32xbf16>)
      } {hivm.unlikely_condition}
      %subview_9 = memref.subview %reinterpret_cast_7[0, 0] [%94, %96] [1, 1] : memref<64x32xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>>
      %subview_10 = memref.subview %alloc_8[%93, %95] [%94, %96] [1, 1] : memref<64x32xbf16> to memref<?x?xbf16, strided<[32, 1], offset: ?>>
      memref.copy %subview_9, %subview_10 : memref<?x?xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[32, 1], offset: ?>>
      %100 = bufferization.to_tensor %alloc_8 restrict writable : memref<64x32xbf16>
      %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%82], sizes: [64, 128], strides: [4096, 1] : memref<?xbf16> to memref<64x128xbf16, strided<[4096, 1], offset: ?>>
      %alloc_12 = memref.alloc() : memref<64x128xbf16>
      %101 = arith.divsi %81, %c4096 : index
      %102 = arith.subi %66, %101 : index
      %103 = arith.maxsi %102, %c0 : index
      %104 = arith.minsi %103, %c64 : index
      %105 = arith.remsi %81, %c4096 : index
      %106 = arith.subi %c128, %105 : index
      %107 = arith.maxsi %106, %c0 : index
      %108 = arith.minsi %107, %c128 : index
      %109 = arith.minsi %73, %104 : index
      %110 = arith.subi %104, %109 : index
      %111 = arith.minsi %108, %c0 : index
      %112 = arith.subi %108, %111 : index
      %113 = arith.cmpi slt, %110, %c64 : index
      %114 = arith.cmpi slt, %112, %c128 : index
      %115 = arith.ori %113, %114 : i1
      scf.if %115 {
        linalg.fill ins(%cst : bf16) outs(%alloc_12 : memref<64x128xbf16>)
      } {hivm.unlikely_condition}
      %subview_13 = memref.subview %reinterpret_cast_11[0, 0] [%110, %112] [1, 1] : memref<64x128xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>>
      %subview_14 = memref.subview %alloc_12[%109, %111] [%110, %112] [1, 1] : memref<64x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
      memref.copy %subview_13, %subview_14 : memref<?x?xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
      %116 = bufferization.to_tensor %alloc_12 restrict writable : memref<64x128xbf16>
      %117 = linalg.matmul {input_precision = "ieee"} ins(%116, %37 : tensor<64x128xbf16>, tensor<128x32xbf16>) outs(%1 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %118 = linalg.fill ins(%60 : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
      %119 = arith.subf %118, %77 {DataUse} : tensor<64xf32>
      %120 = math.exp %119 {DataUse} : tensor<64xf32>
      %121 = arith.index_cast %61 {DataUse} : i32 to index
      %122 = arith.addi %121, %c64 {DataUse} : index
      %123 = arith.index_cast %arg11 {DataUse} : i32 to index
      %124 = arith.maxsi %121, %123 {DataUse} : index
      %125 = arith.minsi %122, %124 {DataUse} : index
      %126 = arith.subi %125, %121 {DataUse} : index
      %extracted_slice_15 = tensor.extract_slice %120[0] [%126] [1] {DataUse} : tensor<64xf32> to tensor<?xf32>
      %inserted_slice = tensor.insert_slice %extracted_slice_15 into %3[0] [%126] [1] {DataUse} : tensor<?xf32> into tensor<64xf32>
      %broadcasted = linalg.broadcast ins(%inserted_slice : tensor<64xf32>) outs(%0 : tensor<64x32xf32>) dimensions = [1] 
      %127 = arith.mulf %117, %broadcasted {DataUse} : tensor<64x32xf32>
      %alloc_16 = memref.alloc() : memref<64x32xbf16>
      scf.if %99 {
        linalg.fill ins(%cst : bf16) outs(%alloc_16 : memref<64x32xbf16>)
      } {hivm.unlikely_condition}
      %subview_17 = memref.subview %reinterpret_cast_5[0, 0] [%94, %96] [1, 1] : memref<64x32xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>>
      %subview_18 = memref.subview %alloc_16[%93, %95] [%94, %96] [1, 1] : memref<64x32xbf16> to memref<?x?xbf16, strided<[32, 1], offset: ?>>
      memref.copy %subview_17, %subview_18 : memref<?x?xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[32, 1], offset: ?>>
      %128 = bufferization.to_tensor %alloc_16 restrict writable : memref<64x32xbf16>
      %129 = arith.extf %128 {DataUse} : tensor<64x32xbf16> to tensor<64x32xf32>
      %130 = arith.addf %127, %129 {DataUse} : tensor<64x32xf32>
      %131 = arith.truncf %130 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
      %extracted_slice_19 = tensor.extract_slice %131[%93, %95] [%94, %96] [1, 1] : tensor<64x32xbf16> to tensor<?x?xbf16>
      %subview_20 = memref.subview %reinterpret_cast_6[0, 0] [%94, %96] [1, 1] : memref<64x32xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_19 in writable %subview_20 : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[4096, 1], offset: ?>>) -> ()
      %reinterpret_cast_21 = memref.reinterpret_cast %arg4 to offset: [%82], sizes: [64, 128], strides: [4096, 1] : memref<?xbf16> to memref<64x128xbf16, strided<[4096, 1], offset: ?>>
      %alloc_22 = memref.alloc() : memref<64x128xbf16>
      scf.if %115 {
        linalg.fill ins(%cst : bf16) outs(%alloc_22 : memref<64x128xbf16>)
      } {hivm.unlikely_condition}
      %subview_23 = memref.subview %reinterpret_cast_21[0, 0] [%110, %112] [1, 1] : memref<64x128xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>>
      %subview_24 = memref.subview %alloc_22[%109, %111] [%110, %112] [1, 1] : memref<64x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
      memref.copy %subview_23, %subview_24 : memref<?x?xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
      %132 = bufferization.to_tensor %alloc_22 restrict writable : memref<64x128xbf16>
      %133 = tensor.empty() : tensor<128x64xbf16>
      %transposed = linalg.transpose ins(%132 : tensor<64x128xbf16>) outs(%133 : tensor<128x64xbf16>) permutation = [1, 0] 
      %reinterpret_cast_25 = memref.reinterpret_cast %arg2 to offset: [%82], sizes: [64, 128], strides: [4096, 1] : memref<?xbf16> to memref<64x128xbf16, strided<[4096, 1], offset: ?>>
      %alloc_26 = memref.alloc() : memref<64x128xbf16>
      scf.if %115 {
        linalg.fill ins(%cst : bf16) outs(%alloc_26 : memref<64x128xbf16>)
      } {hivm.unlikely_condition}
      %subview_27 = memref.subview %reinterpret_cast_25[0, 0] [%110, %112] [1, 1] : memref<64x128xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>>
      %subview_28 = memref.subview %alloc_26[%109, %111] [%110, %112] [1, 1] : memref<64x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
      memref.copy %subview_27, %subview_28 : memref<?x?xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
      %134 = bufferization.to_tensor %alloc_26 restrict writable : memref<64x128xbf16>
      %transposed_29 = linalg.transpose ins(%134 : tensor<64x128xbf16>) outs(%133 : tensor<128x64xbf16>) permutation = [1, 0] 
      %135 = linalg.fill ins(%extracted : f32) outs(%4 : tensor<128x32xf32>) -> tensor<128x32xf32>
      %136 = arith.mulf %arg19, %135 {DataUse} : tensor<128x32xf32>
      %137 = arith.extf %transposed_29 {DataUse} : tensor<128x64xbf16> to tensor<128x64xf32>
      %138 = tensor.empty() : tensor<128x64xf32>
      %broadcasted_30 = linalg.broadcast ins(%80 : tensor<64xf32>) outs(%138 : tensor<128x64xf32>) dimensions = [0] 
      %139 = arith.mulf %137, %broadcasted_30 {DataUse} : tensor<128x64xf32>
      %140 = arith.extf %100 {DataUse} : tensor<64x32xbf16> to tensor<64x32xf32>
      %141 = linalg.matmul {input_precision = "ieee"} ins(%139, %140 : tensor<128x64xf32>, tensor<64x32xf32>) outs(%5 : tensor<128x32xf32>) -> tensor<128x32xf32>
      %142 = arith.mulf %141, %26 {DataUse} : tensor<128x32xf32>
      %143 = linalg.matmul {input_precision = "ieee"} ins(%transposed, %131 : tensor<128x64xbf16>, tensor<64x32xbf16>) outs(%5 : tensor<128x32xf32>) -> tensor<128x32xf32>
      %144 = arith.subf %142, %143 {DataUse} : tensor<128x32xf32>
      %145 = arith.addf %136, %144 {DataUse} : tensor<128x32xf32>
      scf.yield %145 : tensor<128x32xf32>
    } {Undefined}
    return
  }
}