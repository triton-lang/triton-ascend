// ============================================================================
// INPUT: Before multibuffer
//   block_id=27: 每次迭代 memref.alloc + copy + to_tensor → %74 (在 scf.if 外)
//   scf.if %89 内 block_id=19: load broadcasted_45, compute arith.mulf %broadcasted_45, %74
//   问题: 每次迭代临时 alloc/free, tensor 跨 block (27→19)
// ============================================================================

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @before_multibuffer(%gm_data: memref<?xf32>, %cond: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %empty_64 = tensor.empty() : tensor<64x64xf32>

    scope.scope : () -> () {
      %init = linalg.fill ins(%cst : f32) outs(%empty_64 : tensor<64x64xf32>) -> tensor<64x64xf32>

      %result = scf.for %iv = %c0_i32 to %c32_i32 step %c1_i32
          iter_args(%acc = %init) -> (tensor<64x64xf32>) : i32 {

        // === block_id=27: 每次迭代临时 alloc, copy, to_tensor → %74 ===
        %alloc_20 = memref.alloc() {ssbuffer.block_id = 27 : i32} : memref<64x64xf32>
        %reinterpret_cast = memref.reinterpret_cast %gm_data to offset: [%c0],
          sizes: [64, 64], strides: [64, 1] {ssbuffer.block_id = 27 : i32}
          : memref<?xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
        %subview = memref.subview %reinterpret_cast[0, 0] [%c64, %c64] [1, 1]
          {ssbuffer.block_id = 27 : i32}
          : memref<64x64xf32, strided<[64, 1], offset: ?>> to
            memref<?x?xf32, strided<[64, 1], offset: ?>>
        %subview_21 = memref.subview %alloc_20[0, 0] [%c64, %c64] [1, 1]
          {ssbuffer.block_id = 27 : i32}
          : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview, %subview_21 {ssbuffer.block_id = 27 : i32}
          : memref<?x?xf32, strided<[64, 1], offset: ?>> to
            memref<?x?xf32, strided<[64, 1], offset: ?>>
        %74 = bufferization.to_tensor %alloc_20 restrict writable
          {ssbuffer.block_id = 27 : i32} : memref<64x64xf32>

        // === scf.if %cond: block_id=19 load + compute, 使用外面的 %74 ===
        %loop_result = scf.if %cond -> (tensor<64x64xf32>) {
          // block_id=19: 同层级 load broadcasted_45
          %alloc_40 = memref.alloc() {ssbuffer.block_id = 19 : i32} : memref<64x1xf32>
          %reinterpret_cast_41 = memref.reinterpret_cast %gm_data to offset: [%c0],
            sizes: [64, 1], strides: [64, 1] {ssbuffer.block_id = 19 : i32}
            : memref<?xf32> to memref<64x1xf32, strided<[64, 1], offset: ?>>
          %sub_42 = memref.subview %reinterpret_cast_41[0, 0] [%c64, %c1] [1, 1]
            {ssbuffer.block_id = 19 : i32}
            : memref<64x1xf32, strided<[64, 1], offset: ?>> to
              memref<?x?xf32, strided<[64, 1], offset: ?>>
          %sub_43 = memref.subview %alloc_40[0, 0] [%c64, %c1] [1, 1]
            {ssbuffer.block_id = 19 : i32}
            : memref<64x1xf32> to memref<?x?xf32, strided<[1, 1], offset: ?>>
          memref.copy %sub_42, %sub_43 {ssbuffer.block_id = 19 : i32}
            : memref<?x?xf32, strided<[64, 1], offset: ?>> to
              memref<?x?xf32, strided<[1, 1], offset: ?>>
          %148 = bufferization.to_tensor %alloc_40 restrict writable
            {ssbuffer.block_id = 19 : i32} : memref<64x1xf32>
          %collapsed = tensor.collapse_shape %148 [[0, 1]]
            {ssbuffer.block_id = 19 : i32} : tensor<64x1xf32> into tensor<64xf32>
          %broadcasted_45 = linalg.broadcast
            ins(%collapsed : tensor<64xf32>)
            outs(%empty_64 : tensor<64x64xf32>) dimensions = [1]
            {ssbuffer.block_id = 19 : i32}

          // ★ compute: %74(if外, block_id=27) + %broadcasted_45(if内, block_id=19)
          %mulf = arith.mulf %broadcasted_45, %74
            {DataUse, ssbuffer.block_id = 19 : i32} : tensor<64x64xf32>
          scf.yield %mulf : tensor<64x64xf32>
        } else {
          scf.yield %acc : tensor<64x64xf32>
        } {ssbuffer.block_id = 21 : i32}

        scf.yield %loop_result : tensor<64x64xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
