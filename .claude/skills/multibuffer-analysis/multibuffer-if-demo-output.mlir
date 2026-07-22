// ============================================================================
// OUTPUT: After multibuffer (round-robin double buffer on memref via arith.select)
//
// diff:
//   (a) memref.alloc<64x64xf32> 从循环内 hoist 到循环外 x2 (slot0/slot1)
//   (b) 循环内 arith.remsi %iv, 2 + arith.select 选 memref slot
//   (c) memref.copy → to_tensor 都通过同一个 %alloc_20 (选中 slot), %74 仍在 if 外
//   (d) scf.if 内 compute 完全不变
//   (e) 不插入核内同步
// ============================================================================

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @after_multibuffer(%gm_data: memref<?xf32>, %cond: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %empty_64 = tensor.empty() : tensor<64x64xf32>

    scope.scope : () -> () {
      %init = linalg.fill ins(%cst : f32) outs(%empty_64 : tensor<64x64xf32>) -> tensor<64x64xf32>

      // === (a) 双 buffer slot 分配在循环外 ===
      %slot0 = memref.alloc() {ssbuffer.block_id = 27 : i32} : memref<64x64xf32>
      %slot1 = memref.alloc() {ssbuffer.block_id = 27 : i32} : memref<64x64xf32>

      %result = scf.for %iv = %c0_i32 to %c32_i32 step %c1_i32
          iter_args(%acc = %init) -> (tensor<64x64xf32>) : i32 {

        // === (b) round-robin: arith.select 在 memref 层面选 slot ===
        %idx = arith.remsi %iv, %c2_i32 {ssbuffer.block_id = 27 : i32} : i32
        %is_slot1 = arith.cmpi eq, %idx, %c1_i32 {ssbuffer.block_id = 27 : i32} : i32
        %alloc_20 = arith.select %is_slot1, %slot1, %slot0
          {ssbuffer.block_id = 27 : i32} : memref<64x64xf32>

        // === (c) block_id=27: copy + to_tensor 仍在 if 外, 通过 %alloc_20 ===
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

        // === (d) scf.if 内 compute 完全不变 ===
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

          // ★ compute: 不变, %74 仍在 if 外
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
