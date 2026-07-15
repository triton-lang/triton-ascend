module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">}  {
func.func @sparse_flash_attention_prefill_kernel(%arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
  %c8_i32 = arith.constant {Undefined, ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} 8 : i32
  %c0_i32 = arith.constant {Undefined, ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
  %c1_i32 = arith.constant {Undefined, ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32

  scf.for %arg43 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
    scf.for %arg44 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {

      %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [1], sizes: [576], strides: [1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<576xf16, strided<[1], offset: ?>>
      %alloc = memref.alloc() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<576xf16>
      memref.copy %reinterpret_cast_8, %alloc {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<576xf16, strided<[1], offset: ?>> to memref<576xf16>
      %44 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<576xf16>

      %reinterpret_cast_9 = memref.reinterpret_cast %arg7 to offset: [1], sizes: [576], strides: [1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<576xf16, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %44 in writable %reinterpret_cast_9 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<576xf16>, memref<576xf16, strided<[1], offset: ?>>) -> ()

    } {Undefined, ssbuffer.block_id = 14 : i32}
    scf.for %arg45 = %c0_i32 to %c8_i32 step %c1_i32 : i32 {

      %reinterpret_cast_7 = memref.reinterpret_cast %arg7 to offset: [1], sizes: [128, 576], strides: [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<128x576xf16, strided<[?, 1], offset: ?>>
      %alloc_8 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x576xf16>
      memref.copy %reinterpret_cast_7, %alloc_8 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x576xf16, strided<[?, 1], offset: ?>> to memref<128x576xf16>
      %42 = bufferization.to_tensor %alloc_8 restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x576xf16>

      %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [1], sizes: [576], strides: [1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<576xf16, strided<[1], offset: ?>>
      %alloc = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : memref<576xf16>
      memref.copy %reinterpret_cast_8, %alloc {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : memref<576xf16, strided<[1], offset: ?>> to memref<576xf16>
      %44 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : memref<576xf16>

      %reinterpret_cast_9 = memref.reinterpret_cast %arg8 to offset: [1], sizes: [576], strides: [1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<576xf16, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %44 in writable %reinterpret_cast_9 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : (tensor<576xf16>, memref<576xf16, strided<[1], offset: ?>>) -> ()

    } {Undefined, ssbuffer.block_id = 16 : i32}

    %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [1], sizes: [128, 576], strides: [1, 1] {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x576xf16, strided<[?, 1], offset: ?>>
    %alloc_8 = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x576xf16>
    memref.copy %reinterpret_cast_7, %alloc_8 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x576xf16, strided<[?, 1], offset: ?>> to memref<128x576xf16>
    %42 = bufferization.to_tensor %alloc_8 restrict writable {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x576xf16>

  } {Undefined, ssbuffer.block_id = 17 : i32}
  return
}
}

// CHECK-LABEL: func.func @sparse_flash_attention_prefill_kernel
// CHECK: scf.for
// CHECK: scf.for
// CHECK: } {Undefined, ssbuffer.block_id = 14 : i32}
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = [[FLAG_1:[0-9]+]]
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE2>] flag = [[FLAG_1]]
// CHECK: scf.for
// CHECK: } {Undefined, ssbuffer.block_id = 16 : i32}
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"}[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = [[FLAG_2:[0-9]+]]
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"}[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = [[FLAG_2]]
// CHECK: memref.reinterpret_cast
