// RUN: triton-opt --ta-simt-auto-blockify-v1="physical-vector-core-count=64 superblock-factor=1" %s | FileCheck %s --check-prefix=V1
// RUN: triton-opt --ta-simt-auto-blockify-v1="physical-vector-core-count=64 superblock-factor=2" %s | FileCheck %s --check-prefix=SUPERBLOCK
// RUN: triton-opt --ta-simt-auto-blockify-v1="physical-vector-core-count=64 superblock-factor=1" --ta-refine-simt-auto-blockify-v1-superblock="superblock-factor=2" %s | FileCheck %s --check-prefix=REFINE
// RUN: triton-opt --ta-simt-auto-blockify-v1="physical-vector-core-count=64 superblock-factor=32" %s | FileCheck %s --check-prefix=SUPERBLOCK32
// RUN: triton-opt --ta-simt-auto-blockify-v1="physical-vector-core-count=64 superblock-factor=1" --ta-refine-simt-auto-blockify-v1-superblock="superblock-factor=32" %s | FileCheck %s --check-prefix=REFINE32

// V1-LABEL: tt.func public @v1_keeps_tile_shape(
// V1-SAME: attributes {ta.auto_blockify_v1, ta.auto_blockify_v1.superblock_factor = 1 : i32}
// V1: %[[GX:.*]] = tt.get_num_programs x {{.*}} : i32
// V1: %[[GY:.*]] = tt.get_num_programs y {{.*}} : i32
// V1: %[[GZ:.*]] = tt.get_num_programs z {{.*}} : i32
// V1: %[[YZ:.*]] = arith.muli %[[GY]], %[[GZ]] {{.*}} : i32
// V1: %[[LOGICAL:.*]] = arith.muli %[[GX]], %[[YZ]] {{.*}} : i32
// V1: %[[HW:.*]] = tt.get_program_id x {{.*}} : i32
// V1: %[[PHYSICAL:.*]] = arith.constant {{.*}}64 : i32
// V1: %[[CHUNK:.*]] = arith.ceildivui %[[LOGICAL]], %[[PHYSICAL]] {{.*}} : i32
// V1: scf.for %[[IV:.*]] = {{.*}} to {{.*}} step {{.*}} {
// V1-NOT: tt.get_program_id
// V1: %[[PX:.*]] = arith.remui %[[IV]], %[[GX]] {{.*}} : i32
// V1: tt.store
// V1: } {ta.auto_blockify_v1.loop, ta.auto_blockify_v1.schedule}
// V1: tt.return
// V1-NOT: auto_blockify_size
// V1-NOT: tensor<2x8xf32>

// SUPERBLOCK-LABEL: tt.func public @v1_keeps_tile_shape(
// SUPERBLOCK-SAME: attributes {ta.auto_blockify_v1, ta.auto_blockify_v1.superblock_factor = 2 : i32}
// SUPERBLOCK: tt.get_program_id x {{.*}} : i32
// SUPERBLOCK: %[[TWO:.*]] = arith.constant {{.*}}2 : i32
// SUPERBLOCK: scf.for %[[IV:.*]] = {{.*}} to %[[UPPER:.*]] step %[[TWO]] : i32 {
// SUPERBLOCK-NOT: tt.get_program_id
// SUPERBLOCK: %[[TID_INDEX:.*]] = gpu.thread_id x {{.*}}
// SUPERBLOCK: %[[TID:.*]] = arith.index_cast %[[TID_INDEX]] {{.*}} : index to i32
// SUPERBLOCK: %[[WARP:.*]] = arith.divui %[[TID]], {{.*}} : i32
// SUPERBLOCK: %[[REM:.*]] = arith.remui %[[WARP]], %[[TWO]] {{.*}} : i32
// SUPERBLOCK: %[[LINEAR:.*]] = arith.addi %[[IV]], %[[REM]] {{.*}} : i32
// SUPERBLOCK: scf.if
// SUPERBLOCK: tt.store
// SUPERBLOCK: } {ta.auto_blockify_v1.loop, ta.auto_blockify_v1.schedule}
// SUPERBLOCK-NOT: auto_blockify_size
// SUPERBLOCK-NOT: tensor<2x8xf32>

// REFINE-LABEL: tt.func public @v1_keeps_tile_shape(
// REFINE-SAME: attributes {ta.auto_blockify_v1, ta.auto_blockify_v1.superblock_factor = 2 : i32}
// REFINE: %[[TWO:.*]] = arith.constant {{.*}}2 : i32
// REFINE: scf.for %[[IV:.*]] = {{.*}} to %[[UPPER:.*]] step %[[TWO]] : i32 {
// REFINE: %[[TID_INDEX:.*]] = gpu.thread_id x
// REFINE: %[[TID:.*]] = arith.index_cast %[[TID_INDEX]] {{.*}} : index to i32
// REFINE: %[[WARP:.*]] = arith.divui %[[TID]], {{.*}} {{.*}} : i32
// REFINE: %[[TASK:.*]] = arith.remui %[[WARP]], %[[TWO]] {{.*}} : i32
// REFINE: %[[LINEAR:.*]] = arith.addi %[[IV]], %[[TASK]] {{.*}} : i32
// REFINE: scf.if
// REFINE: %[[PX:.*]] = arith.remui %[[LINEAR]],
// REFINE: tt.store

// SUPERBLOCK32-LABEL: tt.func public @v1_keeps_tile_shape(
// SUPERBLOCK32-SAME: attributes {ta.auto_blockify_v1, ta.auto_blockify_v1.superblock_factor = 32 : i32}
// SUPERBLOCK32: %[[THIRTY_TWO:.*]] = arith.constant {{.*}}32 : i32
// SUPERBLOCK32: scf.for %[[IV:.*]] = {{.*}} to %[[UPPER:.*]] step %[[THIRTY_TWO]] : i32 {
// SUPERBLOCK32: %[[TASK:.*]] = arith.remui {{.*}}, %[[THIRTY_TWO]] {{.*}} : i32
// SUPERBLOCK32: %[[LINEAR:.*]] = arith.addi %[[IV]], %[[TASK]] {{.*}} : i32
// SUPERBLOCK32: scf.if
// SUPERBLOCK32: tt.store

// REFINE32-LABEL: tt.func public @v1_keeps_tile_shape(
// REFINE32-SAME: attributes {ta.auto_blockify_v1, ta.auto_blockify_v1.superblock_factor = 32 : i32}
// REFINE32: %[[THIRTY_TWO:.*]] = arith.constant {{.*}}32 : i32
// REFINE32: scf.for %[[IV:.*]] = {{.*}} to %[[UPPER:.*]] step %[[THIRTY_TWO]] : i32 {
// REFINE32: %[[TASK:.*]] = arith.remui {{.*}}, %[[THIRTY_TWO]] {{.*}} : i32
// REFINE32: %[[LINEAR:.*]] = arith.addi %[[IV]], %[[TASK]] {{.*}} : i32
// REFINE32: scf.if
// REFINE32: tt.store

module {
  tt.func public @v1_keeps_tile_shape(%arg0: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %c8 = arith.constant 8 : i32
    %base = arith.muli %pid, %c8 : i32
    %splat = tt.splat %base : i32 -> tensor<8xi32>
    %range = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %offset = arith.addi %splat, %range : tensor<8xi32>
    %ptr = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %out = tt.addptr %ptr, %offset : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<8xf32>
    tt.store %out, %zero : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
