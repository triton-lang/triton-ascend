// REQUIRES: proton
// RUN: triton-opt %s --assign-ascend-proton-scope-ids --lower-ascend-proton-to-hivm='data-segment-bytes=16' | FileCheck %s

// No GPU launch attributes are required by the Ascend path.
module {
  func.func @profiled(%input: memref<?xf32, #hivm.address_space<gm>>,
                      %gx: i32, %gy: i32, %gz: i32,
                      %px: i32, %py: i32, %pz: i32)
      attributes {global_kernel = "local", hivm.func_core_type = #hivm.func_core_type<AIV>} {
    proton.record start "outer"
    proton.record start "inner"
    proton.record end "inner"
    proton.record end "outer"
    return
  }
}

// CHECK-LABEL: func.func @profiled(
// CHECK-SAME: %{{.*}}: memref<?xf32, #hivm.address_space<gm>>,
// CHECK-SAME: %[[BUFFER:.*]]: memref<?xi32, #hivm.address_space<gm>>,
// CHECK-SAME: profile_num_units = 1 : i32
// CHECK-SAME: profile_sample_every_n = 1 : i32
// CHECK-SAME: profile_scratch_align = 64 : i32
// CHECK-SAME: profile_scratch_size = 64 : i32
// CHECK-SAME: profile_sections_per_block = 1 : i32
// CHECK-COUNT-4: "__builtin_proton_circular_store" ins(%[[BUFFER]],
// CHECK-NOT: proton.record
