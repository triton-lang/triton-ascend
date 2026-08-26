// RUN: triton-opt --triton-to-linalg="named-ops=True" %s | FileCheck %s

// The six launch-grid args appended by addProgramInfo carry the TA<->NPUIR
// contract as hacc.arg_type annotations: program_num x/y/z, then
// program_id x/y/z.
// CHECK-LABEL: func.func @kernel_grid_arg_attrs(
// CHECK-SAME: %{{.*}}: i32 {hacc.arg_type = #hacc.arg_type<program_num_x>}
// CHECK-SAME: %{{.*}}: i32 {hacc.arg_type = #hacc.arg_type<program_num_y>}
// CHECK-SAME: %{{.*}}: i32 {hacc.arg_type = #hacc.arg_type<program_num_z>}
// CHECK-SAME: %{{.*}}: i32 {hacc.arg_type = #hacc.arg_type<program_id_x>}
// CHECK-SAME: %{{.*}}: i32 {hacc.arg_type = #hacc.arg_type<program_id_y>}
// CHECK-SAME: %{{.*}}: i32 {hacc.arg_type = #hacc.arg_type<program_id_z>}

tt.func public @kernel_grid_arg_attrs(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
  tt.return
}
