// RUN: triton-opt %s --verify-each '-graph-optimize=rule-mask=262144 ub-capacity-bytes=253952 mapping-ub-capacity-bytes=253952 ub-safety-percent=100 reserved-ub-bytes=0' | FileCheck %s --check-prefix=RETILE
// RUN: sed 's/"tt.reduce"(%dy)/"tt.reduce"(%dy_with_sum)/' %s | triton-opt --verify-each '-graph-optimize=rule-mask=262144 ub-capacity-bytes=253952 mapping-ub-capacity-bytes=253952 ub-safety-percent=100 reserved-ub-bytes=0' | FileCheck %s --check-prefix=LOCKED
// RUN: triton-opt %s --verify-each -graph-optimize | FileCheck %s --check-prefix=LOCKED

// RETILE: hacc.program_grid_transforms
// RETILE-SAME: kind = "cap"
// RETILE: tt.make_range {end = 4096
// RETILE-NOT: tt.atomic
// LOCKED-NOT: hacc.program_grid_transforms
// LOCKED: tt.make_range {end = 8192
// LOCKED: tt.atomic_cas
module {
  tt.func public @grouped_update(%dx: !tt.ptr<f32>, %x: !tt.ptr<f32>, %dy_in: !tt.ptr<f32>,
                                %dw: !tt.ptr<f32>, %db: !tt.ptr<f32>, %locks: !tt.ptr<i32>,
                                %n: i32, %stride: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %groups = arith.constant 96 : i32
    %true = arith.constant true
    %zero_vec = arith.constant dense<0.0> : tensor<8192xf32>
    %row = tt.get_program_id x : i32
    %group = arith.remsi %row, %groups : i32
    %lock = tt.addptr %locks, %group : !tt.ptr<i32>, i32
    %counter = tt.addptr %lock, %groups : !tt.ptr<i32>, i32
    %col = tt.make_range {start = 0 : i32, end = 8192 : i32} : tensor<8192xi32>
    %n_vec = tt.splat %n : i32 -> tensor<8192xi32>
    %mask = arith.cmpi slt, %col, %n_vec : tensor<8192xi32>
    %row_start = arith.muli %row, %stride : i32
    %row_start_vec = tt.splat %row_start : i32 -> tensor<8192xi32>
    %row_offset = arith.addi %row_start_vec, %col : tensor<8192xi32>
    %x_base = tt.splat %x : !tt.ptr<f32> -> tensor<8192x!tt.ptr<f32>>
    %x_ptr = tt.addptr %x_base, %row_offset : tensor<8192x!tt.ptr<f32>>, tensor<8192xi32>
    %x_val = tt.load %x_ptr, %mask, %zero_vec : tensor<8192x!tt.ptr<f32>>
    %dy_base = tt.splat %dy_in : !tt.ptr<f32> -> tensor<8192x!tt.ptr<f32>>
    %dy_ptr = tt.addptr %dy_base, %row_offset : tensor<8192x!tt.ptr<f32>>, tensor<8192xi32>
    %dy = tt.load %dy_ptr, %mask, %zero_vec : tensor<8192x!tt.ptr<f32>>
    %sum_x = "tt.reduce"(%x_val) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<8192xf32>) -> f32
    %sum_x_vec = tt.splat %sum_x : f32 -> tensor<8192xf32>
    %dy_with_sum = arith.addf %dy, %sum_x_vec : tensor<8192xf32>
    %sum_dy = "tt.reduce"(%dy) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<8192xf32>) -> f32
    %total = arith.addf %sum_x, %sum_dy : f32
    %value = tt.splat %total : f32 -> tensor<8192xf32>
    %dx_base = tt.splat %dx : !tt.ptr<f32> -> tensor<8192x!tt.ptr<f32>>
    %dx_ptr = tt.addptr %dx_base, %row_offset : tensor<8192x!tt.ptr<f32>>, tensor<8192xi32>
    tt.store %dx_ptr, %value, %mask : tensor<8192x!tt.ptr<f32>>
    %group_start = arith.muli %group, %n : i32
    %group_start_vec = tt.splat %group_start : i32 -> tensor<8192xi32>
    %group_offset = arith.addi %group_start_vec, %col : tensor<8192xi32>
    %dw_base = tt.splat %dw : !tt.ptr<f32> -> tensor<8192x!tt.ptr<f32>>
    %db_base = tt.splat %db : !tt.ptr<f32> -> tensor<8192x!tt.ptr<f32>>
    %dw_ptr = tt.addptr %dw_base, %group_offset : tensor<8192x!tt.ptr<f32>>, tensor<8192xi32>
    %db_ptr = tt.addptr %db_base, %group_offset : tensor<8192x!tt.ptr<f32>>, tensor<8192xi32>
    scf.while : () -> () {
      %old_lock = tt.atomic_cas acq_rel, gpu, %lock, %zero, %one : (!tt.ptr<i32>, i32, i32) -> i32
      %busy = arith.cmpi eq, %old_lock, %one : i32
      scf.condition(%busy)
    } do {
      scf.yield
    }
    %count = tt.load %counter : !tt.ptr<i32>
    %first = arith.cmpi eq, %count, %zero : i32
    %partial:2 = scf.if %first -> (tensor<8192xf32>, tensor<8192xf32>) {
      %old_count = tt.atomic_rmw exch, acq_rel, gpu, %counter, %one, %true : (!tt.ptr<i32>, i32, i1) -> i32
      scf.yield %dy_with_sum, %dy : tensor<8192xf32>, tensor<8192xf32>
    } else {
      %prev_dw = tt.load %dw_ptr, %mask, %zero_vec : tensor<8192x!tt.ptr<f32>>
      %prev_db = tt.load %db_ptr, %mask, %zero_vec : tensor<8192x!tt.ptr<f32>>
      %next_dw = arith.addf %dy_with_sum, %prev_dw : tensor<8192xf32>
      %next_db = arith.addf %dy, %prev_db : tensor<8192xf32>
      scf.yield %next_dw, %next_db : tensor<8192xf32>, tensor<8192xf32>
    }
    tt.store %dw_ptr, %partial#0, %mask : tensor<8192x!tt.ptr<f32>>
    tt.store %db_ptr, %partial#1, %mask : tensor<8192x!tt.ptr<f32>>
    %unlock = tt.atomic_rmw exch, acq_rel, gpu, %lock, %zero, %true : (!tt.ptr<i32>, i32, i1) -> i32
    tt.return
  }
}
