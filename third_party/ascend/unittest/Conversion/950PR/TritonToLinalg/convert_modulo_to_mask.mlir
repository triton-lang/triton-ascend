// RUN: triton-opt %s --convert-modulo-to-mask --split-input-file \
// RUN: | FileCheck %s

// -----
// Strategy A: divisibility of bound >= tileSize. The remsi is simply removed.
// tt.divisibility = 16, tileSize = 16. No mask injection needed.
// CHECK-LABEL: func.func @modulo_removal_strategy_a
// CHECK-NOT: arith.remsi
// CHECK: tt.load
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @modulo_removal_strategy_a(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                            %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                            %M: i32 {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    // remsi: offs % M
    %M_splat = tt.splat %M : i32 -> tensor<16xi32>
    %rem = arith.remsi %offs, %M_splat : tensor<16xi32>
    // mask: offs < M (store mask guard)
    %mask = arith.cmpi slt, %offs, %M_splat : tensor<16xi32>
    // load uses rem as address
    %rem_expd = tt.expand_dims %rem {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %rem_bc = tt.broadcast %rem_expd : tensor<16x1xi32> -> tensor<16x16xi32>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %ptr = tt.addptr %base_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %mask_expd = tt.expand_dims %mask {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %mask_bc = tt.broadcast %mask_expd : tensor<16x1xi1> -> tensor<16x16xi1>
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %val = tt.load %ptr, %mask_bc, %zero : tensor<16x16x!tt.ptr<f32>>
    // store with mask
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %dst_ptr, %val, %mask_bc : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Strategy B: divisibility of bound < tileSize (e.g., divisibility=4, tile=16).
// The remsi is removed, the address index is clamped to [0, M-1] (arith.minsi),
// AND a boundary mask is injected into loads.
// CHECK-LABEL: func.func @modulo_removal_strategy_b
// CHECK-NOT: arith.remsi
// CHECK: arith.cmpi slt
// CHECK: arith.minsi
// CHECK: arith.andi
// CHECK: tt.load
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @modulo_removal_strategy_b(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                            %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                            %M: i32 {tt.divisibility = 4 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %M_splat = tt.splat %M : i32 -> tensor<16xi32>
    %rem = arith.remsi %offs, %M_splat : tensor<16xi32>
    %mask = arith.cmpi slt, %offs, %M_splat : tensor<16xi32>
    %rem_expd = tt.expand_dims %rem {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %rem_bc = tt.broadcast %rem_expd : tensor<16x1xi32> -> tensor<16x16xi32>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %ptr = tt.addptr %base_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %mask_expd = tt.expand_dims %mask {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %mask_bc = tt.broadcast %mask_expd : tensor<16x1xi1> -> tensor<16x16xi1>
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %val = tt.load %ptr, %mask_bc, %zero : tensor<16x16x!tt.ptr<f32>>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %dst_ptr, %val, %mask_bc : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Negative: remsi result is a 2-D tensor (not 1-D). The pass only handles 1-D.
// CHECK-LABEL: func.func @modulo_skip_2d_remsi
// CHECK: arith.remsi
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @modulo_skip_2d_remsi(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                       %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                       %M: i32 {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %offs_2d = tt.expand_dims %offs {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %offs_2d_bc = tt.broadcast %offs_2d : tensor<16x1xi32> -> tensor<16x16xi32>
    %M_splat_2d = tt.splat %M : i32 -> tensor<16x16xi32>
    // 2-D remsi — should NOT be matched
    %rem = arith.remsi %offs_2d_bc, %M_splat_2d : tensor<16x16xi32>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %ptr = tt.addptr %base_splat, %rem : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %val = tt.load %ptr : tensor<16x16x!tt.ptr<f32>>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_splat, %rem : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %dst_ptr, %val : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Negative: no store-mask guard on the un-modulo'd dividend. The pass cannot
// prove safety without a boundary mask → must keep the remsi.
// CHECK-LABEL: func.func @modulo_skip_no_store_mask
// CHECK: arith.remsi
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @modulo_skip_no_store_mask(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                            %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                            %M: i32 {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %M_splat = tt.splat %M : i32 -> tensor<16xi32>
    %rem = arith.remsi %offs, %M_splat : tensor<16xi32>
    // NO cmpi slt guard on offs — pass cannot prove safety
    %rem_expd = tt.expand_dims %rem {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %rem_bc = tt.broadcast %rem_expd : tensor<16x1xi32> -> tensor<16x16xi32>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %ptr = tt.addptr %base_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %val = tt.load %ptr : tensor<16x16x!tt.ptr<f32>>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %dst_ptr, %val : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Negative: remsi result is used outside address computation (feeds into a
// store VALUE, not just store ADDRESS). The pass must not remove it.
// CHECK-LABEL: func.func @modulo_skip_result_used_in_store_value
// CHECK: arith.remsi
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @modulo_skip_result_used_in_store_value(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                         %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                         %M: i32 {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %M_splat = tt.splat %M : i32 -> tensor<16xi32>
    %rem = arith.remsi %offs, %M_splat : tensor<16xi32>
    %mask = arith.cmpi slt, %offs, %M_splat : tensor<16xi32>
    // rem is used to store as value (sitofp -> store), not just address
    %rem_f32 = arith.sitofp %rem : tensor<16xi32> to tensor<16xf32>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_splat, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %rem_f32, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Negative: dividend does NOT contain a make_range. It's just a splat scalar.
// The pass requires a linear tile offset pattern.
// CHECK-LABEL: func.func @modulo_skip_no_make_range
// CHECK: arith.remsi
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @modulo_skip_no_make_range(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                            %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                            %M: i32 {tt.divisibility = 16 : i32},
                                            %idx: i32) {
    %M_splat = tt.splat %M : i32 -> tensor<16xi32>
    %idx_splat = tt.splat %idx : i32 -> tensor<16xi32>
    // dividend is just a scalar splat, no make_range
    %rem = arith.remsi %idx_splat, %M_splat : tensor<16xi32>
    %mask = arith.cmpi slt, %idx_splat, %M_splat : tensor<16xi32>
    %rem_expd = tt.expand_dims %rem {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %rem_bc = tt.broadcast %rem_expd : tensor<16x1xi32> -> tensor<16x16xi32>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %ptr = tt.addptr %base_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %mask_expd = tt.expand_dims %mask {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %mask_bc = tt.broadcast %mask_expd : tensor<16x1xi1> -> tensor<16x16xi1>
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %val = tt.load %ptr, %mask_bc, %zero : tensor<16x16x!tt.ptr<f32>>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %dst_ptr, %val, %mask_bc : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Negative: divisor is NOT a splat of a scalar (it's a make_range tensor).
// CHECK-LABEL: func.func @modulo_skip_non_splat_divisor
// CHECK: arith.remsi
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @modulo_skip_non_splat_divisor(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    // divisor is also a range, not a splat
    %div_range = tt.make_range {end = 17 : i32, start = 1 : i32} : tensor<16xi32>
    %rem = arith.remsi %offs, %div_range : tensor<16xi32>
    %rem_expd = tt.expand_dims %rem {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %rem_bc = tt.broadcast %rem_expd : tensor<16x1xi32> -> tensor<16x16xi32>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %ptr = tt.addptr %base_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %val = tt.load %ptr : tensor<16x16x!tt.ptr<f32>>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %dst_ptr, %val : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Negative: no expand_dims on remsi result (axis cannot be determined).
// The pass requires knowing the expansion axis for mask shaping.
// CHECK-LABEL: func.func @modulo_skip_no_expand_dims
// CHECK: arith.remsi
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @modulo_skip_no_expand_dims(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                             %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                             %M: i32 {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %M_splat = tt.splat %M : i32 -> tensor<16xi32>
    %rem = arith.remsi %offs, %M_splat : tensor<16xi32>
    %mask = arith.cmpi slt, %offs, %M_splat : tensor<16xi32>
    // rem is used directly in addptr without expand_dims
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %ptr = tt.addptr %base_splat, %rem : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %val = tt.load %ptr, %mask : tensor<16x!tt.ptr<f32>>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_splat, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptr, %val, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Negative: mask guard uses a DIFFERENT bound scalar than the divisor.
// Pass requires the cmpi's RHS to be the same scalar as the remsi divisor.
// CHECK-LABEL: func.func @modulo_skip_mismatched_bound
// CHECK: arith.remsi
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @modulo_skip_mismatched_bound(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                               %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                               %M: i32 {tt.divisibility = 16 : i32},
                                               %N: i32 {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %M_splat = tt.splat %M : i32 -> tensor<16xi32>
    %rem = arith.remsi %offs, %M_splat : tensor<16xi32>
    // mask bound is N, not M — mismatch
    %N_splat = tt.splat %N : i32 -> tensor<16xi32>
    %mask = arith.cmpi slt, %offs, %N_splat : tensor<16xi32>
    %rem_expd = tt.expand_dims %rem {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %rem_bc = tt.broadcast %rem_expd : tensor<16x1xi32> -> tensor<16x16xi32>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %ptr = tt.addptr %base_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %mask_expd = tt.expand_dims %mask {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %mask_bc = tt.broadcast %mask_expd : tensor<16x1xi1> -> tensor<16x16xi1>
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %val = tt.load %ptr, %mask_bc, %zero : tensor<16x16x!tt.ptr<f32>>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %dst_ptr, %val, %mask_bc : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Negative: result type is i64, not i32. The pass checks elementType.isInteger(32).
// CHECK-LABEL: func.func @modulo_skip_i64_type
// CHECK: arith.remsi
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @modulo_skip_i64_type(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                       %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                       %M: i64) {
    %pid = tt.get_program_id x : i32
    %pid64 = arith.extsi %pid : i32 to i64
    %c16 = arith.constant 16 : i64
    %blk = arith.muli %pid64, %c16 : i64
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %range64 = arith.extsi %range : tensor<16xi32> to tensor<16xi64>
    %blk_splat = tt.splat %blk : i64 -> tensor<16xi64>
    %offs = arith.addi %blk_splat, %range64 : tensor<16xi64>
    %M_splat = tt.splat %M : i64 -> tensor<16xi64>
    %rem = arith.remsi %offs, %M_splat : tensor<16xi64>
    %rem_expd = tt.expand_dims %rem {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>
    %rem_bc = tt.broadcast %rem_expd : tensor<16x1xi64> -> tensor<16x16xi64>
    %rem_bc_i32 = arith.trunci %rem_bc : tensor<16x16xi64> to tensor<16x16xi32>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %ptr = tt.addptr %base_splat, %rem_bc_i32 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %val = tt.load %ptr : tensor<16x16x!tt.ptr<f32>>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_splat, %rem_bc_i32 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %dst_ptr, %val : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Strategy B with unmasked load: the pass clamps the address index (arith.minsi)
// and injects a fresh boundary mask and zero "other" value into the previously
// unmasked load.
// CHECK-LABEL: func.func @modulo_inject_mask_into_unmasked_load
// CHECK-NOT: arith.remsi
// CHECK: arith.cmpi slt
// CHECK: arith.minsi
// CHECK: tt.load
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @modulo_inject_mask_into_unmasked_load(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                        %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                        %M: i32 {tt.divisibility = 4 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %M_splat = tt.splat %M : i32 -> tensor<16xi32>
    %rem = arith.remsi %offs, %M_splat : tensor<16xi32>
    %store_mask = arith.cmpi slt, %offs, %M_splat : tensor<16xi32>
    %rem_expd = tt.expand_dims %rem {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %rem_bc = tt.broadcast %rem_expd : tensor<16x1xi32> -> tensor<16x16xi32>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %ptr = tt.addptr %base_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    // Unmasked load — the pass injects a fresh boundary mask with zero other.
    %val = tt.load %ptr : tensor<16x16x!tt.ptr<f32>>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %store_mask_expd = tt.expand_dims %store_mask {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %store_mask_bc = tt.broadcast %store_mask_expd : tensor<16x1xi1> -> tensor<16x16xi1>
    tt.store %dst_ptr, %val, %store_mask_bc : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Decomposed modulo (divsi+muli+subi): Strategy A — divisibility >= tileSize.
// All three ops (divsi, muli, subi) should be removed.
// CHECK-LABEL: func.func @decomposed_modulo_strategy_a
// CHECK-NOT: arith.divsi
// CHECK-NOT: arith.subi
// CHECK: tt.load
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @decomposed_modulo_strategy_a(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                               %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                               %M: i32 {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %M_splat = tt.splat %M : i32 -> tensor<16xi32>
    // Decomposed modulo: offs - (offs / M) * M
    %div = arith.divsi %offs, %M_splat : tensor<16xi32>
    %mul = arith.muli %div, %M_splat : tensor<16xi32>
    %rem = arith.subi %offs, %mul : tensor<16xi32>
    %mask = arith.cmpi slt, %offs, %M_splat : tensor<16xi32>
    %rem_expd = tt.expand_dims %rem {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %rem_bc = tt.broadcast %rem_expd : tensor<16x1xi32> -> tensor<16x16xi32>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %ptr = tt.addptr %base_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %mask_expd = tt.expand_dims %mask {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %mask_bc = tt.broadcast %mask_expd : tensor<16x1xi1> -> tensor<16x16xi1>
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %val = tt.load %ptr, %mask_bc, %zero : tensor<16x16x!tt.ptr<f32>>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %dst_ptr, %val, %mask_bc : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Decomposed modulo: Strategy B with unmasked load (fused_moe_kernel pattern).
// The pass removes the decomposed modulo (divsi/muli/subi), clamps the address
// index to [0, N-1] (arith.minsi), and injects a boundary mask and zero other
// into the unmasked load. Store mask guard is via expand_dims (2D path).
// Note: divsi uniquely marks the decomposed modulo; the clamp introduces its
// own arith.subi (N-1), so we key the removal check off divsi, not subi.
// CHECK-LABEL: func.func @decomposed_modulo_strategy_b_unmasked
// CHECK-NOT: arith.divsi
// CHECK: arith.cmpi slt
// CHECK: arith.minsi
// CHECK: tt.load
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @decomposed_modulo_strategy_b_unmasked(%base: !tt.ptr<i8> {tt.divisibility = 16 : i32},
                                                        %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                        %N: i32 {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %cst_stride = arith.constant dense<64> : tensor<1x64xi32>
    %blk = arith.muli %pid, %c64 : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<64xi32>
    %offs = arith.addi %blk_splat, %range : tensor<64xi32>
    %N_splat = tt.splat %N : i32 -> tensor<64xi32>
    // Decomposed modulo: offs - (offs / N) * N
    %div = arith.divsi %offs, %N_splat : tensor<64xi32>
    %mul = arith.muli %div, %N_splat : tensor<64xi32>
    %rem = arith.subi %offs, %mul : tensor<64xi32>
    // Build 2D address for load (axis=0 expansion matches fused_moe pattern)
    %row_range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %col_stride = arith.muli %rem_expd, %cst_stride : tensor<1x64xi32>
    %row_bc = tt.broadcast %row_expd : tensor<32x1xi32> -> tensor<32x64xi32>
    %col_bc = tt.broadcast %col_stride : tensor<1x64xi32> -> tensor<32x64xi32>
    %offset_2d = arith.addi %row_bc, %col_bc : tensor<32x64xi32>
    %base_splat = tt.splat %base : !tt.ptr<i8> -> tensor<32x64x!tt.ptr<i8>>
    %ptr = tt.addptr %base_splat, %offset_2d : tensor<32x64x!tt.ptr<i8>>, tensor<32x64xi32>
    // Unmasked load — pass should inject boundary mask with zero other
    %val = tt.load %ptr : tensor<32x64x!tt.ptr<i8>>
    // Store mask guard via expand_dims path
    %offs_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %N_splat_2d = tt.splat %N : i32 -> tensor<1x64xi32>
    %col_mask = arith.cmpi slt, %offs_expd, %N_splat_2d : tensor<1x64xi32>
    %col_mask_bc = tt.broadcast %col_mask : tensor<1x64xi1> -> tensor<32x64xi1>
    %val_f16 = arith.sitofp %val : tensor<32x64xi8> to tensor<32x64xf16>
    %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>>
    %dst_ptr = tt.addptr %dst_splat, %offset_2d : tensor<32x64x!tt.ptr<f16>>, tensor<32x64xi32>
    tt.store %dst_ptr, %val_f16, %col_mask_bc : tensor<32x64x!tt.ptr<f16>>
    tt.return
  }
}

// -----
// Negative safety case: remsi flows through scf.for iter_arg and is used as a
// store value (non-address use). The pass must bail out and keep remsi.
// CHECK-LABEL: func.func @modulo_skip_remsi_used_as_value_in_loop
// CHECK: arith.remsi
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @modulo_skip_remsi_used_as_value_in_loop(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                          %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                          %M: i32 {tt.divisibility = 4 : i32}) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %blk = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
    %offs = arith.addi %blk_splat, %range : tensor<16xi32>
    %M_splat = tt.splat %M : i32 -> tensor<16xi32>
    %rem = arith.remsi %offs, %M_splat : tensor<16xi32>
    %store_mask = arith.cmpi slt, %offs, %M_splat : tensor<16xi32>
    %rem_expd = tt.expand_dims %rem {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %rem_bc = tt.broadcast %rem_expd : tensor<16x1xi32> -> tensor<16x16xi32>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %ptr = tt.addptr %base_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %store_mask_expd = tt.expand_dims %store_mask {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %store_mask_bc = tt.broadcast %store_mask_expd : tensor<16x1xi1> -> tensor<16x16xi1>
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %val = tt.load %ptr, %store_mask_bc, %zero : tensor<16x16x!tt.ptr<f32>>
    // Non-address use through loop-carried iter_arg.
    %loop_res = scf.for %iv = %c0 to %c1 step %c1 iter_args(%acc = %rem) -> (tensor<16xi32>) {
      %acc_next = arith.addi %acc, %M_splat : tensor<16xi32>
      scf.yield %acc_next : tensor<16xi32>
    }
    %loop_f32 = arith.sitofp %loop_res : tensor<16xi32> to tensor<16xf32>
    %loop_f32_expd = tt.expand_dims %loop_f32 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    %loop_f32_bc = tt.broadcast %loop_f32_expd : tensor<16x1xf32> -> tensor<16x16xf32>
    %out = arith.addf %val, %loop_f32_bc : tensor<16x16xf32>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_splat, %rem_bc : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %dst_ptr, %out, %store_mask_bc : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
