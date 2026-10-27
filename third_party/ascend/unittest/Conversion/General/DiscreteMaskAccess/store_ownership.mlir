// RUN: triton-opt --discrete-mask-access-conversion=compile-mode=simd --split-input-file %s | FileCheck %s

// A matching pid*64+range appears in the value, while the address uses stride
// 32. The original selected lanes are disjoint but the RMW windows overlap.
// CHECK-LABEL: tt.func @overlapping_windows
// CHECK-NOT: tt.get_num_programs
// CHECK: hivm.hir.sync_block_lock
// CHECK: %[[OLD:.*]] = tt.load
// CHECK: arith.select {{.*}}, {{.*}}, %[[OLD]]
// CHECK: tt.store
// CHECK: hivm.hir.sync_block_unlock
tt.func @overlapping_windows(%A: !tt.ptr<f32>) {
  %pid = tt.get_program_id x : i32
  %p64 = arith.extsi %pid : i32 to i64
  %c64 = arith.constant 64 : i64
  %c32 = arith.constant 32 : i64
  %c2 = arith.constant 2 : i32
  %two = arith.constant dense<2> : tensor<64xi32>
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %l64 = arith.extsi %lane : tensor<64xi32> to tensor<64xi64>
  %data_base = arith.muli %p64, %c64 : i64
  %address_base = arith.muli %p64, %c32 : i64
  %d = tt.splat %data_base : i64 -> tensor<64xi64>
  %a = tt.splat %address_base : i64 -> tensor<64xi64>
  %value = arith.addi %d, %l64 : tensor<64xi64>
  %offset = arith.addi %a, %l64 : tensor<64xi64>
  %float = arith.sitofp %value : tensor<64xi64> to tensor<64xf32>
  %lane_parity = arith.remsi %lane, %two : tensor<64xi32>
  %pid_parity = arith.remsi %pid, %c2 : i32
  %parity = tt.splat %pid_parity : i32 -> tensor<64xi32>
  %mask = arith.cmpi eq, %lane_parity, %parity : tensor<64xi32>
  %base = tt.splat %A : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %offset : tensor<64x!tt.ptr<f32>>, tensor<64xi64>
  tt.store %ptr, %float, %mask : tensor<64x!tt.ptr<f32>>
  tt.return
}

// -----

// Fixed-length GLA: ((b*T + t*64 + i*16 + row)*2 + h)*64
//                    + i*16 + col, with (b,h) = divmod(pid_z + offset, 2).
// CHECK-LABEL: tt.func @mixed_radix_tile
// CHECK: tt.get_num_programs x
// CHECK: tt.get_num_programs y
// CHECK: tt.get_num_programs z
// CHECK: scf.if %[[GLALOCK:.*]] {
// CHECK-NEXT: hivm.hir.sync_block_lock
// CHECK: tt.load
// CHECK: tt.store
// CHECK: scf.if %[[GLALOCK]] {
// CHECK-NEXT: hivm.hir.sync_block_unlock
tt.func @mixed_radix_tile(%A: !tt.ptr<f32>, %T: i64, %offset: i64) {
  %px = tt.get_program_id x : i32
  %py = tt.get_program_id y : i32
  %pz = tt.get_program_id z : i32
  %t = arith.extsi %px : i32 to i64
  %i = arith.extsi %py : i32 to i64
  %z = arith.extsi %pz : i32 to i64
  %bh = arith.addi %z, %offset : i64
  %c2 = arith.constant 2 : i64
  %c16 = arith.constant 16 : i64
  %c64 = arith.constant 64 : i64
  %c128 = arith.constant 128 : i64
  %b = arith.divsi %bh, %c2 : i64
  %h = arith.remsi %bh, %c2 : i64
  %bos = arith.muli %b, %T : i64
  %tr = arith.muli %t, %c64 : i64
  %ic = arith.muli %i, %c16 : i64
  %br = arith.addi %bos, %tr : i64
  %bir = arith.addi %br, %ic : i64
  %rows = arith.muli %bir, %c128 : i64
  %head = arith.muli %h, %c64 : i64
  %rh = arith.addi %rows, %head : i64
  %base = arith.addi %rh, %ic : i64
  %bs = tt.splat %base : i64 -> tensor<16x16xi64>
  %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %ri = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %ci = tt.expand_dims %r {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rm = tt.broadcast %ri : tensor<16x1xi32> -> tensor<16x16xi32>
  %cm = tt.broadcast %ci : tensor<1x16xi32> -> tensor<16x16xi32>
  %r64 = arith.extsi %rm : tensor<16x16xi32> to tensor<16x16xi64>
  %c64t = arith.extsi %cm : tensor<16x16xi32> to tensor<16x16xi64>
  %stride = arith.constant dense<128> : tensor<16x16xi64>
  %ro = arith.muli %r64, %stride : tensor<16x16xi64>
  %rc = arith.addi %ro, %c64t : tensor<16x16xi64>
  %o = arith.addi %bs, %rc : tensor<16x16xi64>
  %a = tt.splat %A : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
  %p = tt.addptr %a, %o : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi64>
  %mask = arith.cmpi slt, %rm, %cm : tensor<16x16xi32>
  %v = arith.constant dense<0.0> : tensor<16x16xf32>
  tt.store %p, %v, %mask : tensor<16x16x!tt.ptr<f32>>
  tt.return
}

// -----

// An index read from memory is not an affine program coordinate.
// CHECK-LABEL: tt.func @indirect_address
// CHECK-NOT: tt.get_num_programs
// CHECK: hivm.hir.sync_block_lock
// CHECK: tt.load
// CHECK: arith.select
// CHECK: tt.store
// CHECK: hivm.hir.sync_block_unlock
tt.func @indirect_address(%A: !tt.ptr<f32>, %indices: !tt.ptr<i32>) {
  %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %i = tt.splat %indices : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %ip = tt.addptr %i, %r : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  %index = tt.load %ip : tensor<16x!tt.ptr<i32>>
  %a = tt.splat %A : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %p = tt.addptr %a, %index : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %v = arith.constant dense<2.0> : tensor<16xf32>
  %c8 = arith.constant dense<8> : tensor<16xi32>
  %mask = arith.cmpi sgt, %r, %c8 : tensor<16xi32>
  tt.store %p, %v, %mask : tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----

// Each store is self-disjoint, but the second store's RMW crosses into the
// next program's tile. They must not be classified independently.
// CHECK-LABEL: tt.func @cross_store_overlap
// CHECK-NOT: tt.get_num_programs
// CHECK: hivm.hir.sync_block_lock
// CHECK: hivm.hir.sync_block_unlock
// CHECK: hivm.hir.sync_block_lock
// CHECK: hivm.hir.sync_block_unlock
tt.func @cross_store_overlap(%A: !tt.ptr<f32>) {
  %pid = tt.get_program_id x : i32
  %pid64 = arith.extsi %pid : i32 to i64
  %c16 = arith.constant 16 : i64
  %base = arith.muli %pid64, %c16 : i64
  %next = arith.addi %base, %c16 : i64
  %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %r64 = arith.extsi %r : tensor<16xi32> to tensor<16xi64>
  %b = tt.splat %base : i64 -> tensor<16xi64>
  %n = tt.splat %next : i64 -> tensor<16xi64>
  %o = arith.addi %b, %r64 : tensor<16xi64>
  %on = arith.addi %n, %r64 : tensor<16xi64>
  %a = tt.splat %A : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %p = tt.addptr %a, %o : tensor<16x!tt.ptr<f32>>, tensor<16xi64>
  %pn = tt.addptr %a, %on : tensor<16x!tt.ptr<f32>>, tensor<16xi64>
  %v = arith.constant dense<2.0> : tensor<16xf32>
  %two = arith.constant dense<2> : tensor<16xi32>
  %zero = arith.constant dense<0> : tensor<16xi32>
  %remainder = arith.remsi %r, %two : tensor<16xi32>
  %even = arith.cmpi eq, %remainder, %zero : tensor<16xi32>
  %odd = arith.cmpi ne, %remainder, %zero : tensor<16xi32>
  tt.store %p, %v, %even : tensor<16x!tt.ptr<f32>>
  tt.store %pn, %v, %odd : tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----

// Tensor division can follow wrapping i32 arithmetic. PtrAnalysis's
// mathematical strides alone do not prove the original signed computation.
// CHECK-LABEL: tt.func @tensor_division
// CHECK-NOT: tt.get_num_programs
// CHECK: hivm.hir.sync_block_lock
// CHECK: hivm.hir.sync_block_unlock
tt.func @tensor_division(%A: !tt.ptr<f32>, %T: i64, %offset: i64) {
  %px = tt.get_program_id x : i32
  %py = tt.get_program_id y : i32
  %pz = tt.get_program_id z : i32
  %t = arith.extsi %px : i32 to i64
  %i = arith.extsi %py : i32 to i64
  %z = arith.extsi %pz : i32 to i64
  %bh = arith.addi %z, %offset : i64
  %c2 = arith.constant 2 : i64
  %c16 = arith.constant 16 : i64
  %c64 = arith.constant 64 : i64
  %c128 = arith.constant 128 : i64
  %b = arith.divsi %bh, %c2 : i64
  %h = arith.remsi %bh, %c2 : i64
  %bos = arith.muli %b, %T : i64
  %tr = arith.muli %t, %c64 : i64
  %ic = arith.muli %i, %c16 : i64
  %br = arith.addi %bos, %tr : i64
  %bir = arith.addi %br, %ic : i64
  %rows = arith.muli %bir, %c128 : i64
  %head = arith.muli %h, %c64 : i64
  %rh = arith.addi %rows, %head : i64
  %base = arith.addi %rh, %ic : i64
  %bs = tt.splat %base : i64 -> tensor<16x16xi64>
  %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %ri = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %ci = tt.expand_dims %r {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rm = tt.broadcast %ri : tensor<16x1xi32> -> tensor<16x16xi32>
  %cm = tt.broadcast %ci : tensor<1x16xi32> -> tensor<16x16xi32>
  %large = arith.constant dense<1073741824> : tensor<16x16xi32>
  %wrapped = arith.muli %rm, %large : tensor<16x16xi32>
  %wide = arith.extsi %wrapped : tensor<16x16xi32> to tensor<16x16xi64>
  %divisor = arith.constant dense<1073741824> : tensor<16x16xi64>
  %r64 = arith.divsi %wide, %divisor : tensor<16x16xi64>
  %c64t = arith.extsi %cm : tensor<16x16xi32> to tensor<16x16xi64>
  %stride = arith.constant dense<128> : tensor<16x16xi64>
  %ro = arith.muli %r64, %stride : tensor<16x16xi64>
  %rc = arith.addi %ro, %c64t : tensor<16x16xi64>
  %o = arith.addi %bs, %rc : tensor<16x16xi64>
  %a = tt.splat %A : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
  %p = tt.addptr %a, %o : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi64>
  %mask = arith.cmpi slt, %rm, %cm : tensor<16x16xi32>
  %v = arith.constant dense<0.0> : tensor<16x16xf32>
  tt.store %p, %v, %mask : tensor<16x16x!tt.ptr<f32>>
  tt.return
}


// -----

// Constant coefficients may wrap to an apparently valid stride. An i32
// multiplication before extension makes that folded stride unsound.
// CHECK-LABEL: tt.func @overflowing_coefficients
// CHECK-NOT: tt.get_num_programs
// CHECK: hivm.hir.sync_block_lock
// CHECK: hivm.hir.sync_block_unlock
tt.func @overflowing_coefficients(%A: !tt.ptr<f32>, %T: i64, %offset: i64) {
  %px = tt.get_program_id x : i32
  %py = tt.get_program_id y : i32
  %pz = tt.get_program_id z : i32
  %t = arith.extsi %px : i32 to i64
  %i = arith.extsi %py : i32 to i64
  %z = arith.extsi %pz : i32 to i64
  %bh = arith.addi %z, %offset : i64
  %c2 = arith.constant 2 : i64
  %c16 = arith.constant 16 : i64
  %c64 = arith.constant 64 : i64
  %c128 = arith.constant 128 : i64
  %b = arith.divsi %bh, %c2 : i64
  %h = arith.remsi %bh, %c2 : i64
  %bos = arith.muli %b, %T : i64
  %tr = arith.muli %t, %c64 : i64
  %ic = arith.muli %i, %c16 : i64
  %br = arith.addi %bos, %tr : i64
  %bir = arith.addi %br, %ic : i64
  %rows = arith.muli %bir, %c128 : i64
  %head = arith.muli %h, %c64 : i64
  %rh = arith.addi %rows, %head : i64
  %base = arith.addi %rh, %ic : i64
  %bs = tt.splat %base : i64 -> tensor<16x16xi64>
  %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %ri = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %ci = tt.expand_dims %r {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rm = tt.broadcast %ri : tensor<16x1xi32> -> tensor<16x16xi32>
  %cm = tt.broadcast %ci : tensor<1x16xi32> -> tensor<16x16xi32>
  %large32 = arith.constant dense<299369019> : tensor<16x16xi32>
  %wrapped32 = arith.muli %rm, %large32 : tensor<16x16xi32>
  %r64 = arith.extsi %wrapped32 : tensor<16x16xi32> to tensor<16x16xi64>
  %c64t = arith.extsi %cm : tensor<16x16xi32> to tensor<16x16xi64>
  %three = arith.constant dense<3> : tensor<16x16xi64>
  %factor = arith.constant dense<160465489> : tensor<16x16xi64>
  %stride = arith.constant dense<128> : tensor<16x16xi64>
  %scaled = arith.muli %r64, %three : tensor<16x16xi64>
  %scaled2 = arith.muli %scaled, %stride : tensor<16x16xi64>
  %ro = arith.muli %scaled2, %factor : tensor<16x16xi64>
  %rc = arith.addi %ro, %c64t : tensor<16x16xi64>
  %o = arith.addi %bs, %rc : tensor<16x16xi64>
  %a = tt.splat %A : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
  %p = tt.addptr %a, %o : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi64>
  %mask = arith.cmpi slt, %rm, %cm : tensor<16x16xi32>
  %v = arith.constant dense<0.0> : tensor<16x16xf32>
  tt.store %p, %v, %mask : tensor<16x16x!tt.ptr<f32>>
  tt.return
}
