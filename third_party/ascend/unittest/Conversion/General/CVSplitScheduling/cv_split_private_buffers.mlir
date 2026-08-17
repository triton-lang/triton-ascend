// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=OFF
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=OFFTYPE
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4 private-buffer-ub-budget-bytes=4096" 2>/dev/null | FileCheck %s --check-prefix=UNION
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4 private-buffer-ub-budget-bytes=4096" 2>/dev/null | FileCheck %s --check-prefix=UNIONVIEW
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4 private-buffer-ub-budget-bytes=4096 promote-private-buffer-pools=true" 2>/dev/null | FileCheck %s --check-prefix=PROMOTE
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4 private-buffer-ub-budget-bytes=4096 promote-private-buffer-pools=true" 2>/dev/null | FileCheck %s --check-prefix=PROMOTETYPE
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=8 private-buffer-ub-budget-bytes=1048576 promote-private-buffer-pools=true" 2>/dev/null | FileCheck %s --check-prefix=FLAGCAP
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=8 private-buffer-ub-budget-bytes=1048576 promote-private-buffer-pools=true" 2>/dev/null | FileCheck %s --check-prefix=FLAGCAPTYPE
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" 2>/dev/null | FileCheck %s --check-prefix=NOREUSE
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" 2>/dev/null | FileCheck %s --check-prefix=NOREUSETYPE
//
// A transfer slot owned by one unrolled lane is never reused, so it has no
// write-after-read pair to order and the producing core cannot overwrite what
// the consumer is still reading.
//
// The cheapest way to buy that is to merge the two CUBE->VECTOR roles onto one
// union slot per lane: a lane's score tile is dead by the time its product
// tile is written, so one buffer serves both.  The slot is sized for the larger
// role and the smaller one takes a contiguous view of its front, which is what
// lets the two roles differ in size.  That is funded by a UB budget.
//
// Giving an individual *pool* one buffer per lane is a separate, opt-in switch.
// It is off by default because the reuse it removes is already ordered by a
// flag the schedule needs for its own data, so it costs buffers and flags
// without removing a stall.
//
// The candidate has three phases: QK (UB, 16x32xf32 = 2048 bytes a slot),
// P (L1/cbuf, free against a UB budget) and PV (UB, 16x64xf32 = 4096 bytes).
//
// Buffer counts and flag counts are checked on separate prefixes from the
// buffer *types*: the allocations of the three pools interleave in the output,
// so a COUNT/NOT pair and a type-matching DAG group cannot share a prefix.

// Defaults must leave the schedule exactly as it was before either option
// existed -- three phases over two slots each, six flags.  Merging here would
// cost 4*4096 against the 2*2048 + 2*4096 the separate pools use, so it is not
// free and a zero budget declines it.
// OFF-COUNT-6: memref.alloc() {{.*}}address_space
// OFF-NOT: memref.alloc() {{.*}}address_space

// OFFTYPE-DAG: memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
// OFFTYPE-DAG: memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
// OFFTYPE-DAG: flag = 5
// Nothing merged, so no union view and no back-edge release.
// OFFTYPE-NOT: flag = 6
// OFFTYPE-NOT: memref.reinterpret_cast {{.*}}offset: [0]{{.*}}address_space<ub>{{.*}}address_space<ub>

// 4096 is exactly the difference between the two separate pools and four union
// slots, so it buys the merge.  The allocation count is unchanged -- four UB
// slots replace 2 + 2, and the L1 pool is untouched because promotion is off --
// but every UB slot is now owned by one lane.
// UNION-COUNT-6: memref.alloc() {{.*}}address_space
// UNION-NOT: memref.alloc() {{.*}}address_space

// Each UB slot is the larger of the two roles; the smaller role's own type is
// never allocated, only viewed.
// UNIONVIEW-COUNT-4: memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
// UNIONVIEW-NOT: memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
// UNIONVIEW-NOT: memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
// The view is contiguous -- offset 0 and unit-stride sizes -- because a vector
// function's parameters bufferize with an identity layout map and a strided
// window could never be cast across that boundary.
// UNIONVIEW-DAG: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [16, 32], strides: [32, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<16x32xf32, #hivm.address_space<ub>>
// Four QK + four PV forward flags, two L1 flags, one back-edge release: 0..10.
// UNIONVIEW-DAG: flag = 10
// UNIONVIEW-NOT: flag = 11
// The release must not introduce a reverse-pipe channel; both directions stay canonical.
// UNIONVIEW-NOT: sync_block{{.*}}<PIPE_V>, <PIPE_FIX>
// UNIONVIEW-NOT: sync_block{{.*}}<PIPE_MTE1>, <PIPE_MTE3>

// Opting into pool promotion additionally takes the L1 pool to one slot per
// lane.  It is free against a UB budget, so the only thing it spends is flags.
// 4 UB + 4 L1 = 8.
// PROMOTE-COUNT-8: memref.alloc() {{.*}}address_space
// PROMOTE-NOT: memref.alloc() {{.*}}address_space

// PROMOTETYPE-COUNT-4: memref.alloc() : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
// PROMOTETYPE-NOT: memref.alloc() : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
// Two more flags than the union alone: 0..12.
// PROMOTETYPE-DAG: flag = 12
// PROMOTETYPE-NOT: flag = 13

// Eight lanes would need eight forward flags for each merged role plus the
// release -- thirteen more than the six the rotating pools use, against the
// fifteen available.  An unlimited UB budget must not talk the merge into
// overrunning the flag budget: it declines, and the L1 pool (free in UB and
// still affordable in flags) promotes to eight instead.  2 + 2 + 8 = 12.
// FLAGCAP-COUNT-12: memref.alloc() {{.*}}address_space
// FLAGCAP-NOT: memref.alloc() {{.*}}address_space

// FLAGCAPTYPE-COUNT-8: memref.alloc() : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
// FLAGCAPTYPE-NOT: memref.alloc() : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
// FLAGCAPTYPE-DAG: flag = 11
// FLAGCAPTYPE-NOT: flag = 12
// The merge was declined, so no union view was emitted.
// FLAGCAPTYPE-NOT: memref.reinterpret_cast {{.*}}offset: [0]{{.*}}address_space<ub>{{.*}}address_space<ub>

// At two lanes the merge is cheaper than the pools it replaces -- two slots of
// 4096 against 2*2048 + 2*4096 -- so it happens on a zero budget.  Two UB slots
// and two L1 slots: four allocations, down from six.
// NOREUSE-COUNT-4: memref.alloc() {{.*}}address_space
// NOREUSE-NOT: memref.alloc() {{.*}}address_space

// NOREUSETYPE-COUNT-2: memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
// NOREUSETYPE-NOT: memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
// NOREUSETYPE-NOT: memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
// Two QK + two PV forward flags, two L1 flags, one release: 0..6.
// NOREUSETYPE-DAG: flag = 6
// NOREUSETYPE-NOT: flag = 7
