// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=AUTO
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=AUTOVIEW
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4 private-buffer-ub-budget-bytes=0" 2>/dev/null | FileCheck %s --check-prefix=DECLINED
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4 private-buffer-ub-budget-bytes=0" 2>/dev/null | FileCheck %s --check-prefix=DECLINEDTYPE
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=8 promote-private-buffer-pools=true" 2>/dev/null | FileCheck %s --check-prefix=FLAGCAP
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=8 promote-private-buffer-pools=true" 2>/dev/null | FileCheck %s --check-prefix=FLAGCAPTYPE
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" 2>/dev/null | FileCheck %s --check-prefix=NOREUSE
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" 2>/dev/null | FileCheck %s --check-prefix=NOREUSETYPE
//
// A transfer slot owned by one unrolled lane is never reused, so it has no
// write-after-read pair to order and the producing core cannot overwrite what
// the consumer is still reading.  The schedule then has nothing to interleave
// against either, so CUBE issues every score matmul before its first wait
// instead of stopping after `inter_core_buf_count` of them.
//
// The cheapest way to buy that is to merge the two CUBE->VECTOR roles onto one
// union slot per lane: a lane's score tile is dead by the time its product tile
// is written, so one buffer serves both.  The slot is sized for the larger role
// and the smaller one takes a contiguous view of its front, which is what lets
// the two roles differ in size.
//
// How much UB is free is not knowable here -- this pass runs before
// bufferization, and the backend's memory planner is what assigns addresses and
// reports an overflow with the exact requirement.  So the default is to spend
// what the schedule asks for and let that planner arbitrate.  A caller who
// knows better can cap it, and zero declines any spend.
//
// The candidate has three phases: QK (UB, 16x32xf32 = 2048 bytes a slot),
// P (L1/cbuf, free against a UB budget) and PV (UB, 16x64xf32 = 4096 bytes).
//
// Buffer counts and flag counts are checked on separate prefixes from the
// buffer *types*: the allocations of the three pools interleave in the output,
// so a COUNT/NOT pair and a type-matching DAG group cannot share a prefix.

// By default the merge is taken even though the roles differ in size: four
// union slots of 4096 replace 2*2048 + 2*4096, which costs 4096 more.  With the
// four L1 slots the VECTOR->CUBE pool already has, that is eight allocations,
// every one owned by a single lane.
// AUTO-COUNT-8: memref.alloc() {{.*}}address_space
// AUTO-NOT: memref.alloc() {{.*}}address_space

// Each UB slot is the larger of the two roles; the smaller role's own type is
// never allocated, only viewed.
// AUTOVIEW-COUNT-4: memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
// AUTOVIEW-NOT: memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
// AUTOVIEW-NOT: memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
// The view is contiguous -- offset 0 and unit-stride sizes -- because a vector
// function's parameters bufferize with an identity layout map and a strided
// window could never be cast across that boundary.
// AUTOVIEW-DAG: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [16, 32], strides: [32, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<16x32xf32, #hivm.address_space<ub>>
// Four QK + four PV forward flags, four L1 flags, one back-edge release: 0..12.
// AUTOVIEW-DAG: flag = 12
// AUTOVIEW-NOT: flag = 13
// The release must not introduce a reverse-pipe channel; both stay canonical.
// AUTOVIEW-NOT: sync_block{{.*}}<PIPE_V>, <PIPE_FIX>
// AUTOVIEW-NOT: sync_block{{.*}}<PIPE_MTE1>, <PIPE_MTE3>

// A budget of zero declines the spend: the CUBE->VECTOR pools keep the
// inter-core buffer count and rotate, and the schedule goes back to pipelining
// at that depth to order the reuse.  2 + 2 UB and 4 L1.
// DECLINED-COUNT-8: memref.alloc() {{.*}}address_space
// DECLINED-NOT: memref.alloc() {{.*}}address_space

// DECLINEDTYPE-DAG: memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
// DECLINEDTYPE-DAG: memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
// Two QK forward flags, four P, two PV: 0..7.
// DECLINEDTYPE-DAG: flag = 7
// DECLINEDTYPE-NOT: flag = 8
// Nothing merged, so no union view.
// DECLINEDTYPE-NOT: memref.reinterpret_cast {{.*}}address_space<ub>{{.*}}address_space<ub>

// Eight lanes would need eight forward flags for each merged role plus the
// release, against the fifteen available.  An unbounded budget must not talk
// the merge into overrunning the flag budget: it declines regardless of UB, and
// the L1 pool -- free in UB and still affordable in flags -- goes to eight
// instead.  2 + 2 + 8 = 12.
// FLAGCAP-COUNT-12: memref.alloc() {{.*}}address_space
// FLAGCAP-NOT: memref.alloc() {{.*}}address_space

// FLAGCAPTYPE-COUNT-8: memref.alloc() : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
// FLAGCAPTYPE-NOT: memref.alloc() : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
// FLAGCAPTYPE-DAG: flag = 11
// FLAGCAPTYPE-NOT: flag = 12
// The merge was declined, so no union view was emitted.
// FLAGCAPTYPE-NOT: memref.reinterpret_cast {{.*}}address_space<ub>{{.*}}address_space<ub>

// At two lanes the merge is cheaper than the pools it replaces -- two slots of
// 4096 against 2*2048 + 2*4096 -- so it happens even on a zero budget.  Two UB
// slots and two L1 slots: four allocations, down from six.
// NOREUSE-COUNT-4: memref.alloc() {{.*}}address_space
// NOREUSE-NOT: memref.alloc() {{.*}}address_space

// NOREUSETYPE-COUNT-2: memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
// NOREUSETYPE-NOT: memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
// NOREUSETYPE-NOT: memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
// Two QK + two PV forward flags, two L1 flags, one release: 0..6.
// NOREUSETYPE-DAG: flag = 6
// NOREUSETYPE-NOT: flag = 7
