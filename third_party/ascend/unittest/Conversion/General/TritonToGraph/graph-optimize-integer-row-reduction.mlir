// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=262144' | FileCheck %s --check-prefixes=CHECK,F64
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=16384' | FileCheck %s --check-prefixes=CHECK,F16
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=1024' | FileCheck %s --check-prefix=KEEP --implicit-check-not=hacc.program_grid_transforms
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 compile-mode=simt_only' | FileCheck %s --check-prefix=KEEP --implicit-check-not=hacc.program_grid_transforms

// Batch independent integer rows; preserve reduction axis and tail protection.
// F64: axis = 1 : i64, factor = 64 : i64
// F16: axis = 1 : i64, factor = 16 : i64
// CHECK-LABEL: tt.func public @native_rows(
// CHECK: %[[TAIL:.*]] = arith.cmpi ult, {{.*}} {IATRuntimeExtentUnsignedMask}
// CHECK: arith.shrsi
// CHECK: arith.andi
// CHECK: arith.addi
// CHECK: arith.shrsi
// CHECK: %[[INACTIVE:.*]] = arith.cmpi eq, %[[TAIL]],
// CHECK-NEXT: %[[SAFE:.*]] = arith.ori {{.*}}, %[[INACTIVE]]
// CHECK-NEXT: tt.assert %[[SAFE]]
// CHECK: %[[COUNT:.*]] = "tt.reduce"({{.*}}) <{axis = 1 : i32}>
// CHECK: tt.assert {{.*}} {tt.auto_overflow_assert} : i1
// CHECK: (tensor<[[F:64|16]]x16xi32>) -> tensor<[[F]]xi32>
// CHECK: tt.store {{.*}}, {{.*}}, {{.*}} : tensor<[[F]]x16x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, %[[COUNT]], %[[TAIL]] : tensor<[[F]]x!tt.ptr<i32>>
// KEEP: "tt.reduce"({{.*}}) <{axis = 0 : i32}>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @native_rows(%data: !tt.ptr<i32>, %counts: !tt.ptr<i32>, %lengths: !tt.ptr<i32>) {
    %zero = arith.constant 0 : i32
    %width = arith.constant 16 : i32
    %divisor = arith.constant 128 : i32
    %pid = tt.get_program_id y : i32
    %lp = tt.addptr %lengths, %pid : !tt.ptr<i32>, i32
    %length = tt.load %lp : !tt.ptr<i32>
    %limit = arith.divsi %length, %divisor : i32
    %ok = arith.cmpi sge, %length, %zero : i32
    tt.assert %ok, "row" {tt.auto_overflow_assert} : i1
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %base = arith.muli %pid, %width : i32
    %row = tt.splat %base : i32 -> tensor<16xi32>
    %index = arith.addi %row, %r : tensor<16xi32>
    %dp = tt.splat %data : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %ptr = tt.addptr %dp, %index : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %input = tt.load %ptr : tensor<16x!tt.ptr<i32>>
    %limits = tt.splat %limit : i32 -> tensor<16xi32>
    %valid = arith.cmpi slt, %input, %limits : tensor<16xi32>
    %bits = arith.extui %valid : tensor<16xi1> to tensor<16xi32>
    %count = "tt.reduce"(%bits) <{axis = 0 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %sum = arith.addi %a, %b : i32
      %safe = arith.cmpi sge, %sum, %zero : i32
      tt.assert %safe, "combine" {tt.auto_overflow_assert} : i1
      tt.reduce.return %sum : i32
    }) : (tensor<16xi32>) -> i32
    tt.store %ptr, %bits : tensor<16x!tt.ptr<i32>>
    %cp = tt.addptr %counts, %pid : !tt.ptr<i32>, i32
    tt.store %cp, %count : !tt.ptr<i32>
    tt.return
  }
}
