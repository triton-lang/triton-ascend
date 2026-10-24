// RUN: env -u TRITON_DISABLE_LINE_INFO triton-opt --triton-to-linalg --mlir-print-debuginfo --split-input-file %s 2>&1 | FileCheck %s --implicit-check-not="all have no location"
// RUN: env TRITON_DISABLE_LINE_INFO=1 triton-opt --triton-to-linalg --mlir-print-debuginfo --split-input-file %s 2>&1 | FileCheck %s --check-prefix=DISABLED --implicit-check-not="all have no location"

// With real source information present, an unknown producer still inherits
// its user's file/line/column. Skipping all location recovery would be wrong.
// CHECK-LABEL: func.func @recover_line_info
// CHECK: arith.addi {{.*}} loc(#[[USER_LOC:loc[0-9]+]])
// CHECK: arith.muli {{.*}} loc(#[[USER_LOC]])
// CHECK: #[[USER_LOC]] = loc("recover.py":42:7)
// DISABLED: #[[$UNKNOWN:loc[0-9]*]] = loc(unknown)
// DISABLED-LABEL: func.func @recover_line_info
// DISABLED: arith.addi {{.*}} loc(#[[$UNKNOWN]])
// DISABLED: arith.muli {{.*}} loc(#[[KEPT:loc[0-9]+]])
// DISABLED: #[[KEPT]] = loc("recover.py":42:7)
module {
  tt.func @recover_line_info(%arg0: i32 loc(unknown),
                            %arg1: i32 loc(unknown)) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32 loc(unknown)
    %1 = arith.muli %0, %arg0 : i32 loc("recover.py":42:7)
    tt.return %1 : i32 loc("recover.py":43:3)
  } loc("recover.py":1:1)
} loc("recover.py":1:1)

// -----

// Real locations nested inside both a name and a fused location must count as
// source information. Use two distinct locations so the fused node survives.
// CHECK: #[[FIRST:loc[0-9]*]] = loc("nested.py":64:9)
// CHECK: #[[SECOND:loc[0-9]*]] = loc("nested.py":65:3)
// CHECK: #[[FUSED:loc[0-9]*]] = loc(fused[#[[FIRST]], #[[SECOND]]])
// CHECK: #[[$WRAPPED:loc[0-9]*]] = loc("wrapper"(#[[FUSED]]))
// CHECK-LABEL: func.func @recover_nested_line_info
// CHECK: arith.addi {{.*}} loc(#[[$WRAPPED]])
// CHECK: arith.muli {{.*}} loc(#[[$WRAPPED]])
// DISABLED: #[[$NESTED_UNKNOWN:loc[0-9]*]] = loc(unknown)
// DISABLED: #[[$NESTED_WRAPPED:loc[0-9]*]] = loc("wrapper"(
// DISABLED-LABEL: func.func @recover_nested_line_info
// DISABLED: arith.addi {{.*}} loc(#[[$NESTED_UNKNOWN]])
// DISABLED: arith.muli {{.*}} loc(#[[$NESTED_WRAPPED]])
module {
  tt.func @recover_nested_line_info(%arg0: i32 loc(unknown),
                                   %arg1: i32 loc(unknown)) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32 loc(unknown)
    %1 = arith.muli %0, %arg0 : i32 loc("wrapper"(fused["nested.py":64:9, "nested.py":65:3]))
    tt.return %1 : i32 loc("wrapper"(fused["nested.py":64:9, "nested.py":65:3]))
  } loc("wrapper"(fused["nested.py":64:9, "nested.py":65:3]))
} loc("wrapper"(fused["nested.py":64:9, "nested.py":65:3]))
