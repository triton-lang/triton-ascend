// RUN: env TRITON_DISABLE_LINE_INFO=1 triton-opt --triton-to-linalg --mlir-print-debuginfo --split-input-file %s 2>&1 | FileCheck %s --implicit-check-not="all have no location"
// RUN: env -u TRITON_DISABLE_LINE_INFO triton-opt --triton-to-linalg --mlir-print-debuginfo --split-input-file %s 2>&1 | FileCheck %s --implicit-check-not="all have no location"

// Missing file/line information must not trigger a search for every operation,
// whether line information was explicitly disabled or absent from the input.
// CHECK-LABEL: func.func @disabled_line_info
// CHECK: arith.addi
// CHECK: return
module {
  tt.func @disabled_line_info(%arg0: i32 loc(unknown),
                             %arg1: i32 loc(unknown)) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32 loc(unknown)
    tt.return %0 : i32 loc(unknown)
  } loc(unknown)
} loc(unknown)

// -----

// A synthetic name is not a user file/line location either.
// CHECK: #[[$SYNTHETIC_UNKNOWN:loc[0-9]*]] = loc(unknown)
// CHECK-LABEL: func.func @synthetic_location_only
// CHECK: arith.addi {{.*}} loc(#[[$SYNTHETIC_UNKNOWN]])
// CHECK: return
module {
  tt.func @synthetic_location_only(%arg0: i32 loc(unknown),
                                  %arg1: i32 loc(unknown)) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32 loc(unknown)
    tt.return %0 : i32 loc("synthetic_helper")
  } loc("synthetic_function")
} loc("synthetic_module")
