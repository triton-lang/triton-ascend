// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// A canonical SIMT scope must select both the mixed BiShengIR pipeline and
// the SIMT-aware runtime launch path.
// CHECK-LABEL: func.func @simt_scope
// CHECK-SAME: parallel_mode = "mix_simd_simt"
tt.func public @simt_scope(%arg0: !tt.ptr<f32>) {
  scope.scope : () -> () {
    scope.return
  } {vec_mode = "simt"}
  tt.return
}
