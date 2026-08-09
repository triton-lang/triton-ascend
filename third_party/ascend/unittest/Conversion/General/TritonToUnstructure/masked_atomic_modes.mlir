// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=False force-simt-template=False' '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' %s | FileCheck %s --check-prefix=FALLBACK
// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=False force-simt-template=True' '--triton-to-unstructure=compile-on-910-95=False force-simt-template=True' %s | FileCheck %s --check-prefix=FALLBACK
// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=True force-simt-template=False' '--triton-to-unstructure=compile-on-910-95=True force-simt-template=False' %s | FileCheck %s --check-prefix=FALLBACK
// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=True force-simt-template=True' '--triton-to-unstructure=compile-on-910-95=True force-simt-template=True' %s | FileCheck %s --check-prefix=A5-SIMT

// All non-pure-SIMT fallback modes must guard the actual atomic operation.
// A false lane therefore reaches the scalar atomic only through scf.if.
// FALLBACK-LABEL: tt.func @masked_atomic_mode_matrix
// FALLBACK-NOT: hivm.hir.custom
// FALLBACK: scf.if
// FALLBACK: tt.atomic_rmw add, acq_rel, gpu, {{.*}} {DiscreteMemAccess} : (tensor<1x!tt.ptr<i32>>, tensor<1xi32>) -> tensor<1xi32>

// The A5 SIMT-template fast path keeps the complete mask as the fourth
// indirect-atomic operand; it does not replace the mask with an identity
// update.
// A5-SIMT-LABEL: tt.func @masked_atomic_mode_matrix
// A5-SIMT: hivm.hir.custom{{.*}}"__builtin_indirect_atomic"
// A5-SIMT-SAME: tensor<16xi8>
// A5-SIMT-NOT: scf.if
tt.func @masked_atomic_mode_matrix(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst = arith.constant dense<8> : tensor<16xi32>
  %cst_0 = arith.constant dense<2> : tensor<16xi32>
  %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %1 = arith.muli %0, %cst_0 : tensor<16xi32>
  %2 = arith.cmpi slt, %1, %cst : tensor<16xi32>
  %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %4 = tt.addptr %3, %1 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %6 = tt.addptr %5, %1 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  %7 = tt.load %6 : tensor<16x!tt.ptr<i32>>
  %8 = tt.atomic_rmw add, acq_rel, gpu, %4, %7, %2 : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>, tensor<16xi1>) -> tensor<16xi32>
  tt.return
}
