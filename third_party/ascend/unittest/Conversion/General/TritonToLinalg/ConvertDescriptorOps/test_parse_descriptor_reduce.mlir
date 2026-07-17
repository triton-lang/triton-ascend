// RUN: triton-opt --convert-descriptor-ops --split-input-file %s | FileCheck %s

// CHECK-LABEL: tt.func public @kernel
// CHECK-NOT: tt.descriptor_reduce
// CHECK: %[[MASK:.*]] = arith.andi %{{.*}}, %{{.*}} : tensor<2x16xi1>
// CHECK: tt.atomic_rmw add, acq_rel, gpu, %{{.*}}, %{{.*}}, %[[MASK]] : (tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>, tensor<2x16xi1>) -> tensor<2x16xi32>

module {
  tt.func public @kernel(%out_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %val: tensor<2x16xi32>, %M: i32, %N: i32 {tt.divisibility = 16 : i32}, %moffset: i32, %noffset: i32) attributes {noinline = false} {
    %desc = arith.constant 1 : i64
    %desc_14 = arith.extsi %N : i32 to i64
    %desc_15 = tt.make_tensor_descriptor %out_ptr, [%M, %N], [%desc_14, %desc] : <i32>, <tensor<2x16xsi32>>
    tt.descriptor_reduce add, %desc_15[%moffset, %noffset], %val : !tt.tensordesc<tensor<2x16xsi32>>, tensor<2x16xi32>
    tt.return
  }
}
