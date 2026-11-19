// RUN: triton-opt %s | FileCheck %s
// Input: LLVM 23 memory (3 bbargs, unused init). Output: LLVM 20 long form (2 bbargs).
// CHECK-LABEL: func.func @map_long
// CHECK: %{{.*}} = linalg.map
// CHECK-SAME: ins(%{{.*}} : tensor<64xf32>, tensor<64xf32>)
// CHECK-SAME: outs(%{{.*}} : tensor<64xf32>)
// CHECK: (%{{.*}}: f32, %{{.*}}: f32) {
// CHECK: arith.addf
// CHECK: linalg.yield
// CHECK: return %{{.*}} : tensor<64xf32>
// CHECK-NOT: %{{.*}}: f32, %{{.*}}: f32, %{{.*}}: f32)

func.func @map_long(%lhs: tensor<64xf32>, %rhs: tensor<64xf32>, %init: tensor<64xf32>) -> tensor<64xf32> {
  %0 = linalg.map
      ins(%lhs, %rhs : tensor<64xf32>, tensor<64xf32>)
      outs(%init : tensor<64xf32>)
      (%a: f32, %b: f32, %init_arg: f32) {
        %c = arith.addf %a, %b : f32
        %d = arith.addf %c, %c : f32
        linalg.yield %d : f32
      }
  return %0 : tensor<64xf32>
}
