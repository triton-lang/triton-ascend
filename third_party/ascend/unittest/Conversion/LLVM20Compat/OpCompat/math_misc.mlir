// RUN: triton-opt %s | FileCheck %s

// CHECK-LABEL: func.func @math_misc() -> f32 {
// CHECK-NEXT: %[[F0:.*]] = arith.constant 1.250000e+00 : f32
// CHECK-NEXT: %[[F1:.*]] = arith.constant 2.500000e+00 : f32
// CHECK-NEXT: %[[F2:.*]] = arith.constant 3.750000e+00 : f32
// CHECK-NEXT: %[[I0:.*]] = arith.constant -7 : i32
// CHECK-NEXT: %[[ABSF:.*]] = math.absf %[[F0]] : f32
// CHECK-NEXT: %[[ABSI:.*]] = math.absi %[[I0]] : i32
// CHECK-NEXT: %[[CEIL:.*]] = math.ceil %[[F0]] : f32
// CHECK-NEXT: %[[COS:.*]] = math.cos %[[F0]] : f32
// CHECK-NEXT: %[[ERF:.*]] = math.erf %[[F0]] : f32
// CHECK-NEXT: %[[EXP:.*]] = math.exp %[[F0]] : f32
// CHECK-NEXT: %[[FLOOR:.*]] = math.floor %[[F0]] : f32
// CHECK-NEXT: %[[FMA:.*]] = math.fma %[[F0]], %[[F1]], %[[F2]] : f32
// CHECK-NEXT: %[[LOG:.*]] = math.log %[[F1]] : f32
// CHECK-NEXT: %[[RSQRT:.*]] = math.rsqrt %[[F1]] : f32
// CHECK-NEXT: %[[SIN:.*]] = math.sin %[[F0]] : f32
// CHECK-NEXT: %[[SQRT:.*]] = math.sqrt %[[F1]] : f32
// CHECK-NEXT: %[[TANH:.*]] = math.tanh %[[F0]] : f32
// CHECK: return %{{.*}} : f32
func.func @math_misc() -> f32 {
  %f0 = arith.constant 1.250000e+00 : f32
  %f1 = arith.constant 2.500000e+00 : f32
  %f2 = arith.constant 3.750000e+00 : f32
  %i0 = arith.constant -7 : i32
  %absf = math.absf %f0 : f32
  %absi = math.absi %i0 : i32
  %ceil = math.ceil %f0 : f32
  %cos = math.cos %f0 : f32
  %erf = math.erf %f0 : f32
  %exp = math.exp %f0 : f32
  %floor = math.floor %f0 : f32
  %fma = math.fma %f0, %f1, %f2 : f32
  %log = math.log %f1 : f32
  %rsqrt = math.rsqrt %f1 : f32
  %sin = math.sin %f0 : f32
  %sqrt = math.sqrt %f1 : f32
  %tanh = math.tanh %f0 : f32
  %sum0 = arith.addf %absf, %ceil : f32
  %sum1 = arith.addf %cos, %erf : f32
  %sum2 = arith.addf %exp, %floor : f32
  %sum3 = arith.addf %fma, %log : f32
  %sum4 = arith.addf %rsqrt, %sin : f32
  %sum5 = arith.addf %sqrt, %tanh : f32
  %sum6 = arith.addf %sum0, %sum1 : f32
  %sum7 = arith.addf %sum2, %sum3 : f32
  %sum8 = arith.addf %sum4, %sum5 : f32
  %sum9 = arith.addf %sum6, %sum7 : f32
  %out = arith.addf %sum9, %sum8 : f32
  return %out : f32
}
