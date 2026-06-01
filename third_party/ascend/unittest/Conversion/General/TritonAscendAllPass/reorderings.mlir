// RUN: TRITON_DEBUG=1 triton-opt --triton-to-structured '--discrete-mask-access-conversion=compile-on-910-95=False force-simt-template=False' '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation --triton-to-structured --triton-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func  @vector_kernel(
#loc = loc("/workspace/story-locations/device_print_3.py":9:0)
#loc16 = loc("y"(#loc))
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @vector_kernel(%y: f32 loc("y"(#loc))) attributes {noinline = false} {
    // CHECK-NEXT: %cst = arith.constant 3.14159274 : f32
    // CHECK-NEXT: %0 = tensor.empty() : tensor<1xf32>
    // CHECK-NEXT: %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    // CHECK-NEXT: %c0 = arith.constant 0 : index
    %cst = arith.constant 3.14159274 : f32 loc(#loc1)
    // CHECK-NEXT:  %c16_i32 = arith.constant 16 : i32
    %c16_i32 = arith.constant 16 : i32 loc(#loc2)
    // CHECK-NEXT:  call @triton_print_0(%arg2, %c16_i32) : (f32, i32) -> ()
    tt.print " y and 16: " {hex = true, isSigned = array<i32: 0, 1>} : %y, %c16_i32 : f32, i32 loc(#loc2)
    // CHECK-NEXT:  %2 = tensor.empty() : tensor<1xf32>
    // CHECK-NEXT:  %c0_0 = arith.constant 0 : index
    // CHECK-NEXT:  %inserted = tensor.insert %arg2 into %2[%c0_0] : tensor<1xf32>
    // CHECK-NEXT:  %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%inserted : tensor<1xf32>) outs(%inserted : tensor<1xf32>) {
    // CHECK-NEXT:  ^bb0(%in: f32, %out: f32):
    // CHECK-NEXT:  %17 = math.exp %in : f32
    %y_0 = math.exp %y : f32 loc(#loc17)
    %z = arith.mulf %y_0, %cst : f32 loc(#loc18)
    // CHECK:  call @triton_print_1
    tt.print " z: " {hex = false, isSigned = array<i32: 0>} : %z : f32 loc(#loc5)
    // CHECK-NEXT:  %6 = tensor.empty() : tensor<1xf32>
    // CHECK:  %7 = tensor.empty() : tensor<1xf32>
    // CHECK-NEXT:  %c0_6 = arith.constant 0 : index
    // CHECK-NEXT:  %inserted_7 = tensor.insert %extracted_3 into %7[%c0_6] : tensor<1xf32>
    // CHECK-NEXT:  %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%inserted_5, %inserted_7 : tensor<1xf32>, tensor<1xf32>) outs(%inserted_5 : tensor<1xf32>) {
    // CHECK-NEXT:  ^bb0(%in: f32, %in_22: f32, %out: f32):
    // CHECK-NEXT:    %17 = arith.addf %in, %in_22 : f32
    // CHECK-NEXT:    linalg.yield %17 : f32
    // CHECK-NEXT:  } -> tensor<1xf32>
    // CHECK-NEXT:  %extracted_8 = tensor.extract %8[%c0] : tensor<1xf32>
    %a = arith.addf %y_0, %z : f32 loc(#loc19)
    // CHECK:  call @triton_print_2
    tt.print " a: " {hex = false, isSigned = array<i32: 0>} : %a : f32 loc(#loc7)
    %a2 = arith.mulf %a, %a : f32 loc(#loc20)
    %a2_1 = math.exp %a : f32 loc(#loc21)
    %a2_2 = arith.addf %a2, %a2_1 : f32 loc(#loc22)
    tt.print " y: " {hex = false, isSigned = array<i32: 0>} : %y_0 : f32 loc(#loc11)
    tt.print " z: " {hex = false, isSigned = array<i32: 0>} : %z : f32 loc(#loc12)
    tt.print " a: " {hex = false, isSigned = array<i32: 0>} : %a : f32 loc(#loc13)
    tt.print " a2: " {hex = false, isSigned = array<i32: 0>} : %a2_2 : f32 loc(#loc14)
    tt.return loc(#loc15)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/workspace/story-locations/device_print_3.py":12:35)
#loc3 = loc("/workspace/story-locations/device_print_3.py":13:15)
#loc4 = loc("/workspace/story-locations/device_print_3.py":14:18)
#loc5 = loc("/workspace/story-locations/device_print_3.py":15:27)
#loc6 = loc("/workspace/story-locations/device_print_3.py":16:12)
#loc7 = loc("/workspace/story-locations/device_print_3.py":17:27)
#loc8 = loc("/workspace/story-locations/device_print_3.py":18:13)
#loc9 = loc("/workspace/story-locations/device_print_3.py":18:24)
#loc10 = loc("/workspace/story-locations/device_print_3.py":18:17)
#loc11 = loc("/workspace/story-locations/device_print_3.py":19:27)
#loc12 = loc("/workspace/story-locations/device_print_3.py":20:27)
#loc13 = loc("/workspace/story-locations/device_print_3.py":21:27)
#loc14 = loc("/workspace/story-locations/device_print_3.py":22:28)
#loc15 = loc("/workspace/story-locations/device_print_3.py":22:4)
#loc17 = loc("y"(#loc3))
#loc18 = loc("z"(#loc4))
#loc19 = loc("a"(#loc6))
#loc20 = loc("a2"(#loc8))
#loc21 = loc("a2"(#loc9))
#loc22 = loc("a2"(#loc10))
