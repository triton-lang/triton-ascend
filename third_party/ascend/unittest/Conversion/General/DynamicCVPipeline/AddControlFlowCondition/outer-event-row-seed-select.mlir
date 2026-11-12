// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s --check-prefix=ROW
// RUN: sed 's/4xf32/4x4xf32/g' %s | triton-opt --add-control-flow-condition | FileCheck %s --check-prefix=TILE
// RUN: sed 's|// ADD-EFFECT: ||' %s | triton-opt --add-control-flow-condition | FileCheck %s --check-prefix=EFFECT

// Row seeds select exact bits; full tiles and effectful seeds keep their guard.
// ROW-LABEL: func.func @seeds
// ROW: arith.constant dense<0xFF800000> : tensor<4xf32>
// ROW: tensor.splat
// ROW: arith.select
// ROW: arith.constant dense<0.000000e+00> : tensor<4xf32>
// ROW: tensor.splat
// ROW: arith.select
// ROW-NOT: scf.if
// ROW: return
// TILE-LABEL: func.func @seeds
// TILE: scf.if
// TILE: linalg.fill
// TILE: } else
// TILE: return
// EFFECT-LABEL: func.func @seeds
// EFFECT: scf.if
// EFFECT: memref.store
// EFFECT: linalg.fill
// EFFECT: } else
// EFFECT: return
module attributes {ssbuffer.outer_main_loop, hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @seeds(%first: i1, %old_m: tensor<4xf32>, %old_l: tensor<4xf32>,
                   %out_m: memref<4xf32>, %out_l: memref<4xf32>, %n: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %minus_inf = arith.constant 0xFF800000 : f32
    %zero_float = arith.constant 0.0 : f32
    scope.scope : () -> () {
      scf.for %i = %zero to %n step %one {
        %rows:2 = scf.if %first -> (tensor<4xf32>, tensor<4xf32>) {
          // ADD-EFFECT: memref.store %zero_float, %out_m[%zero] : memref<4xf32>
          %m = tensor.empty() : tensor<4xf32>
          %initial_m = linalg.fill ins(%minus_inf : f32) outs(%m : tensor<4xf32>) -> tensor<4xf32>
          %l = tensor.empty() : tensor<4xf32>
          %initial_l = linalg.fill ins(%zero_float : f32) outs(%l : tensor<4xf32>) -> tensor<4xf32>
          scf.yield %initial_m, %initial_l : tensor<4xf32>, tensor<4xf32>
        } else {
          scf.yield %old_m, %old_l : tensor<4xf32>, tensor<4xf32>
        }
        hivm.hir.copy ins(%rows#0 : tensor<4xf32>) outs(%out_m : memref<4xf32>)
        hivm.hir.copy ins(%rows#1 : tensor<4xf32>) outs(%out_l : memref<4xf32>)
      } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
