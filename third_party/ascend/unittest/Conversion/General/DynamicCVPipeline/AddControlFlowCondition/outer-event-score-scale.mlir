// RUN: triton-opt --split-input-file --add-control-flow-condition %s | FileCheck %s

// The selected branch executes one unchanged score multiplication.
// CHECK-LABEL: func.func @sink_score
// CHECK: scf.for
// CHECK-NOT: arith.mulf
// CHECK: scf.if
// CHECK-NEXT: %[[THEN:.*]] = arith.mulf
// CHECK-NEXT: {{.*}}linalg.reduce ins(%[[THEN]]
// CHECK: } else {
// CHECK-NEXT: %[[ELSE:.*]] = arith.mulf
// CHECK-NEXT: {{.*}}linalg.reduce ins(%[[ELSE]]
module attributes {ssbuffer.outer_main_loop, hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @sink_score(%condition: i1, %score: tensor<4x4xf32>,
                   %scale: tensor<4x4xf32>, %init: tensor<4xf32>,
                   %out: memref<4xf32>, %extra: memref<4x4xf32>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scope.scope : () -> () {
      scf.for %iv = %zero to %one step %one {
        %scaled = arith.mulf %score, %scale : tensor<4x4xf32>
        %result = scf.if %condition -> tensor<4xf32> {
          %sum = linalg.reduce ins(%scaled : tensor<4x4xf32>) outs(%init : tensor<4xf32>) dimensions = [1]
            (%in: f32, %acc: f32) {
              %value = arith.addf %in, %acc : f32
              linalg.yield %value : f32
            }
          scf.yield %sum : tensor<4xf32>
        } else {
          %sum = linalg.reduce ins(%scaled : tensor<4x4xf32>) outs(%init : tensor<4xf32>) dimensions = [1]
            (%in: f32, %acc: f32) {
              %value = arith.maximumf %in, %acc : f32
              linalg.yield %value : f32
            }
          scf.yield %sum : tensor<4xf32>
        }

        hivm.hir.copy ins(%result : tensor<4xf32>) outs(%out : memref<4xf32>)
      } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}

// -----
// A use after the branch prevents sinking.
// CHECK-LABEL: func.func @keep_external_use
// CHECK: scf.for
// CHECK: %[[SCALE:.*]] = arith.mulf
// CHECK: scf.if
// CHECK-NOT: arith.mulf
// CHECK: arith.addf %[[SCALE]]
module attributes {ssbuffer.outer_main_loop, hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @keep_external_use(%condition: i1, %score: tensor<4x4xf32>,
                   %scale: tensor<4x4xf32>, %init: tensor<4xf32>,
                   %out: memref<4xf32>, %extra: memref<4x4xf32>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scope.scope : () -> () {
      scf.for %iv = %zero to %one step %one {
        %scaled = arith.mulf %score, %scale : tensor<4x4xf32>
        %result = scf.if %condition -> tensor<4xf32> {
          %sum = linalg.reduce ins(%scaled : tensor<4x4xf32>) outs(%init : tensor<4xf32>) dimensions = [1]
            (%in: f32, %acc: f32) {
              %value = arith.addf %in, %acc : f32
              linalg.yield %value : f32
            }
          scf.yield %sum : tensor<4xf32>
        } else {
          %sum = linalg.reduce ins(%scaled : tensor<4x4xf32>) outs(%init : tensor<4xf32>) dimensions = [1]
            (%in: f32, %acc: f32) {
              %value = arith.maximumf %in, %acc : f32
              linalg.yield %value : f32
            }
          scf.yield %sum : tensor<4xf32>
        }
        %outside = arith.addf %scaled, %score : tensor<4x4xf32>
        hivm.hir.copy ins(%outside : tensor<4x4xf32>) outs(%extra : memref<4x4xf32>)
        hivm.hir.copy ins(%result : tensor<4xf32>) outs(%out : memref<4xf32>)
      } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
