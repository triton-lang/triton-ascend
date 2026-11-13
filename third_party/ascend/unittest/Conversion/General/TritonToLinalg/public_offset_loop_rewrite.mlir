// RUN: triton-opt %s --triton-to-linalg --verify-each --split-input-file | FileCheck %s

// A loop-external yielded tensor must not make its scalar fields get inserted
// after the loop. The verifier checks dominance through the rebuilt body.
// CHECK-LABEL: func.func @external_backedge
// CHECK: scf.for
// CHECK-NOT: tensor<8xi32>
// CHECK: memref.copy
// CHECK: return
module {
  tt.func public @external_backedge(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %step = arith.constant dense<8> : tensor<8xi32>
    %external = arith.addi %range, %step : tensor<8xi32>
    %base = tt.splat %src : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %out = tt.splat %dst : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %outptr = tt.addptr %out, %range : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%offset = %range) -> tensor<8xi32> {
      %ptr = tt.addptr %base, %offset : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
      %value = tt.load %ptr : tensor<8x!tt.ptr<f32>>
      tt.store %outptr, %value : tensor<8x!tt.ptr<f32>>
      scf.yield %external : tensor<8xi32>
    }
    tt.return
  }
}

// -----

// Both carried tensors participate in the next origin/stride. Mapping a
// sampled origin back to a reconstructed tensor would retain tensor carriers.
// CHECK-LABEL: func.func @coupled_offsets
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK-NOT: iter_args({{.*}}tensor<8xi32>
// CHECK: memref.copy
// CHECK: arith.addi
// CHECK: scf.yield
// CHECK-SAME: index
// CHECK: return
module {
  tt.func public @coupled_offsets(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %two = arith.constant dense<2> : tensor<8xi32>
    %step = arith.constant dense<8> : tensor<8xi32>
    %stride2 = arith.muli %range, %two : tensor<8xi32>
    %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %dsts = tt.splat %dst : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %result:2 = scf.for %i = %c0 to %n step %c1 iter_args(%x = %range, %y = %stride2) -> (tensor<8xi32>, tensor<8xi32>) {
      %p = tt.addptr %srcs, %x : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
      %q = tt.addptr %dsts, %y : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
      %value = tt.load %p : tensor<8x!tt.ptr<f32>>
      tt.store %q, %value : tensor<8x!tt.ptr<f32>>
      %nextx = arith.addi %x, %y : tensor<8xi32>
      %nexty = arith.addi %y, %step : tensor<8xi32>
      scf.yield %nextx, %nexty : tensor<8xi32>, tensor<8xi32>
    }
    tt.return
  }
}
