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

// -----

// The inner loop samples the outer tensor before the outer loop is rewritten.
// That earlier sample must reuse the new scalar origin rather than retaining
// a fill/range/add/extract chain just to recover lane zero.
// CHECK-LABEL: func.func @nested_offset_samples
// CHECK-NOT: tensor<4xi32>
// CHECK: scf.for
// CHECK-NOT: tensor<4xi32>
// CHECK: scf.for
// CHECK-NOT: tensor<4xi32>
// CHECK: memref.copy
// CHECK-NOT: tensor<4xi32>
// CHECK: return
module {
  tt.func public @nested_offset_samples(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: index, %bound: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %step = arith.constant dense<4> : tensor<4xi32>
    %limit = tt.splat %bound : i32 -> tensor<4xi32>
    %other = arith.constant dense<0.0> : tensor<4xf32>
    %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %dsts = tt.splat %dst : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %outer = scf.for %i = %c0 to %n step %c1 iter_args(%x = %range) -> tensor<4xi32> {
      %mask = arith.cmpi slt, %x, %limit : tensor<4xi32>
      %inner = scf.for %j = %c0 to %n step %c1 iter_args(%y = %x) -> tensor<4xi32> {
        %p = tt.addptr %srcs, %y : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
        %q = tt.addptr %dsts, %y : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
        %value = tt.load %p, %mask, %other : tensor<4x!tt.ptr<f32>>
        tt.store %q, %value, %mask : tensor<4x!tt.ptr<f32>>
        %nexty = arith.addi %y, %step : tensor<4xi32>
        scf.yield %nexty : tensor<4xi32>
      }
      %nextx = arith.addi %x, %step : tensor<4xi32>
      scf.yield %nextx : tensor<4xi32>
    }
    tt.return
  }
}

// -----

// An existing zero sample uses the original i32 width, even when its scalar
// origin is carried as index. Nonzero/dynamic coordinates and samples in a
// nested block must retain their own values instead of becoming the origin.
// CHECK-LABEL: func.func @sample_width_and_coordinates
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: scf.for
// CHECK: %[[NARROW:.*]] = arith.index_cast %{{.*}} : index to i32
// CHECK: %[[WIDE:.*]] = arith.extsi %[[NARROW]] : i32 to i64
// CHECK: %[[WIDE_PACK:.*]] = tensor.insert %[[WIDE]] into
// CHECK: bufferization.materialize_in_destination %[[WIDE_PACK]] in writable
// CHECK: %[[NONZERO:.*]] = tensor.extract %[[TENSOR:.*]][%[[ONE]]] : tensor<4xi32>
// CHECK: %[[NONZERO_PACK:.*]] = tensor.insert %[[NONZERO]] into
// CHECK: bufferization.materialize_in_destination %[[NONZERO_PACK]] in writable
// CHECK: %[[DYNAMIC:.*]] = tensor.extract %[[TENSOR]][%[[COORDINATE:.*]]] : tensor<4xi32>
// CHECK: %[[DYNAMIC_PACK:.*]] = tensor.insert %[[DYNAMIC]] into
// CHECK: bufferization.materialize_in_destination %[[DYNAMIC_PACK]] in writable
// CHECK: scf.if
// CHECK: %[[NESTED_COORDINATE:.*]] = arith.addi %[[COORDINATE]], %[[ONE]] : index
// CHECK: %[[NESTED:.*]] = tensor.extract %[[TENSOR]][%[[NESTED_COORDINATE]]] : tensor<4xi32>
// CHECK: %[[NESTED_PACK:.*]] = tensor.insert %[[NESTED]] into
// CHECK: bufferization.materialize_in_destination %[[NESTED_PACK]] in writable
// CHECK: return
module {
  tt.func public @sample_width_and_coordinates(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %wide_out: !tt.ptr<i64>, %sample_out: !tt.ptr<i32>, %start: i32, %n: index, %coordinate: index, %condition: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %start_tensor = tt.splat %start : i32 -> tensor<4xi32>
    %initial = arith.addi %range, %start_tensor : tensor<4xi32>
    %step = arith.constant dense<4> : tensor<4xi32>
    %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %dsts = tt.splat %dst : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%x = %initial) -> tensor<4xi32> {
      %p = tt.addptr %srcs, %x : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %q = tt.addptr %dsts, %x : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %value = tt.load %p : tensor<4x!tt.ptr<f32>>
      tt.store %q, %value : tensor<4x!tt.ptr<f32>>
      %zero = tensor.extract %x[%c0] : tensor<4xi32>
      %wide = arith.extsi %zero : i32 to i64
      tt.store %wide_out, %wide : !tt.ptr<i64>
      %nonzero = tensor.extract %x[%c1] : tensor<4xi32>
      tt.store %sample_out, %nonzero : !tt.ptr<i32>
      %dynamic = tensor.extract %x[%coordinate] : tensor<4xi32>
      tt.store %sample_out, %dynamic : !tt.ptr<i32>
      scf.if %condition {
        %nested_coordinate = arith.addi %coordinate, %c1 : index
        %nested = tensor.extract %x[%nested_coordinate] : tensor<4xi32>
        tt.store %sample_out, %nested : !tt.ptr<i32>
      }
      %next = arith.addi %x, %step : tensor<4xi32>
      scf.yield %next : tensor<4xi32>
    }
    tt.return
  }
}

// -----

// The condition edge advances the offset, so before/after samples must use
// their respective region origins. Neither sample needs tensor rebuilding.
// CHECK-LABEL: func.func @while_origin_samples
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK: scf.while (%[[BEFORE_ITER:.*]] = %{{.*}}, %[[BEFORE_ORIGIN:.*]] = %{{.*}}) : (i32, index) -> (i32, index) {
// CHECK-NOT: tensor.extract
// CHECK: %[[BEFORE_I32:.*]] = arith.index_cast %[[BEFORE_ORIGIN]] : index to i32
// CHECK-NOT: tensor.extract
// CHECK: %[[BEFORE_I32_PACK:.*]] = tensor.insert %[[BEFORE_I32]] into
// CHECK: bufferization.materialize_in_destination %[[BEFORE_I32_PACK]] in writable
// CHECK-NOT: tensor.extract
// CHECK: %[[FORWARDED:.*]] = arith.addi %[[BEFORE_ORIGIN]], %[[C4]] : index
// CHECK-NOT: tensor.extract
// CHECK: scf.condition(%{{.*}}) %[[BEFORE_ITER]], %[[FORWARDED]] : i32, index
// CHECK-NOT: tensor.extract
// CHECK: ^bb0(%[[AFTER_ITER:.*]]: i32, %[[AFTER_ORIGIN:.*]]: index):
// CHECK-NOT: tensor.extract
// CHECK: %[[AFTER_I32:.*]] = arith.index_cast %[[AFTER_ORIGIN]] : index to i32
// CHECK-NOT: tensor.extract
// CHECK: %[[AFTER_I32_PACK:.*]] = tensor.insert %[[AFTER_I32]] into
// CHECK: bufferization.materialize_in_destination %[[AFTER_I32_PACK]] in writable
// CHECK-NOT: tensor.extract
// CHECK: %[[NEXT_ORIGIN:.*]] = arith.addi %[[AFTER_ORIGIN]], %[[C4]] : index
// CHECK-NOT: tensor.extract
// CHECK: scf.yield %{{.*}}, %[[NEXT_ORIGIN]] : i32, index
// CHECK-NOT: tensor.extract
// CHECK: return
module {
  tt.func public @while_origin_samples(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %before_out: !tt.ptr<i32>, %after_out: !tt.ptr<i32>, %n: i32) {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %step = arith.constant dense<4> : tensor<4xi32>
    %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %dsts = tt.splat %dst : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %result:2 = scf.while (%x = %range, %i = %zero) : (tensor<4xi32>, i32) -> (tensor<4xi32>, i32) {
      %before = tensor.extract %x[%c0] : tensor<4xi32>
      tt.store %before_out, %before : !tt.ptr<i32>
      %forwarded = arith.addi %x, %step : tensor<4xi32>
      %continue = arith.cmpi slt, %i, %n : i32
      scf.condition(%continue) %forwarded, %i : tensor<4xi32>, i32
    } do {
    ^bb0(%y: tensor<4xi32>, %j: i32):
      %after = tensor.extract %y[%c0] : tensor<4xi32>
      tt.store %after_out, %after : !tt.ptr<i32>
      %p = tt.addptr %srcs, %y : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %q = tt.addptr %dsts, %y : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %value = tt.load %p : tensor<4x!tt.ptr<f32>>
      tt.store %q, %value : tensor<4x!tt.ptr<f32>>
      %nexty = arith.addi %y, %step : tensor<4xi32>
      %nextj = arith.addi %j, %one : i32
      scf.yield %nexty, %nextj : tensor<4xi32>, i32
    }
    tt.return
  }
}
