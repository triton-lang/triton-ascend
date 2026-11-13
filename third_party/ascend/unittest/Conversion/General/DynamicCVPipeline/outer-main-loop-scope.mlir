// RUN: sed 's/ {ssbuffer.outer_main_loop}//g' %s | triton-opt -o %t.original
// RUN: triton-opt %s --normalize-outer-main-loop --merge-outer-vector-block --reorder-outer-event-carry -o %t.outer
// RUN: diff %t.original %t.outer

// A hint is necessary but not sufficient: scalar, Vector-only and Cube-only
// nests must retain exactly the original IR. A hint in a sibling function must
// not enable outer scheduling for an unmarked mixed CV nest.
func.func @scalar_only(%n: index, %initial: i32, %delta: i32) -> i32 {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %result = scf.for %task = %zero to %n step %one iter_args(%state = %initial) -> i32 {
    %next = scf.for %j = %zero to %two step %one iter_args(%value = %state) -> i32 {
      %sum = arith.addi %value, %delta : i32
      scf.yield %sum : i32
    }
    scf.yield %next : i32
  } {ssbuffer.outer_main_loop}
  return %result : i32
}

func.func @vector_only(%n: index, %a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %result = scf.for %task = %zero to %n step %one iter_args(%state = %a) -> tensor<4x4xf32> {
    %next = scf.for %j = %zero to %n step %one iter_args(%value = %state) -> tensor<4x4xf32> {
      %sum = arith.addf %value, %a : tensor<4x4xf32>
      scf.yield %sum : tensor<4x4xf32>
    }
    scf.yield %next : tensor<4x4xf32>
  } {ssbuffer.outer_main_loop}
  return %result : tensor<4x4xf32>
}

func.func @cube_only(%n: index, %a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %result = scf.for %task = %zero to %n step %one iter_args(%state = %a) -> tensor<4x4xf32> {
    %next = scf.for %j = %zero to %n step %one iter_args(%value = %state) -> tensor<4x4xf32> {
      %product = linalg.matmul ins(%a, %a : tensor<4x4xf32>, tensor<4x4xf32>)
          outs(%value : tensor<4x4xf32>) -> tensor<4x4xf32>
      scf.yield %product : tensor<4x4xf32>
    }
    scf.yield %next : tensor<4x4xf32>
  } {ssbuffer.outer_main_loop}
  return %result : tensor<4x4xf32>
}

func.func @unmarked_mixed_cv(%n: index, %a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %result = scf.for %task = %zero to %n step %one iter_args(%state = %a) -> tensor<4x4xf32> {
    %next = scf.for %j = %zero to %n step %one iter_args(%value = %state) -> tensor<4x4xf32> {
      %product = linalg.matmul ins(%a, %a : tensor<4x4xf32>, tensor<4x4xf32>)
          outs(%a : tensor<4x4xf32>) -> tensor<4x4xf32>
      %sum = arith.addf %value, %product : tensor<4x4xf32>
      scf.yield %sum : tensor<4x4xf32>
    }
    scf.yield %next : tensor<4x4xf32>
  }
  return %result : tensor<4x4xf32>
}

// Without an outer mainloop marker, nested loops, tensor recurrences and
// memory operations must pass through the new scheduling passes unchanged.
func.func @unmarked_loop_nest(%a: tensor<128x64xbf16>,
                             %b: tensor<64x128xbf16>,
                             %initial: tensor<128x128xf32>,
                             %source: memref<64xf32>,
                             %dest: memref<64xf32>, %count: index)
    -> tensor<128x128xf32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %result = scf.for %i = %zero to %count step %one
      iter_args(%carry = %initial) -> tensor<128x128xf32> {
    memref.copy %source, %dest : memref<64xf32> to memref<64xf32>
    %next = scf.for %j = %zero to %count step %one
        iter_args(%tile = %carry) -> tensor<128x128xf32> {
      %product = linalg.matmul ins(%a, %b : tensor<128x64xbf16>, tensor<64x128xbf16>)
          outs(%tile : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield %product : tensor<128x128xf32>
    }
    scf.yield %next : tensor<128x128xf32>
  }
  return %result : tensor<128x128xf32>
}
