// RUN: triton-opt --process-args --split-input-file --verify-diagnostics %s -o /dev/null

func.func @payload_capture_is_not_private_control(%limit: i32) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  // expected-error@+1 {{outer event control update captures a payload loop argument}}
  %r:4 = scf.for %iv = %zero to %limit step %one
      iter_args(%task = %zero, %j = %zero, %phase = %zero, %payload = %zero)
      -> (i32, i32, i32, i32) : i32 {
    %next = arith.addi %task, %payload {ssbuffer.block_id = 1 : i32} : i32
    scf.yield %next, %j, %phase, %payload : i32, i32, i32, i32
  } {ssbuffer.main_loop = 0 : i32,
                ssbuffer.outer_event_control_count = 3 : i32}
  return
}

// -----

func.func @mixed_payload_control_if(%limit: i32, %data: tensor<1xf32>) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %r:3 = scf.for %iv = %zero to %limit step %one
      iter_args(%task = %zero, %j = %zero, %phase = %zero)
      -> (i32, i32, i32) : i32 {
    %active = arith.cmpi eq, %phase, %zero {ssbuffer.block_id = 1 : i32} : i32
    // expected-error@+1 {{outer event control must have a separate scalar update graph containing only pure operations and metadata loads}}
    %mixed:2 = scf.if %active -> (i32, tensor<1xf32>) {
      scf.yield %task, %data : i32, tensor<1xf32>
    } else {
      scf.yield %j, %data : i32, tensor<1xf32>
    } {ssbuffer.block_id = 1 : i32}
    scf.yield %mixed#0, %j, %phase : i32, i32, i32
  } {ssbuffer.main_loop = 0 : i32,
                ssbuffer.outer_event_control_count = 3 : i32}
  return
}
