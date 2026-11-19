// RUN: triton-opt %s --separate-cv-scope --verify-diagnostics -o /dev/null

// A marker is not permission to duplicate a payload effect in both engines.
module attributes {ssbuffer.outer_main_loop} {
func.func @reject_marked_store(%out: memref<1xi32>, %condition: i1, %value: i32) {
  %zero = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
  // expected-error@+1 {{outer event control marker requires scalar operations and read-only metadata loads}}
  scf.if %condition {
    memref.store %value, %out[%zero] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
  } {ssbuffer.core_type = "VECTOR", ssbuffer.outer_event_control}
  return
}
}
