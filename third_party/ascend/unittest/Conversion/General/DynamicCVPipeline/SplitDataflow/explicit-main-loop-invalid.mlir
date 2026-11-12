// RUN: triton-opt --mark-main-loop --split-input-file --verify-diagnostics %s

module attributes {ssbuffer.outer_main_loop} {
func.func @unfulfilled_request(%lb: index, %ub: index, %step: index) {
  // expected-error @+1 {{outer main-loop request was not normalized before main-loop selection}}
  scf.for %i = %lb to %ub step %step {
  } {ssbuffer.outer_main_loop}
  return
}
}

// -----

module attributes {ssbuffer.outer_main_loop} {
func.func @nested_explicit_domains(%lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
    // expected-error @+1 {{nested explicit main-loop selections are unsupported}}
    scf.for %j = %lb to %ub step %step {
    } {ssbuffer.explicit_main_loop}
  } {ssbuffer.explicit_main_loop}
  return
}
}
